import os, torch
import numpy as np
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, functional, InterpolationMode
from torchvision import transforms 

class LAIPAS_Dataset(Dataset):

    def __init__(self,data_root, split='train', data_transform=None, 
                 target_transform=None, crop=True, crop_size=512, flip=False,
                 rot = True, resize=True, resize_factor = (0.5,1.2),ignore=True):
        """
        Categories:
            0: background
            'PLACENTA':1, 'BLADDER':2,'UTERUS':3
        
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        assert split in ['train', 'val','unlabeled','train_pat_bal', 'unlabeled_pat_bal','ctl']
        self.data_root = data_root
        self.ignore = ignore
        
        namefiles = open(data_root+f'{split}.txt','r')
        self.filenames = namefiles.read().split('\n')[:-1]
        
        self.data_transform = data_transform
        self.target_transform = target_transform
        
        self.resize = resize if split not in ['val','ctl'] else False
        self.resize_factor = resize_factor
        
        self.crop = crop
        self.crop_size = crop_size
        
        self.flip = flip
        self.rot = rot
        self.split = split
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        msk_type ='msks_aug2'if self.ignore else 'msks'
        idx = torch.tensor(idx).flatten()
        img_name = f'{self.data_root}/img/{self.filenames[idx]}' 
        if not 'unlabeled' in self.split: 
            msk_name = f'{self.data_root}/{msk_type}/{self.filenames[idx]}'
            if 'train' in self.split:
                etp_name = f'{self.data_root}/entropy_msk/{self.filenames[idx]}'
        
        img = Image.open(img_name)
        if not 'unlabeled' in self.split: 
            msk = Image.open(msk_name)
            if 'train' in self.split:
                etp = Image.open(etp_name)
        
        if self.resize:
            factor = torch.rand(1).item()*(self.resize_factor[1]-
                                           self.resize_factor[0]) + self.resize_factor[0]
            w,h = img.size
            w = int(w*factor)
            h = int(h*factor)
        
            img = transforms.Resize((w,h), interpolation=InterpolationMode.BICUBIC)(img)
            if not 'unlabeled' in self.split: 
                msk = transforms.Resize((w,h), interpolation=InterpolationMode.NEAREST)(msk)
                if 'train' in self.split:
                    etp = transforms.Resize((w,h), interpolation=InterpolationMode.BICUBIC)(etp)
            
        if self.crop:
            im_w, im_h = img.size
            diff_w = max(0,self.crop_size-im_w)
            diff_h = max(0,self.crop_size-im_h)
            
            padding = (diff_w//2, diff_h//2, diff_w-diff_w//2, diff_h-diff_h//2)
            img = functional.pad(img, padding, 0, 'symmetric')
            if not 'unlabeled' in self.split: 
                msk = functional.pad(msk, padding, 255, 'constant')
                if 'train' in self.split:
                    etp = functional.pad(etp, padding, 0, 'symmetric')

            t,l,h,w=RandomCrop.get_params(img,(self.crop_size,self.crop_size))

            img = functional.crop(img, t, l, h,w)
            if not 'unlabeled' in self.split: 
                msk = functional.crop(msk, t, l, h,w)
                if 'train' in self.split:
                    etp = functional.crop(etp, t, l, h,w)
        if self.flip:
            if torch.rand(1)>0.5:
                img = functional.rotate(img,180)
                if not 'unlabeled' in self.split: 
                    msk = functional.rotate(msk,180)
                    if 'train' in self.split:
                        etp = functional.rotate(etp,180)
                
        if self.rot:
            angle = torch.rand(1).item()*20-10
            img = functional.rotate(img, angle)
            if not 'unlabeled' in self.split: 
                msk = functional.rotate(msk, angle)
                if 'train' in self.split:
                    etp = functional.rotate(etp, angle)
            
        if self.data_transform:
            img = self.data_transform(img)
        if not 'unlabeled' in self.split: 
            if self.target_transform:
                msk = torch.tensor(np.array(self.target_transform(msk)),dtype=torch.long)
                ignore = msk==255
                msk[ignore]=255
            if 'train' in self.split:
                etp = transforms.ToTensor()(etp)
                return img, msk, etp
            else:
                return img, msk
        else:
            return img, torch.empty(img.shape[-2:],dtype=torch.long)