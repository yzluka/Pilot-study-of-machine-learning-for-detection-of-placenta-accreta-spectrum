import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from multiprocessing import get_context
from functools import partial

def sample_from_patient_labeled(patient_id, stratify_set, root = '/data/LAIPAS_Images/'):
    resize, color_jitter, equalize, flip = True, True,False,True
    resize_factor = (0.5, 2-0.5)
    crop_size= 512
    patient_frame = np.random.choice(stratify_set[patient_id])
    img = Image.open(root + 'img/' + patient_frame)
    msk = Image.open(root + 'msks/' + patient_frame)
    
    if resize:
        factor = np.random.rand()*resize_factor[1]+resize_factor[0]
        w,h = img.size
        w = int(w*factor)
        h = int(h*factor)
        
        img = transforms.Resize((w,h), interpolation='bicubic')(img)
        msk = transforms.Resize((w,h), interpolation='nearest')(msk)
        
    if equalize:
        trans = transforms.RandomEqualize(p=0.5)
        img = trans(img)
        
    if color_jitter:
        trans = transforms.ColorJitter(brightness=0.1, contrast=0.1)
        img = trans(img)
    
    diff_h = max(0, crop_size - h)
    diff_w = max(0, crop_size - w)
    
    img = transforms.Pad((diff_w//2+1,diff_h//2+1),
                         fill=0, padding_mode='symmetric')(img)
    msk = transforms.Pad((diff_w//2+1,diff_h//2+1),
                         fill=255, padding_mode='constant')(msk)
    
    top = np.random.choice(img.height - crop_size+1)
    left= np.random.choice(img.width  - crop_size+1)
    img = transforms.functional.crop(img, top, left, crop_size,crop_size)
    msk = transforms.functional.crop(msk, top, left, crop_size,crop_size)
    trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0,1)]) 
    
    return (trans(img), torch.tensor(np.array(msk),dtype=torch.long))

def parallel_sample_labeled(setname='train_stratify.npy', batch_size=16):
    batch_size=16
    n_workers=16

    my_set = np.load(setname,allow_pickle=True).item()
    patients = np.random.choice(list(my_set.keys()),batch_size)
    my_trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0,1)]) 
    with get_context("spawn").Pool() as p:
        temp = p.map(partial(sample_from_patient_labeled,stratify_set=my_set), patients)

    imgs = torch.stack([item[0] for item in temp])
    msks = torch.stack([item[1] for item in temp])
    return imgs, msks
if __name__ == '__main__':
    X,y = parallel_sample_labeled()