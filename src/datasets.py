import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    
    def __init__(self, image_dir : str, anno_dir: str, transforms = None):
        self.image_paths = []
        self.mask_paths  = []
        self.transforms = transforms

        for category in os.listdir(image_dir):
            for image_name in os.listdir(os.path.join(image_dir, category)):
                # add image path
                self.image_paths.append(os.path.join(image_dir, category, image_name))
                # add mask path (mask has the same name, but different extension)
                self.mask_paths.append(os.path.join(anno_dir, category, f"{image_name[:-4]}.npy"))
                 
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # read image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # read mask
        mask = np.load(self.mask_paths[idx])
        
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
           
        return image, mask.to(torch.long)


class MultiViewSegmentationDataset(SegmentationDataset):
    
    def __getitem__(self, idx):
        assert self.transforms is not None, "Transforms have to be defined"
        # read image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # read mask
        mask = np.load(self.mask_paths[idx])
        # get augmentation for student
        augmented_s = self.transforms(image=image, mask=mask)
        # get augmentation for teacher
        augmented_t = self.transforms(image=image, mask=mask)
        # extract mask and image
        image_s, mask = augmented_s['image'], augmented_s['mask']
        image_t, _ = augmented_t['image'], augmented_t['mask']
           
        return image_s, image_t, mask.to(torch.long)



        