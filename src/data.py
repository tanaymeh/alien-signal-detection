import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config
from .augmentations import Augments


class SETIData(Dataset):
    def __init__(self, images, targets, is_test=False, augmentations=None): 
        self.images = images
        self.targets = targets
        self.is_test = is_test
        self.augmentations = augmentations
        
    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        
        img = np.load(img)
        img = np.vstack(img)
        img = img.transpose(1, 0)
        img = img.astype("float")[..., np.newaxis]
        
        if self.augmentations:
            img = self.augmentations(image=img)['image']
        
        if self.is_test:
            return img
        
        else:
            target = self.targets[index]
            return img, target
    
    def __len__(self):
        return len(self.images)