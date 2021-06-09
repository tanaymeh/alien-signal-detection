import platform
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm.notebook import tqdm
import warnings
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

from src.config import Config
from src.augmentations import Augments
from src.data import SETIData
from src.models import *
from src.trainer import train_one_epoch, valid_one_epoch

def yield_loss(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def prepare_data():
    """
    Takes the dataframe and prepares it for training
    """
    train_labels = pd.read_csv(Config.FILE)
    train_labels['path'] = train_labels['id'].apply(lambda x: f'{Config.FOLDER}/{x[0]}/{x}.npy')
    
    return train_labels

def run(device, data):
    kfold = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=2021)
    fold_scores = {}
    for fold_, (trn_idx, val_idx) in enumerate(kfold.split(data, data['target'])):
        print(f"{'='*40} Fold: {fold_} {'='*40}")
        
        train_data = data.loc[trn_idx]
        valid_data = data.loc[val_idx]
        
        print(f"[INFO] Training on {trn_idx.shape[0]} samples and validating on {valid_data.shape[0]} samples")

        # Make Training and Validation Datasets
        training_set = SETIData(
            images=train_data['path'].values,
            targets=train_data['target'].values,
            augmentations=Augments.train_augments
        )

        validation_set = SETIData(
            images=valid_data['path'].values,
            targets=valid_data['target'].values,
            augmentations=Augments.valid_augments
        )
        
        train = DataLoader(
            training_set,
            batch_size=Config.TRAIN_BS,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

        valid = DataLoader(
            validation_set,
            batch_size=Config.VALID_BS,
            shuffle=False,
            num_workers=8
        )
        
        model = VITModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        train_loss_fn = yield_loss()
        valid_loss_fn = yield_loss()
        print(f"[INFO] Training Model: {Config.model_name}")
        
        per_fold_score = []
        best_roc = 0
        
        for epoch in range(1, Config.NB_EPOCHS+1):
            print(f"\n{'--'*5} EPOCH: {epoch} {'--'*5}\n")

            # Train for 1 epoch
            train_loss = train_one_epoch(model, device, optimizer, train, train_loss_fn)
            
            # Validate for 1 epoch
            current_roc, avg_val_loss = valid_one_epoch(model, device, valid, valid_loss_fn)
            print(f"Validation ROC-AUC: {current_roc:.4f}")
            
            per_fold_score.append(current_roc)
            
            if current_roc > best_roc:
                current_roc = best_roc
                torch.save(model.state_dict(), f"{Config.model_name}_fold_{fold_}.pt")
                print(f"Saved best model in this fold with ROC-AUC: {current_roc}")
        
        fold_scores[fold_] = per_fold_score
        
        del training_set, validation_set, train, valid, model, optimizer, current_roc, best_roc
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        DEVICE = torch.device('cuda:0')
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
        DEVICE = torch.device('cpu')
    
    # Get the prepared data
    data = prepare_data()
    
    # Run the training
    run(DEVICE, data)