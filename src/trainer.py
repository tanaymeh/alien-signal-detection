import platform
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import cv2
import gc
import matplotlib.pyplot as plt

import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score

def train_one_epoch(model, device, optimizer, dataloader, loss_fn, scheduler=None):
    """Trains a given model for 1 epoch on the given data

    Args:
        model: Main model
        device: Device on which model will be trained
        optimizer: Optimizer that will optimize during training
        dataloader: Training Dataloader
        loss_fn: Training Loss function. Will be optimized
        scheduler (optional): Scheduler for the learning rate. Defaults to None.
    """
    prog_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    model.train()
    running_loss = 0
    for idx, (img, target) in prog_bar:
        img = img.to(device, torch.float)
        target = target.to(device, torch.float)
        
        output = model(img).view(-1)
        loss = loss_fn(output, target)
        
        # Sending the data from GPU to CPU in a numpy form (using .item()) consumes memory
        # So only do it once
        loss_item = loss.item()
        prog_bar.set_description('loss: {:.2f}'.format(loss_item))
        
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss_item
        
    return running_loss / len(dataloader)

@torch.no_grad()
def valid_one_epoch(model, device, dataloader, loss_fn):
    """Validates the model on the validation set through all batches

    Args:
        model: Main model
        device: Device on which model will be validated
        dataloader: Validation Dataloader
        loss_fn: Validation Loss function. Will NOT be optimized
    """
    prog_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    all_targets, all_predictions = [], []
    running_loss = 0
    model.eval()
    for idx, (img, target) in prog_bar:
        img = img.to(device, torch.float)
        target = target.to(device, torch.float)
        
        output = model(img).view(-1)
        
        loss = loss_fn(output, target)
        loss_item = loss.item()
        
        prog_bar.set_description('val_loss: {:.2f}'.format(loss_item))
        
        all_targets.extend(target.cpu().detach().numpy().tolist())
        all_predictions.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())
        
        running_loss += loss_item
        
    val_roc_auc = roc_auc_score(all_targets, all_predictions)
    return val_roc_auc, running_loss / len(dataloader)