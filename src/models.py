import torch
import timm
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

class VITModel(nn.Module):
    """
    Model Class for VIT Model
    """
    def __init__(self, model_name=Config.model_name, pretrained=True):
        super(VITModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained, in_chans=1)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, Config.LABELS)
    
    def forward(self, x):
        x = self.backbone(x)
        return x
    
class MLPMixer(nn.Module):
    """
    Model Class for MLP Mixer Model
    """
    def __init__(self, model_name=Config.model_name, pretrained=True):
        super(MLPMixer, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained, in_chans=1)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, Config.LABELS)
    
    def forward(self, x):
        x = self.backbone(x)
        return x