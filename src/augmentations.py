from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

from .config import Config

class Augments:
    """
    Contains Train, Validation Augments
    """
    train_augments = Compose([
        Resize(*Config.resize, p=1.0),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=0, value=0, mask_value=0),
        RandomResizedCrop(*Config.resize, p=1.0),
        ToTensorV2(p=1.0),
    ],p=1.)
    
    valid_augments = Compose([
        Resize(*Config.resize, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)