import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv
import os

from matplotlib import pyplot as plt
from glob import glob
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from IPython.display import clear_output

import albumentations as A
import albumentations.pytorch

# +
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

from tools import *


# -


def bindice(p=0.5):
    return np.random.uniform() < p


mean1 = np.array([0.0, 0.0, 0.0])
std1 = np.array([1.0, 1.0, 1.0])


train_transform_a = A.Compose([
#     A.Resize(512, 512, interpolation=cv.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.HueSaturationValue(p=0.25),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=(-0.1, 0.7),rotate_limit=90, p=0.5),
    A.ElasticTransform(p=0.2, alpha=90, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.ElasticTransform(p=0.5, alpha=90, sigma=120*0.7, alpha_affine=120 * 0.8),
    
    A.GridDistortion(p=0.25),
    A.Blur(blur_limit=7, p=0.1),
    A.GaussNoise(var_limit=(20, 100), p=0.4),
    A.ChannelDropout(p=0.05),
    A.RandomGamma(p=0.1),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], 
                max_pixel_value=255, always_apply=True),
    albumentations.pytorch.ToTensorV2()
], p=1)


val_transform_a = A.Compose([
#     A.Resize(512, 512, interpolation = cv.INTER_LINEAR),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1],
                 max_pixel_value=255, always_apply=True),
    albumentations.pytorch.ToTensorV2()
], p=1)

# +
train_transform_b1 = A.Compose([
#     A.Resize(512, 512, interpolation=cv.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=(-0.1, 0.7),rotate_limit=90, p=0.5),
    A.ElasticTransform(p=0.3, alpha=90, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.ElasticTransform(p=0.5, alpha=90, sigma=120*0.7, alpha_affine=120 * 0.8),
    A.ElasticTransform(p=0.1, alpha=7000, sigma=80, alpha_affine=120*0.07),
    
    A.GridDistortion(p=0.25),
    A.Blur(blur_limit=7, p=0.1),
    A.GaussNoise(var_limit=(20, 100), p=0.4),
    A.OneOf([
        A.Cutout(num_holes=5, max_h_size=48, max_w_size=48),
        A.Cutout(num_holes=2, max_h_size=64, max_w_size=64),
        A.Cutout(num_holes=10, max_h_size=32, max_w_size=32),       
    ], p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.PiecewiseAffine(p=0.3),
    ], p=0.2),
    A.ChannelDropout(p=0.05),
    A.RandomGamma(p=0.1),
    A.RandomBrightnessContrast(p=0.2)
])

train_transform_b2 = A.Compose([
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], 
                max_pixel_value=255, always_apply=True),
    albumentations.pytorch.ToTensorV2()
], p=1)


# -

def protate_shakal(image):
    orig_shape = image.shape[:2]
    image = cv.resize(image, None, fx=0.125, fy=0.125)
    image = cv.resize(image, orig_shape)
    return image


def train_transform_b(image, mask, organ_id: int):
    """
    with prostate shakal
    with harden augmentations
    """
    data = dict()
    data['image'] = image
    data['mask'] = mask
    
    if organ_id == ORGAN2ID['prostate']:
        if bindice(0.4):
            data['image'] = protate_shakal(data['image'])
    
    data.update(train_transform_b1(image=data['image'], mask=data['mask']))
    data.update(train_transform_b2(image=data['image'], mask=data['mask']))
      
    return data
