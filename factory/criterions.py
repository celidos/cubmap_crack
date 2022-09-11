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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

import segmentation_models_pytorch as smp


kl_loss = nn.KLDivLoss(reduction="mean")


def criterion_aux_scaled(preds, masks, organs):
    B = preds.shape[0]
    pred_shape = tuple(np.array(preds.shape[-2:]))
    masks_np = masks.detach().cpu().numpy().astype(np.float32).transpose((1, 2, 0)) # H W B
#     print(masks_np.shape)
    
#     print(masks_np.shape, masks_np.max())
    masks_np = masks_np / (masks_np.max() + 1e-8)
#     plt.figure(figsize=(9, 9))
#     plt.imshow(masks_np[:, :, 0])
#     plt.show()
#     print(masks_np.max())
    
#     print(pred_shape)
    
    masks_np = cv.resize(masks_np, dsize=pred_shape, interpolation=cv.INTER_AREA)
    
    if len(masks_np.shape) < 3:
        masks_np = masks_np[:, :, None]
#     print(masks_np.shape)
    masks_rescaled = torch.tensor(masks_np).permute((2, 0, 1)).unsqueeze(1)
    
    scales = torch.eye(6)[np.array(organs) + 1].unsqueeze(2).unsqueeze(3)
    scales_back = torch.eye(6)[[0] * B].unsqueeze(2).unsqueeze(3)
    
#     print('!!')
#     print(scales.shape)
#     print(scales_back.shape)
#     print(masks_rescaled.shape)
    
#     new_masks = torch.zeros(B, 6, *pred_shape, dtype=torch.float16)
    
    masks_out = (masks_rescaled * scales + (1 - masks_rescaled) * scales_back).to(preds.get_device())
    
    input_aux = F.log_softmax(preds, dim=1).to(preds.get_device())
    loss = kl_loss(input_aux, masks_out)
    
    return loss


ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)

criterion_aux_loss = ce_loss

criterion_tversky_punish_fn = smp.losses.TverskyLoss(
    mode='multiclass', classes=None, log_loss=False, 
    from_logits=True, smooth=1e-6, ignore_index=None, 
    eps=1e-07, alpha=0.3, beta=0.7, gamma=3/4
)

criterion_tversky_punish_fp = smp.losses.TverskyLoss(
    mode='multiclass', classes=None, log_loss=False, 
    from_logits=True, smooth=1e-6, ignore_index=None, 
    eps=1e-07, alpha=0.7, beta=0.3, gamma=3/4
)

criterion_dice = smp.losses.TverskyLoss(
    mode='multiclass', classes=None, log_loss=False, 
    from_logits=True, smooth=0.01, ignore_index=None, 
    eps=1e-07, alpha=0.5, beta=0.5, gamma=1
)

def criterion_image_1(predicts, masks):
    return 0.3 * ce_loss(predicts, masks) + 0.7 * criterion_tversky_punish_fn(predicts, masks)

def criterion_image_2(predicts, masks):
    return 0.3 * ce_loss(predicts, masks) + 0.7 * criterion_tversky_punish_fp(predicts, masks)

def criterion_image_3(predicts, masks):
    return 0.4 * ce_loss(predicts, masks) + 0.6 * criterion_dice(predicts, masks)

def criterion_schedule_1(epoch):
#     if epoch < 40:
    criterion_image = criterion_image_3
    criterion_aux = criterion_aux_scaled
#     elif 40 <= epoch <= 70:
#         criterion_image = criterion_image_1
#         criterion_aux = criterion_aux_loss
#     elif 70 <= epoch <= 80:
#         criterion_image = criterion_image_2
#         criterion_aux = criterion_aux_loss
#     else:
#         criterion_image = criterion_image_3
#         criterion_aux = criterion_aux_loss
        
    return criterion_image, criterion_aux