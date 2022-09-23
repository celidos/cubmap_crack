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

import torch.cuda.amp as amp
is_amp = True

from tools import *
import models.load_model as modelloader
import criterions as crits
import validation as valid_stuff
import augmentations.augmentations_base as augbase
from datasets.base_dataset import base_dataset_read, get_dataloaders
import postprocess
from logger import TrainLogger
import augmentations.augmentations_base as augbase
from models.load_model import load_model


def make_swa_coat_a(args):
    print('COAT SWA depth 4 organ token.')
    print('Warning! This function may not be appliable to other models! Beware of shared weights!')
    OUTPUT_PTH = os.path.join(args['output_folder'], './checkpoint_fold_{}'.format(args['current_fold']))
    ARTIFACTS_PTH = os.path.join(OUTPUT_PTH, 'artifacts/')
    Path(OUTPUT_PTH).mkdir(parents=True, exist_ok=True)
    
    model = load_model(args)
    
    swa_model = AveragedModel(model)
    
    chkpts = []
    fnames = list(sorted(glob(os.path.join(OUTPUT_PTH, 'coat-small_ep_*_dice_*.pt'))))
    fnames = [x for x in fnames if ('BEST' not in x.upper() and 'SWA' not in x.upper() and 'LAST' not in x.upper())]
    for fname in fnames:
    #     print(fname)
        bname = os.path.basename(fname)
        spt = bname.split('_')
        epoch = int(spt[2])
        dice = float(spt[4].rsplit('.', maxsplit=1)[0])
        print(epoch, dice)
        chkpts.append((fname, dice, epoch))
    
    top_n = args['swa_top_n']
    device = args['swa_device']
     
    print('-'*80)
    print("keeping top {} checkpoints...".format(top_n))
        
    chkpts = list(sorted(chkpts, key=lambda x: x[1]))
    for el in chkpts[-top_n:]:
        print(el)
    
    resp = model.load_state_dict(torch.load(chkpts[-1][0], map_location=device))
    print(resp)
    
    swa_model = AveragedModel(model)
    
    for el in chkpts[-top_n:]:
        model.load_state_dict(torch.load(el[0], map_location=device))
        swa_model.update_parameters(model)
        
    # restoring some layers
    model.load_state_dict(torch.load(chkpts[-1][0], map_location=device))
    
    swa_state_dict = swa_model.state_dict()
    
    for key in model.state_dict().keys():
        if 'num_batches_tracked' in key or \
            '.cpe.' in key or \
            '.crpe.' in key:
            print('replaced:', key)
            swa_state_dict['module.' + key] = model.state_dict()[key]
    swa_model.load_state_dict(swa_state_dict)
    
    print('-'*80)
    print('Fixing batch norm...')
    
    print('  Loading dataset')
    swa_model = swa_model.to(device)
    
    train_df, train_df_orig = base_dataset_read(args)
    val_df_id = train_df_orig[train_df_orig['fold'] == args['current_fold']]
    loader_train, loader_val = get_dataloaders(args, train_df, for_swa=True)
   
    print('  Updating BN')   
    torch.optim.swa_utils.update_bn(loader_train, swa_model)
    
    print('  Saving swa checkpoint')   
    
    torch.save(swa_model.state_dict(), os.path.join(
                    OUTPUT_PTH,
                    'coat-small-organ-token-fold{}_SWA_TOP_{}_BEST.pt'.format(args['current_fold'], top_n)
                )) 


def make_swa_coat_unet(args):
    print('COAT SWA depth 4 organ token.')
    print('Warning! This function may not be appliable to other models! Beware of shared weights!')
    OUTPUT_PTH = os.path.join(args['output_folder'], './checkpoint_fold_{}'.format(args['current_fold']))
    ARTIFACTS_PTH = os.path.join(OUTPUT_PTH, 'artifacts/')
    Path(OUTPUT_PTH).mkdir(parents=True, exist_ok=True)
    
    model = load_model(args)
    
    swa_model = AveragedModel(model)
    
    chkpts = []
    fnames = list(sorted(glob(os.path.join(OUTPUT_PTH, 'coat-small-unet_ep_*_dice_*.pt'))))
    fnames = [x for x in fnames if ('BEST' not in x.upper() and 'SWA' not in x.upper() and 'LAST' not in x.upper())]
    for fname in fnames:
    #     print(fname)
        bname = os.path.basename(fname)
        spt = bname.split('_')
        epoch = int(spt[2])
        dice = float(spt[4].rsplit('.', maxsplit=1)[0])
        print(epoch, dice)
        chkpts.append((fname, dice, epoch))
    
    top_n = args['swa_top_n']
    device = args['swa_device']
     
    print('-'*80)
    print("keeping top {} checkpoints...".format(top_n))
        
    chkpts = list(sorted(chkpts, key=lambda x: x[1]))
    for el in chkpts[-top_n:]:
        print(el)
    
    resp = model.load_state_dict(torch.load(chkpts[-1][0], map_location=device))
    print(resp)
    
    swa_model = AveragedModel(model)
    
    for el in chkpts[-top_n:]:
        model.load_state_dict(torch.load(el[0], map_location=device))
        swa_model.update_parameters(model)
        
    # restoring some layers
    model.load_state_dict(torch.load(chkpts[-1][0], map_location=device))
    
    swa_state_dict = swa_model.state_dict()
    
    for key in model.state_dict().keys():
        if 'num_batches_tracked' in key or \
            '.cpe.' in key or \
            '.crpe.' in key or \
            'encoder' in key:
            print('replaced:', key)
            swa_state_dict['module.' + key] = model.state_dict()[key]
    swa_model.load_state_dict(swa_state_dict)
    
    print('-'*80)
    print('Fixing batch norm...')
    
    print('  Loading dataset')
    swa_model = swa_model.to(device)
    
    train_df, train_df_orig = base_dataset_read(args)
    val_df_id = train_df_orig[train_df_orig['fold'] == args['current_fold']]
    loader_train, loader_val = get_dataloaders(args, train_df, for_swa=True)
   
    print('  Updating BN')   
    torch.optim.swa_utils.update_bn(loader_train, swa_model)
    
    print('  Saving swa checkpoint')   
    
    torch.save(swa_model.state_dict(), os.path.join(
                    OUTPUT_PTH,
                    'coat-small-organ-token-fold{}_SWA_TOP_{}_BEST.pt'.format(args['current_fold'], top_n)
                )) 
