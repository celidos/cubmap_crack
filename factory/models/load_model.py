import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv
import os

# +
from matplotlib import pyplot as plt
from glob import glob
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from IPython.display import clear_output

from collections import OrderedDict
# -

import albumentations as A
import albumentations.pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR


def load_model(args, log=None):
    if args['model']['encoder'] == 'coat_parallel_small_organ_token' and \
       args['model']['decoder'] == 'daformer_3x3_conv':
        from models.coat_daformer_organ_token.coat import coat_parallel_small
        from models.coat_daformer_organ_token.daformer import daformer_conv3x3
        from models.coat_daformer_organ_token.model_coat_daformer import Net
        encoder = coat_parallel_small
        decoder = daformer_conv3x3

        model = Net(encoder=encoder, decoder=decoder, n_classes=args['model']['n_classes']).to(args['device'])
        if log is not None:
            log(str(model))
        return model
    elif args['model']['encoder'] == 'coat_parallel_small_organ_token' and \
       args['model']['decoder'] == 'unet_decoder':
        from models.coat_daformer_organ_token.coat import coat_parallel_small
        from models.coat_daformer_organ_token.daformer import daformer_conv3x3
        from models.coat_unet_decoder.model_coat_unet import Net, UnetDecoder
        encoder = coat_parallel_small
        decoder = UnetDecoder

        model = Net(encoder=encoder, decoder=decoder, n_classes=args['model']['n_classes']).to(args['device'])
        if log is not None:
            log(str(model))
        return model
    else:
        raise NotImplementedError("Unknown combination encoder+decoder: `{}` + `{}`".format(args['model']['encoder'], args['model']['decoder']))

def empty_loader(model):
    pass

def load_checkpoint_full(model, path_to_checkpoint: str):
    raise NotImplementedError("Good method, but not implemented!")

def load_encoder_pretrained(model, device:str, path_to_checkpoint: str, strict:bool=True, log=None):
    response = model.load_state_dict(torch.load(path_to_checkpoint, map_location=device), strict=strict)
    if log is None:
        print(response)
    else:
        log(response)

def load_encoder_from_swa_checkpoint(model, device:str, path_to_checkpoint: str, strict:bool=True, log=None, 
                                     ignore_aux_heads=False):
    state_dict = torch.load(path_to_checkpoint)
    state_dict = OrderedDict([
        (k[7:], state_dict[k]) if k.startswith('module.') else (k, v) for k, v in state_dict.items()
    ])
    if ignore_aux_heads:
        state_dict = OrderedDict([
            (k, v) for k, v in state_dict.items() if 'aux_heads_serial' not in k and 'aux_heads_parallel' not in k
        ])
    response = model.load_state_dict(state_dict, strict=False)
    if log is None:
        print(response)
    else:
        log(response)


def get_checkpoint_loader(args):
    if args['model']['checkpoint_preload']['function'] == 'load_checkpoint_full':
        return load_checkpoint_full
    elif args['model']['checkpoint_preload']['function'] == 'load_encoder_pretrained':
        return load_encoder_pretrained
    elif args['model']['checkpoint_preload']['function'] == 'load_encoder_from_swa_checkpoint':
        return load_encoder_from_swa_checkpoint
    else:
        raise NotImplementedError("Unknown load model method: `{}`".format(args['model']['checkpoint_preload']))


def freeze_encoder_a(args, model):
    print('Trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('---------- ENCODER FREEZED! ---------------')
    for param in model.rgb.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False
    print('Trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
