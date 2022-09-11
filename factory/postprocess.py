# +
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
# -



def keep_top_n_checkpoints_coat_by_ep(args, top_n: int=10):
    path = os.path.join(args['output_folder'], 'checkpoint_fold_{}'.format(args['current_fold']))
    
    chkpts = []
    all_ckpts = list(sorted(glob(path + '/{}_ep_*_dice_*.pt'.format(args['short_name']))))
    all_ckpts = [x for x in all_ckpts 
                 if ('BEST' not in x.upper()) and 
                 ('LAST' not in x.upper()) and
                 ('SWA' not in x.upper())]
    for fname in all_ckpts:
#     print(fname)
        bname = os.path.basename(fname)
        spt = bname.split('_')
        epoch = int(spt[2])
        dice = float(spt[4].rsplit('.', maxsplit=1)[0])
        print(epoch, dice)
        chkpts.append((fname, dice, epoch))
        
    chkpts = list(sorted(chkpts, key=lambda x: x[1]))
    whitelist = [x[0] for x in chkpts[-top_n:]]
    
    for fname in all_ckpts:
        if fname not in whitelist:
            print('To remove:', fname)
            os.remove(fname)
    
    print(whitelist)




