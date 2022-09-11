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

import pickle

class TrainLogger:
    def __init__(self, path: str, args):
        self.path = path
        self.args = args
        self.train_loss = []
        self.val_epoch_dice = [] 
        
    def log_train_loss(self, it, loss):
        self.train_loss.append((it, loss))
        
    def log_train_lr(self, it, lr):
        self.train_loss.append((it, lr))
        
    def log_val_dice(self, epoch, metrics):
        self.val_epoch_dice.append((epoch, metrics))
        
    def dump(self):
        with open(os.path.join(self.path, 'history.pkl'), 'wb') as f:
            pickle.dump(self, f)