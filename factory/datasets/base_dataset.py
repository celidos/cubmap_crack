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

import augmentations.augmentations_base as augbase
from tools import *


def make_train_image_path(row, train_images_dir: str):
    return os.path.join(train_images_dir, str(row['id']) + '.png')

def make_train_mask_path(row):
    return os.path.join(train_images_dir, str(row['id']) + '.png')

def create_folds(df: pd.DataFrame, n_splits: int, random_seed: int) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["organ"])):
        df.loc[val_idx, "fold"] = int(fold)

    return df

def make_768_dataset(df, args):
    new_df = []
    for index, row in df.iterrows():
        id = row['id']
        glb = list(glob(os.path.join(args['dataset']['train_images_dir'],'{}_*.png'.format(id))))
        for fname in glb:
            newrow = dict()
            newrow['id'] = id
            newrow['image'] = fname
            newrow['mask'] = fname.replace('train_images_', 'train_masks_')
            newrow['organ'] = row['organ']
            newrow['pixel_size'] = row['pixel_size']
            newrow['fold'] = row['fold']
            
            new_df.append(newrow)
    return pd.DataFrame(new_df)

def base_dataset_read(args):
    train_df = pd.read_csv(args['dataset']['train_csv'])
    train_df = create_folds(train_df, n_splits=args['dataset']['n_cross_valid_splits'],
                            random_seed=args['dataset']['random_seed'])
    
    if args['dataset']['function'] == 'make_768_dataset':
        train_df_out = make_768_dataset(train_df, args)
        print(train_df_out.shape)
        print(train_df_out.head())
        if args['debug']:
            return train_df_out.iloc[:100]
        return train_df_out, train_df
    raise NotImplementedError("args['function'] == {}".format(args['dataset']['function']))
    return None


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)
    random.seed(torch_seed + worker_id + 77777)


class HubmapDataset768(Dataset):
    def __init__(self, df, transform=None, for_swa:bool=False, device=None):

        self.df = df
        self.transform = transform
        self.length = len(self.df)
        
        self.for_swa = for_swa
        self.device = device

    def __str__(self):
        string = ''
        string += '\tlen = %d\n' % len(self)

        d = self.df.organ.value_counts().to_dict()
        for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
            string +=  '%24s %3d (%0.3f) \n'%(k,d.get(k,0),d.get(k,0)/len(self.df))
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.for_swa:
            if index % 100 == 0:
                print(index, '/', self.length)
                
        d = self.df.iloc[index]
        organ = ORGAN2ID[d['organ']]

        image = cv.cvtColor(cv.imread(d['image']), cv.COLOR_BGR2RGB) # .astype(np.float32) / 255.0
        mask = cv.imread(d['mask'], cv.IMREAD_GRAYSCALE)

        mask = mask / max(1, mask.max())
        
        mask_multiclass = mask * (organ + 1)
        
        data = {
            'image': image,
            'mask': mask_multiclass,
            'organ': organ,
        }
        
        upd_data = self.transform(image=data['image'], mask=data['mask'])
        data.update(upd_data)
        
        if self.for_swa:
            data['image'] = data['image'].to(self.device)
            data['mask'] = data['mask'].to(self.device)
            
        return data


class HubmapDataset768OrganID(Dataset):
    def __init__(self, df, transform=None, for_swa:bool=False, device=None):

        self.df = df
        self.transform = transform
        self.length = len(self.df)
        self.for_swa = for_swa
        self.device = device

    def __str__(self):
        string = ''
        string += '\tlen = %d\n' % len(self)

        d = self.df.organ.value_counts().to_dict()
        for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
            string +=  '%24s %3d (%0.3f) \n'%(k,d.get(k,0),d.get(k,0)/len(self.df))
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.for_swa:
            if index % 100 == 0:
                print(index, '/', self.length)
        d = self.df.iloc[index]
        organ = ORGAN2ID[d['organ']]

        image = cv.cvtColor(cv.imread(d['image']), cv.COLOR_BGR2RGB) # .astype(np.float32) / 255.0
        mask = cv.imread(d['mask'], cv.IMREAD_GRAYSCALE)

        mask = mask / max(1, mask.max())
        
        mask_multiclass = mask * (organ + 1)
        
        data = {
            'image': image,
            'mask': mask_multiclass,
            'organ': organ,
        }
        
        upd_data = self.transform(image=data['image'], mask=data['mask'], organ_id=organ)  
        data.update(upd_data)
        
        if self.for_swa:
            data['image'] = data['image'].to(self.device)
            data['mask'] = data['mask'].to(self.device)
        
        return data


def get_dataloaders(args, train_df, for_swa:bool = False):
    train_transform = getattr(augbase, args['dataset']['augmentations']['train'])
    val_transform = getattr(augbase, args['dataset']['augmentations']['val'])

    if args['dataset']['dataset_class'] == 'hubmap_p768':
        train_dataset = HubmapDataset768(train_df[train_df['fold'] != args['current_fold']], 
                                         train_transform if not for_swa else val_transform, for_swa, args['swa_device'])
        val_dataset = HubmapDataset768(train_df[train_df['fold'] == args['current_fold']], val_transform)
    elif args['dataset']['dataset_class'] == 'hubmap_p768_organid':
        train_dataset = HubmapDataset768OrganID(train_df[train_df['fold'] != args['current_fold']], 
                                                train_transform if not for_swa else val_transform, for_swa, args['swa_device'])
        val_dataset = HubmapDataset768OrganID(train_df[train_df['fold'] == args['current_fold']], val_transform)
    else:
        raise NotImplementedError("Unknown dataset: `{}`".format(args['dataset']))
    
    loader_params = {'shuffle': True,
                     'num_workers': 0,
                     'worker_init_fn': worker_init_fn}
    loader_train = DataLoader(train_dataset, 
                              batch_size=args['batch_size'], 
                              shuffle=True if not for_swa else False,
                              worker_init_fn=worker_init_fn,
                              num_workers=12 if not for_swa else 0
                             )
    loader_val = DataLoader(val_dataset, 
                            batch_size=1, 
                            shuffle=False,
                            worker_init_fn=worker_init_fn,
                            num_workers=0
                           )
    
    # verbose ---------------------------------------------
    sample = train_dataset[1]
    print('IMAGE')
    print(sample['image'].shape)
    print('image values: ', float(sample['image'].min()), float(sample['image'].max()))
    plt.imshow(sample['image'].detach().cpu().permute((1, 2, 0)))
    plt.show()

    print('MASK')
    print(sample['mask'].shape)
    print('mask values: ', sample['mask'].min(), sample['mask'].max())
    plt.imshow(sample['mask'].detach().cpu())
    plt.show()
    
    sample = val_dataset[2]
    print('IMAGE')
    print(sample['image'].shape)
    print('image values: ', float(sample['image'].min()), float(sample['image'].max()))
    plt.imshow(sample['image'].detach().cpu().permute((1, 2, 0)))
    plt.show()

    print('MASK')
    print(sample['mask'].shape)
    print('mask values: ', sample['mask'].min(), sample['mask'].max())
    plt.imshow(sample['mask'].detach().cpu())
    plt.show()
    
    return loader_train, loader_val

