import sys
# sys.path.append('../datasets/')

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

# +
import augmentations.augmentations_base as augbase
from datasets.base_dataset import base_dataset_read, get_dataloaders
from models.load_model import load_model, get_checkpoint_loader
import models.load_model as modelloader
import criterions as crits
import validation as valid_stuff

import postprocess
from logger import TrainLogger
from tools import *
# -


logfile = None


def log(string):
    global logfile
    print(string)
    logfile.write(str(string) + '\n')


def lr_function_a1(step):
    start_lr = 1e-5; min_lr = 1e-5; max_lr = 5e-4    #A
    rampup_epochs = 1500; sustain_epochs = 20000; exp_decay = .99997    #B
 
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs,
           sustain_epochs, exp_decay):
        if epoch < rampup_epochs:    #C
            lr = ((max_lr - start_lr) / rampup_epochs
                        * epoch + start_lr)
        elif epoch < rampup_epochs + sustain_epochs:    #D
            lr = max_lr
        else:    #E
#             lr = max((max_lr - min_lr) *
#                       exp_decay**(epoch - rampup_epochs -
#                                     sustain_epochs)* (0.8+0.00*np.sin(epoch / 100)), 0) + min_lr
            lr = 0.00025
        return lr
 
    return lr(step, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay) / max_lr


def show_lr_function(f):
    xxx = []
    for i in range(30000):
        xxx.append(f(i))

    plt.plot(xxx)
    plt.grid()
    plt.show()

def train_a1(args, mean=np.array([0.0, 0.0, 0.0]), std=np.array([1.0, 1.0, 1.0])):
    global logfile
    
    Path(args['output_folder']).mkdir(parents=True, exist_ok=True)
    OUTPUT_PTH = os.path.join(args['output_folder'], './checkpoint_fold_{}'.format(args['current_fold']))
    ARTIFACTS_PTH = os.path.join(OUTPUT_PTH, 'artifacts/')
    Path(OUTPUT_PTH).mkdir(parents=True, exist_ok=True)
    Path(ARTIFACTS_PTH).mkdir(parents=True, exist_ok=True)

    trlog = TrainLogger(ARTIFACTS_PTH, args)
         
    logfile = open(os.path.join(ARTIFACTS_PTH, 'log_train.txt'), mode='a')
    logfile.write('\n--- [START %s] %s\n\n' % (args['short_name'], '-' * 64))
    
    # preparation ------------------------------------------------------------------
    log(args)
    
    device = args['device']
    
    train_df = base_dataset_read(args)
    
    val_df_id = train_df[train_df['fold'] == args['current_fold']]
    
    loader_train, loader_val = get_dataloaders(args, train_df)
    
    model = load_model(args, log=log)
    
    if args['model']['checkpoint_preload']:
        ckpt_loader = get_checkpoint_loader(args)
        ckpt_loader(model, device=args['device'], log=log, **args['model']['checkpoint_preload']['args'])
    
    for el in args['before_train']:
        f = getattr(modelloader, el)
        f(args, model)
    
    learning_rate = args['base_lr']
    if args['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, **args['optimizer_args'])
    else:
        raise NotImplementedError("args['optimizer'] == {}".format(args['optimizer']))
    
    scaler = amp.GradScaler(enabled = is_amp)
    
    if args['lr_schedule_function'] == 'lr_function_a1':
        lr_function = lr_function_a1
    else:
        raise NotImplementedError("Unknown LR schedule function: {}".format(args['lr_schedule_function']))
        
    show_lr_function(lr_function)

    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_function)
    
    train_loss = []
    accuracy = [] 
    lr_hist = []
    
    criterion_schedule = getattr(crits, args['criterion_schedule'])
    validate = getattr(valid_stuff, args['validation_function'])
    
    # EPOCHS -------------------------------------------------------------------------
    
    global_it = args['start_global_it']
    for epoch in range(args['start_epoch'], args['n_epochs']):
        criterion_image, criterion_aux = criterion_schedule(epoch)

        model.train()

        batch_train_loss = []

        for iteration, batch in enumerate(loader_train):
            global_it += 1

            batch['image'] = batch['image'].half().to(device)
            batch['mask' ] = batch['mask'].to(device, dtype=torch.long)

            with amp.autocast(enabled = is_amp):
                output = model(batch)

                loss_logit = criterion_image(output['logits'], batch['mask'])
                
                loss_aux = 0.0
                for loss_item in args['aux_losses']:
                    loss_aux_value = loss_item[1] * criterion_aux  (output[loss_item[0]],  batch['mask'], batch['organ']).mean()
                    loss_aux += loss_aux_value
  
            optimizer.zero_grad()
            loss = loss_logit + loss_aux

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            if global_it % args['virtual_batch_size'] == 0:
                scaler.step(optimizer)
                
            scaler.update()

            # ---

            batch_train_loss.append(loss.item())          

            if global_it % args['virtual_batch_size'] == 0:
                loss_value = np.mean(batch_train_loss)
                train_loss.append(loss_value)
                trlog.log_train_loss(global_it, loss_value)
                batch_train_loss = []
                lr_hist.append(optimizer.param_groups[0]['lr'])
                trlog.log_train_lr(global_it, optimizer.param_groups[0]['lr'])
                scheduler_warmup.step()

            if global_it % (args['virtual_batch_size'] * 5) == 0:
                log('==> Epoch {} ({:03d}/{:03d}) | loss: {:.5f}'.format(epoch, iteration, len(loader_train), loss.item()))

            if iteration % (args['virtual_batch_size'] * 40) == 0:
                clear_output()

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
                fig.suptitle('Sample image from training (organ {})'.format(ID2ORGAN[int(batch['organ'][0].item())]))
                ax1.imshow(np.clip(batch['image'][0].detach().permute((1, 2, 0)).cpu().numpy() * std + mean, 0, 1.0))
                ax2.imshow(batch['mask'].detach().cpu().numpy()[0])
                pred_mask = torch.softmax(output['logits'].detach(), dim=1)[0, batch['organ'][0].item() + 1]
                ax3.imshow(pred_mask.cpu().numpy().astype(np.float32), vmin=0.0, vmax=1.0)
                if iteration % 800 == 0:
                    plt.savefig(os.path.join(ARTIFACTS_PTH, 'sample_it_{:06d}.png'.format(global_it)))
                plt.show()

                plt.figure(figsize=(10, 5))
                plt.yscale('log')
                plt.plot(train_loss)
                plt.grid()
                plt.savefig(os.path.join(ARTIFACTS_PTH, 'train_loss.png'))
                plt.show()

                plt.figure(figsize=(10, 4))
                plt.plot(lr_hist, color='red')
                plt.title('lr')
                plt.grid()
                plt.savefig(os.path.join(ARTIFACTS_PTH, 'lr.png'))
                plt.show()

        # val -------------------------------
        if epoch % 1 == 0:
            log('Eval')
            model.eval()

            with torch.no_grad():
                val_res = validate(val_df_id, model)
                val_dice = np.mean(val_res['dices'])
                log('DICE: {}'.format(val_dice))
                for key, value in val_res['by_organ'].items():
                    log('{:20}: {:6.5f}'.format(key, np.mean(value)))

                torch.save(model.state_dict(), os.path.join(
                    OUTPUT_PTH, 
                    '{}_ep_{:03d}_dice_{:08.6f}.pt'.format(args['short_name'], epoch, val_dice)
                )) 
                torch.save(model.state_dict(), os.path.join(
                    OUTPUT_PTH, 
                    '{}_ep_{:03d}_dice_{:08.6f}_LAST.pt'.format(args['short_name'], epoch, val_dice)
                ))
                logfile.flush()
                
                logtr.log_val_dice(epoch, val_res)
                logtr.dump()

        if args['keep_last_n_checkpoints'] is not None:
            if args['keep_last_n_checkpoints']['function'] is not None:
                keep_function = getattr(postprocess, args['criterion_schedule']['function'])
                keep_function(args, top_n=args['criterion_schedule']['n'])
        
        if args['unfreeze_encoder']['active']:
            if epoch == args['unfreeze_encoder']['n_epoch']:
                # unfreeze by ceratin epoch
                
                for param in model.rgb.parameters():
                    param.requires_grad = True
                for param in model.encoder.parameters():
                    param.requires_grad = True
