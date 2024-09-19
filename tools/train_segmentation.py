# -*- coding: utf-8 -*-
# @Author: Simon Dahan
#
# Created on Wed Sep 18 2024
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2024 MeTrICS Lab
#

'''
This file implements the training procedure to train a MS-SiT model for segmentation.
'''

import os
import argparse
import yaml
import sys
import timm
from datetime import datetime
#remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric, GeneralizedDiceScore
from monai.utils import MetricReduction
from monai.networks import one_hot

from models.ms_sit_unet import MSSiTUNet
from models.ms_sit_unet_shifted import MSSiTUNet_shifted

from tools.utils import logging_ms_sit
from tools.utils import get_data_path_segmentation, get_dataloaders_segmentation, get_dimensions, get_scheduler, save_segmentation_results_UKB, save_segmentation_results_MindBoggle, save_segmentation_results_MindBoggle_test, save_segmentation_results_UKB_test

from tools.metrics import dice_coeff

def train(config):

    #############################
    ######     CONFIG      ######
    #############################

    print('')
    print('#'*30)
    print('########### Config ###########')
    print('#'*30)
    print('')

    #mesh_resolution
    ico_mesh = config['mesh_resolution']['ico_mesh']
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['sub_ico_{}'.format(ico_grid)]['num_patches']
    num_vertices = config['sub_ico_{}'.format(ico_grid)]['num_vertices']

    #data
    dataset = config['data']['dataset']
    task = config['data']['task']
    configuration = config['data']['configuration']
    num_classes = config['transformer']['num_classes']

    #training
    gpu = config['training']['gpu']
    LR = config['training']['LR']
    loss = config['training']['loss']
    epochs = config['training']['epochs']
    val_epoch = config['training']['val_epoch']
    testing = config['training']['testing']
    log_training_epoch = config['training']['log_training_epoch']
    log_iteration = config['training']['log_iteration']
    early_stopping = config['training']['early_stopping']
    use_confounds = False


    segmentation_task = True 

    #data path
    try:
        data_path, labels_path = get_data_path_segmentation(config)
    except:
        raise("can't get data path")

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    print('gpu: {}'.format(device))   
    print('dataset: {}'.format(dataset))  
    print('task: {}'.format(task))  
    print('model: {}'.format(config['MODEL']))
    print('configuration: {}'.format(configuration))  
    print('data path: {}'.format(data_path))
    print('label path: {}'.format(labels_path))

    ##############################
    ######     DATASET      ######
    ##############################

    print('')
    print('#'*30)
    print('######## Loading data ########')
    print('#'*30)
    print('')
    if  config['MODEL'] == 'ms-sit':
        print('Mesh resolution - ico {}'.format(ico_mesh))
        print('Grid resolution - ico {}'.format(ico_grid))
        print('Number of patches - {}'.format(num_patches))
        print('Number of vertices - {}'.format(num_vertices))
        print('Reorder patches: {}'.format(config['mesh_resolution']['reorder']))
        print('')

    try:
        train_loader, val_loader, test_loader = get_dataloaders_segmentation(config,data_path,labels_path)
    except:
        raise("can't get dataloaders")
    

    ##############################
    ######      LOGGING     ######
    ##############################

    print('')
    print('#'*30)
    print('########## Logging ###########')
    print('#'*30)
    print('')

    # creating folders for logging. 
    if config['MODEL'] == 'ms-sit':
        folder_to_save_model = logging_ms_sit(config) 
    else:
        raise('not implemented yet')
    
    try:
        os.makedirs(folder_to_save_model,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save_model))
    except OSError:
        print('folder already exist: {}'.format(folder_to_save_model))
    
    #tensorboard
    writer = SummaryWriter(log_dir=folder_to_save_model)

    ##############################
    #######     MODEL      #######
    ##############################
    
    print('')
    print('#'*30)
    print('######### Init model #########')
    print('#'*30)
    print('')

    if config['MODEL'] == 'ms-sit':    

        T, N, V, _, _ = get_dimensions(config)

    if config['MODEL'] == 'ms-sit':
        if config['transformer']['shifted_attention']:
            
            num_classes = config['transformer']['num_classes']
            print('*** using shifted attention with shifting factor {}***'.format(config['transformer']['window_size_factor']))
            model = MSSiTUNet_shifted(ico_init_resolution=config['mesh_resolution']['ico_grid'],
                            num_channels=T,
                            num_classes=num_classes,
                            embed_dim=config['transformer']['dim'],
                            depths=config['transformer']['depth'],
                            num_heads=config['transformer']['heads'],
                            window_size=config['transformer']['window_size'],
                            window_size_factor=config['transformer']['window_size_factor'],
                            mlp_ratio=config['transformer']['mlp_ratio'],
                            qkv_bias=True,
                            qk_scale=True,
                            dropout=config['transformer']['dropout'],
                            attention_dropout=config['transformer']['attention_dropout'],
                            dropout_path=config['transformer']['dropout_path'],
                            norm_layer=nn.LayerNorm,
                            use_pos_emb=config['transformer']['use_pos_emb'],
                            patch_norm=True,
                            use_confounds=use_confounds,
                            device=device,
                            reorder=config['mesh_resolution']['reorder'],
                            path_to_workdir=config['data']['path_to_workdir']
                            )
            
        else:
            
            num_classes = config['transformer']['num_classes']

            model = MSSiTUNet(ico_init_resolution=config['mesh_resolution']['ico_grid'],
                            num_channels=T,
                            num_classes=num_classes,
                            embed_dim=config['transformer']['dim'],
                            depths=config['transformer']['depth'],
                            num_heads=config['transformer']['heads'],
                            window_size=config['transformer']['window_size'],
                            mlp_ratio=config['transformer']['mlp_ratio'],
                            qkv_bias=True,
                            qk_scale=True,
                            dropout=config['transformer']['dropout'],
                            attention_dropout=config['transformer']['attention_dropout'],
                            dropout_path=config['transformer']['dropout_path'],
                            norm_layer=nn.LayerNorm,
                            use_pos_emb=config['transformer']['use_pos_emb'],
                            patch_norm=True,
                            use_confounds=use_confounds,
                            device=device,
                            reorder=config['mesh_resolution']['reorder'],
                            path_to_workdir=config['data']['path_to_workdir']
                            )

    print('')

    print('Training from scratch')
        
    model.to(device)
  
    print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('')

    #####################################
    #######     OPTIMISATION      #######
    #####################################
    
    if config['optimisation']['optimiser']=='SGD':
        print('Using SGD optimiser')
        optimizer = optim.SGD(model.parameters(), lr=LR, 
                                                weight_decay=config['SGD']['weight_decay'],
                                                momentum=config['SGD']['momentum'],
                                                nesterov=config['SGD']['nesterov'])
                                                
    elif config['optimisation']['optimiser']=='Adam':
        print('Using Adam optimiser')
        optimizer = optim.Adam(model.parameters(), lr=LR,
                                weight_decay=config['Adam']['weight_decay'])
      
    elif config['optimisation']['optimiser']=='AdamW':
        print('Using AdamW optimiser')
        optimizer = optim.AdamW(model.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])
    else:
        raise('not implemented yet')
    
    
    print(optimizer)

    ##############################
    #######     LOSS       #######
    ##############################

    if  segmentation_task: 
        if loss == 'ce':
            criterion = nn.CrossEntropyLoss(reduction='mean')
            print('Using {} criterion'.format(criterion))
        elif loss == 'dice':
            criterion = DiceLoss(to_onehot_y=True, softmax=True)
            print('Using {} criterion'.format(criterion))
        elif loss == 'diceCE':
            criterion = DiceCELoss(to_onehot_y=True, softmax=True,lambda_dice=config['training']['lambda_dice'], lambda_ce=config['training']['lambda_ce'])
            print('Using {} criterion'.format(criterion))
        elif loss == 'dice+CE':
            criterion1 = nn.CrossEntropyLoss(reduction='mean')
            criterion2 = DiceLoss(to_onehot_y=True, softmax=True)
            print('Using {} and {}'.format(criterion1,criterion2))
        elif loss == 'gdl':
            criterion = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
            print('Using {} criterion'.format(criterion))
        elif loss == 'gdl+CE':
            criterion1 = nn.CrossEntropyLoss(reduction='mean')
            criterion2 = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
            print('Using {} and {}'.format(criterion1,criterion2))
    
    ##############################
    #######     METRIC     #######
    ##############################

    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN)
    gdl_metric = GeneralizedDiceScore(include_background=True, reduction=MetricReduction.MEAN_BATCH,weight_type='square')
    
    ###################################
    #######     SCHEDULING      #######
    ###################################

    it_per_epoch = np.ceil(len(train_loader))
    print('number of iterations per epoch: {}'.format(it_per_epoch))

    scheduler = get_scheduler(config, it_per_epoch, optimizer)
    print('scheduler: {}'.format(scheduler))

    ##############################
    ######     TRAINING     ######
    ##############################
    
    print('')
    print('#'*30)
    print('###### Starting training #####')
    print('#'*30)
    print('')

    best_loss = 1000000000000000 
    best_dice = -10
    c_early_stop = 0

    for epoch in range(epochs):

        running_loss = 0

        model.train()

        targets_ =  []
        preds_ = []
        
        dice_pred = []
        dice_monai_pred = []
        gdl_pred = []

        for i, data in enumerate(train_loader):
 
            inputs, targets = data[0].to(device), data[1].to(device)
            if config['MODEL']=='spherical-unet':
                inputs = inputs.permute(2,1,0)
            outputs = model(inputs)

            optimizer.zero_grad()
            if config['training']['loss'] == 'ce':
                loss = criterion(outputs, targets)
                loss.backward()
            elif config['training']['loss'] == 'dice' or config['training']['loss'] == 'gdl':
                loss = criterion(outputs, targets.unsqueeze(1))
                loss.backward()
            elif config['training']['loss'] == 'diceCE':
                loss = criterion(outputs, targets.unsqueeze(1))
                loss.backward(retain_graph=False)
            elif config['training']['loss'] == 'dice+CE' or config['training']['loss'] == 'gdl+CE':
                loss1 = criterion1(outputs, targets)
                loss2 = criterion2(outputs, targets.unsqueeze(1))
                loss = loss1 + loss2
                loss.backward()

            y_pred = one_hot(torch.argmax(torch.nn.functional.softmax(outputs,dim=1),dim=1).unsqueeze(1),dim=1,num_classes=config['transformer']['num_classes'])
            y_target = one_hot(targets.unsqueeze(1),dim=1,num_classes=config['transformer']['num_classes'])
                        
            gdl_pred.append(gdl_metric(y_pred.cpu(),y_target.cpu()).numpy())
            dice_monai_pred.append(dice_metric(y_pred,y_target).cpu().numpy().mean(axis=1))
            dice_pred.append(dice_coeff(y_pred,y_target).cpu().detach().numpy())
            
            optimizer.step()

            running_loss += loss.item() #average over batch (default)
            
            ##############################
            ##########   TB    ###########
            ##############################
                        
            writer.add_scalar('loss/train_it', loss.item(), epoch*it_per_epoch + i +1)

            if config['optimisation']['use_scheduler']:
                if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                    if config['optimisation']['warmup']:
                        if (epoch+1)<config['optimisation']['nbr_step_warmup']:
                            writer.add_scalar('LR',scheduler.get_lr()[0], epoch*it_per_epoch + i +1 )
                        else:
                            writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 )
                    else:
                        writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 )
                elif config['optimisation']['scheduler'] == 'CosineDecay':
                    scheduler.step()
                    writer.add_scalar('LR',scheduler.get_last_lr()[0], epoch*it_per_epoch + i +1 )
                else:
                    scheduler.step()
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 )
            else:
                if config['optimisation']['warmup']:
                    scheduler.step()
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 )
                else:
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 )

            ##############################
            #########  LOG IT  ###########
            ##############################

            if (epoch*it_per_epoch + i+1)%log_iteration==0:

                loss_train_it = running_loss / (i+1)
                
                gdl_train_it =  np.hstack(gdl_pred).mean()
                dice_train_it =  np.hstack(dice_pred).mean()
                dice_monai_train_it =  np.hstack(dice_monai_pred).mean()

                if config['optimisation']['use_scheduler']:
                    if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                        if config['optimisation']['warmup']:
                            if (epoch+1)<config['optimisation']['nbr_step_warmup']:
                                print('| It - {} | Loss - {:.4f} |  DICE - {:.4f} | DICE MONAI - {:.4f} |  GDL - {:.4f} |  LR - {}'.format(epoch*it_per_epoch + i +1,loss_train_it, dice_train_it, dice_monai_train_it, gdl_train_it, scheduler.get_lr()[0] ))
                            else: 
                                print('| It - {} | Loss - {:.4f} | DICE - {:.4f} | DICE MONAI - {:.4f} |  GDL - {:.4f} | LR - {}'.format(epoch*it_per_epoch + i +1, loss_train_it, dice_train_it, dice_monai_train_it,gdl_train_it, optimizer.param_groups[0]['lr'] ))
                        else:
                            print('| It - {} | Loss - {:.4f} | DICE - {:.4f} | DICE MONAI - {:.4f} |  GDL - {:.4f} |  LR - {}'.format(epoch*it_per_epoch + i +1, loss_train_it, dice_train_it, dice_monai_train_it,gdl_train_it,  optimizer.param_groups[0]['lr'] ))
                    else:
                        print('| It - {} | Loss - {:.4f} | DICE - {:.4f} | DICE MONAI - {:.4f} |  GDL - {:.4f} |  LR - {}'.format(epoch*it_per_epoch + i +1, loss_train_it,  dice_train_it, dice_monai_train_it,gdl_train_it, scheduler.get_last_lr()[0] ))
                else:
                    print('| It - {} | Loss - {:.4f} |  DICE - {:.4f} | DICE MONAI - {:.4f} |  GDL - {:.4f} | LR - {}'.format(epoch*it_per_epoch + i +1, loss_train_it, dice_train_it, dice_monai_train_it, gdl_train_it, optimizer.param_groups[0]['lr']))

                
                config['logging']['folder_model_saved'] = folder_to_save_model
                config['logging']['iterations'] = int(epoch*it_per_epoch + i +1)

                ###############################
                ######    SAVING CKPT    ######
                ###############################

                if config['training']['save_ckpt']:
                    torch.save({'epoch': epoch,
                                'model_state_dict':model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_train_it,
                                },
                                os.path.join(folder_to_save_model,'checkpoint_it.pth'))
                with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                    yaml.dump(config, yaml_file)
        
        #################################
        #########  LOG EPOCH  ###########
        #################################
            
        loss_train_epoch = running_loss / (i+1)
        
        gdl_train_epoch =  np.hstack(gdl_pred).mean()
        dice_train_epoch =  np.hstack(dice_pred).mean()
        dice_monai_train_epoch =  np.hstack(dice_monai_pred).mean()
        writer.add_scalar('loss/train', loss_train_epoch, epoch+1)
        writer.add_scalar('dice/train', dice_train_epoch, epoch+1)
        writer.add_scalar('dice_monai/train', dice_monai_train_epoch, epoch+1)

        if (epoch+1)%log_training_epoch==0:
            
            if config['optimisation']['use_scheduler']:
                if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                    if config['optimisation']['warmup']:
                        if (epoch+1)<config['optimisation']['nbr_step_warmup']:
                            print('| Epoch - {} | Loss - {:.4f} |  DICE - {:.4f} |  DICE MONAI - {:.4f} | GDL - {:.4f} |  LR - {}'.format(epoch+1,loss_train_epoch, dice_train_epoch, dice_monai_train_epoch,  gdl_train_epoch, scheduler.get_lr()[0] ))
                        else: 
                            print('| Epoch - {} | Loss - {:.4f} | DICE - {:.4f} |  DICE MONAI - {:.4f} | GDL - {:.4f} | LR - {}'.format(epoch+1, loss_train_epoch, dice_train_epoch, dice_monai_train_epoch,  gdl_train_epoch, optimizer.param_groups[0]['lr'] ))
                    else:
                        print('| Epoch - {} | Loss - {:.4f} | DICE - {:.4f} |  DICE MONAI - {:.4f} | GDL - {:.4f} |  LR - {}'.format(epoch+1, loss_train_epoch, dice_train_epoch, dice_monai_train_epoch,  gdl_train_epoch,  optimizer.param_groups[0]['lr'] ))
                else:
                    print('| Epoch - {} | Loss - {:.4f} | DICE - {:.4f} |  DICE MONAI - {:.4f} | GDL - {:.4f} |  LR - {}'.format(epoch+1, loss_train_epoch,  dice_train_epoch,  dice_monai_train_epoch, gdl_train_epoch, scheduler.get_last_lr()[0] ))
            else:
                print('| Epoch - {} | Loss - {:.4f} |  DICE - {:.4f} |  DICE MONAI - {:.4f} | GDL - {:.4f} | LR - {}'.format(epoch+1, loss_train_epoch, dice_train_epoch,  dice_monai_train_epoch, gdl_train_epoch, optimizer.param_groups[0]['lr']))

        ##############################
        ######    VALIDATION    ######
        ##############################

        if (epoch+1)%val_epoch==0:

            running_val_loss = 0
            model.eval()

            preds_ = []
            
            dice_pred_val = []
            dice_monai_pred_val = []
            gdl_pred_val = []

            with torch.no_grad():

                for i, data in enumerate(val_loader):
                    inputs, targets = data[0].to(device), data[1].to(device)
                    if config['MODEL']=='spherical-unet':
                        inputs = inputs.permute(2,1,0)
                    outputs = model(inputs)

                    preds_.append(torch.argmax(torch.nn.functional.softmax(outputs,dim=1),dim=1).detach().cpu().numpy())
                    
                    if config['training']['loss'] == 'ce':
                        loss = criterion(outputs, targets)
                    elif config['training']['loss'] == 'dice' or config['training']['loss'] == 'gdl' or config['training']['loss'] == 'diceCE':
                        loss = criterion(outputs, targets.unsqueeze(1))
                    elif config['training']['loss'] == 'dice+CE' or config['training']['loss'] == 'gdl+CE':
                        loss1 = criterion1(outputs, targets)
                        loss2 = criterion2(outputs, targets.unsqueeze(1))
                        loss = loss1 + loss2

                    running_val_loss += loss.item()
                    
                    y_pred = one_hot(torch.argmax(torch.nn.functional.softmax(outputs,dim=1),dim=1),dim=0,num_classes=config['transformer']['num_classes']).unsqueeze(0)
                    y_target = one_hot(targets,dim=0,num_classes=config['transformer']['num_classes']).unsqueeze(0)
                    
                    gdl_pred_val.append(gdl_metric(y_pred.cpu(),y_target.cpu()).numpy())
                    dice_monai_pred_val.append(dice_metric(y_pred,y_target).cpu().numpy().mean())
                    dice_pred_val.append(dice_coeff(y_pred,y_target).cpu().detach().numpy())


            loss_val_epoch = running_val_loss/(i+1)
            
            gdl_val_epoch = np.mean(gdl_pred_val)
            dice_val_epoch = np.mean(dice_pred_val)
            dice_monai_val_epoch = np.mean(dice_monai_pred_val)

            writer.add_scalar('loss/val',loss_val_epoch , epoch+1)
            writer.add_scalar('dice/val',dice_val_epoch , epoch+1)
            writer.add_scalar('dice_monai/val',dice_monai_val_epoch , epoch+1)

            print('| Validation | Epoch - {} | DICE - {:.4f} | DICE MONAI - {:.4f} | GDL - {:.4f} | Loss - {:.4f}'.format(epoch+1, dice_val_epoch, dice_monai_val_epoch, gdl_val_epoch,loss_val_epoch,))

            if loss_val_epoch< best_loss or dice_val_epoch>best_dice:
                print('saving checkpoint at epoch: {}'.format(epoch+1))
                best_dice = dice_val_epoch
                best_epoch = epoch+1
                best_loss = loss_val_epoch
                c_early_stop = 0

                config['logging']['folder_model_saved'] = folder_to_save_model
                config['results'] = {}
                config['results']['best_val_dice'] = float(best_dice)
                config['results']['best_val_epoch'] = best_epoch
                config['results']['best_val_loss'] = best_loss
                config['results']['training_finished'] = False

                ###############################
                ######    SAVING CKPT    ######
                ###############################
                
                if dataset == 'UKB':
                    save_segmentation_results_UKB(config,preds_,os.path.join(folder_to_save_model,'results_val','epoch_{}'.format(epoch+1)), epoch+1)
                
                elif dataset == 'MindBoggle':
                    save_segmentation_results_MindBoggle(config,preds_,os.path.join(folder_to_save_model,'results_val','epoch_{}'.format(epoch+1)), epoch+1)

                if config['training']['save_ckpt']:
                    torch.save({'epoch': epoch,
                                'model_state_dict':model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_train_epoch,
                                },
                                os.path.join(folder_to_save_model,'checkpoint_best.pth'))
                with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                    yaml.dump(config, yaml_file)


            elif early_stopping:
                c_early_stop += 1

        if early_stopping and (c_early_stop>=early_stopping):
            print('stop training - early stopping')
            break

        ##################################
        ######   UPDATE SCHEDULER  #######
        ##################################

        if config['optimisation']['use_scheduler']:
                if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(metrics=dice_val_epoch)
    
    if early_stopping and (c_early_stop>early_stopping):
        config['results']['training_finished'] = 'early stopping' 
    else:
        config['results']['training_finished'] = True 

    #####################################
    ######    SAVING FINAL CKPT    ######
    #####################################

    if config['training']['save_ckpt']:
        torch.save({'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train_epoch,
                    },
                    os.path.join(folder_to_save_model,'checkpoint_final.pth'))
    
    with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file)

    ##############################
    ######     TESTING      ######
    ##############################

    if testing:
        
        print('')
        print('#'*30)
        print('###### Starting testing #####')
        print('#'*30)
        print('')

        del train_loader
        del val_loader
        del model

        if config['MODEL'] == 'ms-sit':
            if config['transformer']['shifted_attention']:
                
                num_classes = config['transformer']['num_classes']
                print('*** using shifted attention with shifting factor {}***'.format(config['transformer']['window_size_factor']))
                test_model = MSSiTUNet_shifted(ico_init_resolution=config['mesh_resolution']['ico_grid'],
                                num_channels=T,
                                num_classes=num_classes,
                                embed_dim=config['transformer']['dim'],
                                depths=config['transformer']['depth'],
                                num_heads=config['transformer']['heads'],
                                window_size=config['transformer']['window_size'],
                                window_size_factor=config['transformer']['window_size_factor'],
                                mlp_ratio=config['transformer']['mlp_ratio'],
                                qkv_bias=True,
                                qk_scale=True,
                                dropout=config['transformer']['dropout'],
                                attention_dropout=config['transformer']['attention_dropout'],
                                dropout_path=config['transformer']['dropout_path'],
                                norm_layer=nn.LayerNorm,
                                use_pos_emb=config['transformer']['use_pos_emb'],
                                patch_norm=True,
                                use_confounds=use_confounds,
                                device=device,
                                reorder=config['mesh_resolution']['reorder'],
                                path_to_workdir=config['data']['path_to_workdir']
                                )
                
            else:
                num_classes = config['transformer']['num_classes']
                test_model = MSSiTUNet(ico_init_resolution=config['mesh_resolution']['ico_grid'],
                                num_channels=T,
                                num_classes=num_classes,
                                embed_dim=config['transformer']['dim'],
                                depths=config['transformer']['depth'],
                                num_heads=config['transformer']['heads'],
                                window_size=config['transformer']['window_size'],
                                mlp_ratio=config['transformer']['mlp_ratio'],
                                qkv_bias=True,
                                qk_scale=True,
                                dropout=config['transformer']['dropout'],
                                attention_dropout=config['transformer']['attention_dropout'],
                                dropout_path=config['transformer']['dropout_path'],
                                norm_layer=nn.LayerNorm,
                                use_pos_emb=config['transformer']['use_pos_emb'],
                                patch_norm=True,
                                use_confounds=use_confounds,
                                device=device,
                                reorder=config['mesh_resolution']['reorder'],
                                path_to_workdir=config['data']['path_to_workdir']
                                )

        elif config['MODEL'] == 'spherical-unet':
            test_model = sphericalunet_regression(num_features = config['spherical-unet']['num_features'],
                                            in_channels=len(config['spherical-unet']['channels']))

        test_model.load_state_dict(torch.load(os.path.join(folder_to_save_model,'checkpoint_best.pth'))['model_state_dict'])
        test_model.to(device)

        test_model.eval()

        print('Starting testing')

        with torch.no_grad():

            targets_ = []
            preds_ = []

            dice_pred_test = []
            dice_monai_pred_test = []
            gdl_pred_test = []

            for i, data in enumerate(test_loader):

                inputs, targets = data[0].to(device), data[1].to(device)
                if config['MODEL']=='spherical-unet':
                    inputs = inputs.permute(2,1,0)
                outputs = test_model(inputs)

                preds_.append(torch.argmax(torch.nn.functional.softmax(outputs,dim=1),dim=1).detach().cpu().numpy())

                y_pred = one_hot(torch.argmax(torch.nn.functional.softmax(outputs,dim=1),dim=1),dim=0,num_classes=config['transformer']['num_classes']).unsqueeze(0)
                y_target = one_hot(targets,dim=0,num_classes=config['transformer']['num_classes']).unsqueeze(0)
                
                gdl_pred_test.append(gdl_metric(y_pred.cpu(),y_target.cpu()).numpy())
                dice_monai_pred_test.append(dice_metric(y_pred,y_target).cpu().numpy().mean())
                dice_pred_test.append(dice_coeff(y_pred,y_target).cpu().detach().numpy())


            gdl_test_epoch = np.mean(gdl_pred_test)
            dice_test_epoch = np.mean(dice_pred_test)
            dice_monai_test_epoch = np.mean(dice_monai_pred_test)        

            print('| TESTING RESULTS | DICE - {:.4f} | DICE MONAI - {:.4f} | GDL - {:.4f} | '.format(dice_test_epoch, dice_monai_test_epoch, gdl_test_epoch))


            config['logging']['folder_model_saved'] = folder_to_save_model
            config['results_testing'] = {}
            config['results_testing']['gdl'] = float(gdl_test_epoch)
            config['results_testing']['dice'] = dice_test_epoch
            config['results_testing']['training_finished'] = False

            if dataset == 'UKB':
                save_segmentation_results_UKB_test(config,preds_,os.path.join(folder_to_save_model,'results_test','epoch_{}'.format(epoch+1)), epoch+1)
                
            elif dataset == 'MindBoggle':
                save_segmentation_results_MindBoggle_test(config,preds_,os.path.join(folder_to_save_model,'results_test','epoch_{}'.format(epoch+1)), epoch+1)
            
          
            with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                yaml.dump(config, yaml_file)

    else:
        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ViT')

    parser.add_argument(
                        'config',
                        type=str,
                        default='./config/hparams.yml',
                        help='path where the data is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    train(config)
