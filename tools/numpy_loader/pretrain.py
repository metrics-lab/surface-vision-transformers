# -*- coding: utf-8 -*-
# @Author: Simon Dahan
#
# Created on Fri Oct 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#


'''
This file implements the training procedure to train a SiT model.
Models can be either trained:
    - from scratch
    - from pretrained weights (after self-supervision or ImageNet for instance)
Models can be trained for two tasks:
    - age at scan prediction
    - birth age prediction

Pretrained ImageNet models are downloaded from the Timm library. 
'''

import os
import argparse
import yaml
import sys
import timm
from datetime import datetime


sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


from models.sit import SiT
from models.mpp import masked_patch_pretraining


from warmup_scheduler import GradualWarmupScheduler

from utils.utils import load_weights_imagenet

from torch.utils.tensorboard import SummaryWriter


def train(config):

    gpu = config['training']['gpu']
    LR = config['training']['LR']
    use_l1loss = config['training']['l1loss']
    epochs = config['training']['epochs']
    val_epoch = config['training']['val_epoch']
    testing = config['training']['testing']
    bs = config['training']['bs']
    bs_val = config['training']['bs_val']
    configuration = config['data']['configuration']
    task = config['data']['task']

    ico = config['resolution']['ico']
    sub_ico = config['resolution']['sub_ico']

    data_path = config['data']['data_path'].format(task,configuration)
    
    folder_to_save_model = config['logging']['folder_to_save_model']

    num_patches = config['sub_ico_{}'.format(sub_ico)]['num_patches']
    num_vertices = config['sub_ico_{}'.format(sub_ico)]['num_vertices']

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    print('')
    print('#'*30)
    print('##### Config #####')
    print('#'*30)
    print('')

    print(device)  
    print(data_path)

    ##############################
    ######     DATASET      ######
    ##############################

    print('')
    print('#'*30)
    print('##### Loading data#####')
    print('#'*30)
    print('')

    print('LOADING DATA: ICO {} - sub-res ICO {}'.format(ico,sub_ico))

    #loading already processed and patched cortical surfaces. 

    train_data = np.load(os.path.join(data_path,'train_data.npy'))
    train_label = np.load(os.path.join(data_path,'train_labels.npy'))

    print('training data: {}'.format(train_data.shape))

    val_data = np.load(os.path.join(data_path,'validation_data.npy'))
    val_label = np.load(os.path.join(data_path,'validation_labels.npy'))

    print('validation data: {}'.format(val_data.shape))

    train_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float(),
                                                    torch.from_numpy(train_label).float())

    train_loader = torch.utils.data.DataLoader(train_data_dataset,
                                                    batch_size = bs,
                                                    shuffle=True,
                                                    num_workers=16)
    
    val_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_data).float(),
                                                    torch.from_numpy(val_label).float())

 
    val_loader = torch.utils.data.DataLoader(val_data_dataset,
                                            batch_size = bs_val,
                                            shuffle=False,
                                            num_workers=16)

        
    if testing:
        test_data = np.load(os.path.join(data_path,'test_data.npy'))
        test_label = np.load(os.path.join(data_path,'test_labels.npy')).reshape(-1)

        print('testing data: {}'.format(test_data.shape))
        print('')

        test_data_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float(),
                                                        torch.from_numpy(test_label).float())

        test_loader = torch.utils.data.DataLoader(test_data_dataset,
                                                batch_size = bs_val,
                                                shuffle=False,
                                                num_workers=16)


    ##############################
    ######      LOGGING     ######
    ##############################

    # creating folders for logging. 
    try:
        os.mkdir(folder_to_save_model)
        print('Creating folder: {}'.format(folder_to_save_model))
    except OSError:
        print('folder already exist: {}'.format(folder_to_save_model))

    date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    # folder time
    folder_to_save_model = os.path.join(folder_to_save_model,date)
    print(folder_to_save_model)
    if config['transformer']['dim'] == 192:
        folder_to_save_model = folder_to_save_model + '-tiny'
    elif config['transformer']['dim'] == 384:
        folder_to_save_model = folder_to_save_model + '-small'
    elif config['transformer']['dim'] == 768:
        folder_to_save_model = folder_to_save_model + '-base'
    
    if config['training']['load_weights_imagenet']:
        folder_to_save_model = folder_to_save_model + '-imgnet'
    if config['training']['load_weights_ssl']:
        folder_to_save_model = folder_to_save_model + '-ssl'
        if config['training']['dataset_ssl']=='hcp':
            folder_to_save_model = folder_to_save_model + '-hcp'
        elif config['training']['dataset_ssl']=='dhcp-hcp':
            folder_to_save_model = folder_to_save_model + '-dhcp-hcp'
        elif config['training']['dataset_ssl']=='dhcp':
            folder_to_save_model = folder_to_save_model + '-dhcp'
    if config['training']['finetuning']:
        folder_to_save_model = folder_to_save_model + '-finetune'
    else:
        folder_to_save_model = folder_to_save_model + '-freeze'

    try:
        os.mkdir(folder_to_save_model)
        print('Creating folder: {}'.format(folder_to_save_model))
    except OSError:
        print('folder already exist: {}'.format(folder_to_save_model))

    writer = SummaryWriter(log_dir=folder_to_save_model)


    ##############################
    #######     MODEL      #######
    ##############################

    print('')
    print('#'*30)
    print('##### Init model #####')
    print('#'*30)
    print('')

    if config['transformer']['model'] == 'SiT':

        model = SiT(dim=config['transformer']['dim'],
                        depth=config['transformer']['depth'],
                        heads=config['transformer']['heads'],
                        mlp_dim=config['transformer']['mlp_dim'],
                        pool=config['transformer']['pool'], 
                        num_patches=num_patches,
                        num_classes=config['transformer']['num_classes'],
                        num_channels=config['transformer']['num_channels'],
                        num_vertices=num_vertices,
                        dim_head=config['transformer']['dim_head'],
                        dropout=config['transformer']['dropout'],
                        emb_dropout=config['transformer']['emb_dropout'])
    
    if config['training']['load_weights_ssl']:

        print('Loading weights from self-supervision training')
        model.load_state_dict(torch.load(config['weights']['ssl_mpp'],map_location=device),strict=False)
    
    if config['training']['load_weights_imagenet']:

        print('Loading weights from imagenet pretraining')
        model_trained = timm.create_model(config['weights']['imagenet'], pretrained=True)
        new_state_dict = load_weights_imagenet(model.state_dict(),model_trained.state_dict(),config['transformer']['depth'])
        model.load_state_dict(new_state_dict)
    

    model.to(device)

    print('Number of parameters encoder: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('')

    ##################################################
    #######     SELF-SUPERVISION PIPELINE      #######
    ##################################################

    if config['SSL'] == 'mpp':

        print('Pretrain using Masked Patch Prediction')
        ssl = masked_patch_pretraining(transformer=model,
                                    dim_in = config['transformer']['dim'],
                                    dim_out= num_vertices*config['transformer']['num_channels'],
                                    device=device,
                                    mask_prob=config['pretraining_mpp']['mask_prob'],
                                    replace_prob=config['pretraining_mpp']['replace_prob'],
                                    swap_prob=config['pretraining_mpp']['swap_prob'],
                                    num_vertices=num_vertices,
                                    channels=config['transformer']['num_channels'])
    else:
        raise('not implemented yet')  
    
    ssl.to(device)

    print('Number of parameters pretraining pipeline : {:,}'.format(sum(p.numel() for p in ssl.parameters() if p.requires_grad)))
    print('')

    #####################################
    #######     OPTIMISATION      #######
    #####################################

    if config['optimisation']['optimiser']=='Adam':
        print('using Adam optimiser')
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=config['Adam']['weight_decay'])
    elif config['optimisation']['optimiser']=='SGD':
        print('using SGD optimiser')
        optimizer = optim.SGD(model.parameters(), lr=LR, 
                                                weight_decay=config['SGD']['weight_decay'],
                                                momentum=config['SGD']['momentum'],
                                                nesterov=config['SGD']['nesterov'])
    elif config['optimisation']['optimiser']=='AdamW':
        print('using AdamW optimiser')
        optimizer = optim.AdamW(model.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])
    else:
        raise('not implemented yet')

    ###################################
    #######     SCHEDULING      #######
    ###################################
    
    it_per_epoch = np.ceil(len(train_loader))

    ##################################
    ######     PRE-TRAINING     ######
    ##################################

    print('')
    print('#'*30)
    print('#### Starting pre-training ###')
    print('#'*30)
    print('')

    best_val_loss = 100000000000
    c_early_stop = 0
    
    for epoch in range(epochs):

        ssl.train()

        running_loss = 0

        for i, data in enumerate(train_loader):

            inputs, _ = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            if config['SSL'] == 'mpp':
                mpp_loss, _ = ssl(inputs)
                
            mpp_loss.backward()
            optimizer.step()

            running_loss += mpp_loss.item()

            writer.add_scalar('loss/train_it', mpp_loss.item(), epoch*it_per_epoch+1)

        ##############################
        #########  LOG IT  ###########
        ##############################

        if (epoch+1)%5==0:

            print('| Epoch - {} | It - {} | Loss - {:.4f} | LR - {}'.format(epoch+1, epoch*it_per_epoch + i +1, running_loss / (i+1), optimizer.param_groups[0]['lr']))
        
        loss_pretrain_epoch = running_loss / (i+1)

        writer.add_scalar('loss/train', loss_pretrain_epoch, epoch+1)


        ##############################
        ######    VALIDATION    ######
        ##############################

        if (epoch+1)%val_epoch==0: 
            
            running_val_loss = 0
            ssl.eval()

            with torch.no_grad():

                for i, data in enumerate(val_loader):

                    inputs, _ = data[0].to(device), data[1].to(device)

                    if config['SSL'] == 'mpp':
                        mpp_loss, _ = ssl(inputs)

                    running_val_loss += mpp_loss.item()

            loss_pretrain_val_epoch = running_val_loss /(i+1)

            writer.add_scalar('loss/val', loss_pretrain_val_epoch, epoch+1)

            print('| Validation | Epoch - {} | Loss - {} | '.format(epoch+1, loss_pretrain_val_epoch))

            if loss_pretrain_val_epoch < best_val_loss:
                best_val_loss = loss_pretrain_val_epoch
                best_epoch = epoch+1
                c_early_stop = 0

                config['results'] = {}
                config['results']['best_epoch'] = best_epoch
                config['results']['best_current_loss'] = loss_pretrain_epoch
                config['results']['best_current_loss_validation'] = best_val_loss

                with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                        yaml.dump(config, yaml_file)

                print('saving_model')
                torch.save({ 'epoch':epoch+1,
                             'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss':loss_pretrain_epoch,
                             },
                            os.path.join(folder_to_save_model, 'encoder-best.pt'))
                torch.save({ 'epoch':epoch+1,
                             'model_state_dict': ssl.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss':loss_pretrain_epoch,
                             },
                            os.path.join(folder_to_save_model, 'encoder-decoder-best.pt'))

    print('')
    print('Final results: best model obtained at epoch {} - loss {}'.format(best_epoch,best_val_loss))

    config['logging']['folder_model_saved'] = folder_to_save_model
    config['results']['final_loss'] = loss_pretrain_epoch
    config['results']['training_finished'] = True 

    with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file)


    #####################################
    ######    SAVING FINAL CKPT    ######
    #####################################

    torch.save({'epoch':epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss_pretrain_epoch,
                },
                os.path.join(folder_to_save_model,'encoder-final.pt'))

    torch.save({'epoch':epoch+1,
                'model_state_dict': ssl.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss_pretrain_epoch,
                },
                os.path.join(folder_to_save_model,'encoder-decoder-final.pt'))


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
