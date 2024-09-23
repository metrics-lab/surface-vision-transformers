# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Your name
# @Last Modified time: 2022-04-07 15:51:18
#
# Created on Wed Oct 20 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#
import os
import torch

from datetime import datetime
import pandas as pd
import nibabel as nb
import numpy as np

from tools.dataloader import loader_metrics_segmentation, loader_metrics

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from warmup_scheduler import GradualWarmupScheduler

def get_dataloaders_numpy(data_path, testing, bs, bs_val):
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
        
        return train_loader, val_loader, test_loader
    
    else:
        return train_loader, val_loader, None


def logging_ms_sit(config, pretraining=False):

    if pretraining:
        folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],'pretraining',config['data']['task'],config['SSL'],config['mesh_resolution']['ico_grid'],config['data']['configuration'])
    
    else:
        if config['data']['task'] =='segmentation':
            folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'])
        else:
            if config['data']['dataset']=='dHCP':
                folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],config['data']['task'],config['mesh_resolution']['ico_grid'],config['data']['configuration'])
            elif config['data']['dataset']=='HCP':
                folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],config['data']['task'],config['mesh_resolution']['ico_grid'],config['data']['registration'])
            elif config['data']['dataset']=='UKB':
                folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],config['data']['task'],config['mesh_resolution']['ico_grid'],config['data']['registration'])
    
    if config['augmentation']['prob_augmentation']:
        folder_to_save_model = os.path.join(folder_to_save_model,'augmentation')
    else:
        folder_to_save_model = os.path.join(folder_to_save_model,'no_augmentation')

    date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    folder_to_save_model = os.path.join(folder_to_save_model,date)

    if config['transformer']['dim'] == 96:
        folder_to_save_model = folder_to_save_model + '-tiny'
    elif config['transformer']['dim'] == 48:
        folder_to_save_model = folder_to_save_model + '-very-tiny'
    

    if config['training']['init_weights']!=False:
        folder_to_save_model = folder_to_save_model + '-'+config['training']['init_weights']

    if config['training']['finetuning']:
        folder_to_save_model = folder_to_save_model + '-finetune'
    else:
        folder_to_save_model = folder_to_save_model + '-freeze'

    return folder_to_save_model



def get_data_path_segmentation(config):

    dataloader = config['data']['loader']
    dataset = config['data']['dataset']
    configuration = config['data']['configuration']

    if str(dataloader) == 'metrics':
        if dataset == 'UKB':
            data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample/')
            labels_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/resample_segmentation_maps')  
                            
        elif dataset == 'MindBoggle':
            data_path = os.path.join(config['data']['path_to_data'],dataset,'{}/mindboggle_merged_metrics'.format(configuration))
            labels_path = os.path.join(config['data']['path_to_data'],dataset,'{}/mindboggle_resample_labels_ico6'.format(configuration)) 
                          
    else:
        raise('not implemented yet')
    
    return data_path, labels_path


def get_dataloaders_segmentation(config, 
                    data_path,
                    labels_path,):

    dataloader = config['data']['loader']
    if str(dataloader)=='metrics':
        train_loader, val_loader, test_loader = loader_metrics_segmentation(data_path,labels_path,config)
    else:
        raise('not implemented yet')
    
    return train_loader, val_loader, test_loader


def get_dimensions(config):

    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['sub_ico_{}'.format(ico_grid)]['num_patches']
    num_vertices = config['sub_ico_{}'.format(ico_grid)]['num_vertices']

    if config['MODEL'] in ['sit','ms-sit']:    
        channels = config['transformer']['channels']
    elif config['MODEL']== 'spherical-unet':
        channels = config['spherical-unet']['channels']
    elif config['MODEL']== 'monet':
        channels = config['monet']['channels']
    num_channels = len(channels)

    if config['MODEL'] in ['sit','ms-sit']:    
        
        T = num_channels
        N = num_patches
        
        V = num_vertices

        use_bottleneck = False
        bottleneck_dropout = 0.0

        print('Number of channels {}; Number of patches {}; Number of vertices {}'.format(T, N, V))
        print('Using bottleneck {}; Dropout bottleneck {}'.format(use_bottleneck,bottleneck_dropout))
        print('')

        return T, N, V, use_bottleneck, bottleneck_dropout
    

def get_scheduler(config, nbr_iteration_per_epoch ,optimizer):

    epochs = config['training']['epochs']

    if config['optimisation']['use_scheduler']:

        print('Using learning rate scheduler')

        if config['optimisation']['scheduler'] == 'StepLR':

            scheduler = StepLR(optimizer=optimizer,
                                step_size= config['StepLR']['stepsize'],
                                gamma= config['StepLR']['decay'])
        
        elif config['optimisation']['scheduler'] == 'CosineDecay':

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                    T_max = config['CosineDecay']['T_max'],
                                                                    eta_min= config['CosineDecay']['eta_min'],
                                                                    )

        elif config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer,
                                            mode='max',
                                            factor=0.5,
                                            patience=2,
                                            cooldown=0,
                                            min_lr=0.0000001
                                                )

        if config['optimisation']['warmup']:

            scheduler = GradualWarmupScheduler(optimizer,
                                                multiplier=1, 
                                                total_epoch=config['optimisation']['nbr_step_warmup'], 
                                                after_scheduler=scheduler)
     
    else:
        # to use warmup without fancy scheduler
        if config['optimisation']['warmup']:
            scheduler = StepLR(optimizer,
                                step_size=epochs*nbr_iteration_per_epoch)

            scheduler = GradualWarmupScheduler(optimizer,
                                                multiplier=1, 
                                                total_epoch=config['optimisation']['nbr_step_warmup'], 
                                                after_scheduler=scheduler)
        else:

            return None
            
    return scheduler



def save_segmentation_results_UKB(config,predictions,folder_to_save, epoch):

    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    val_ids = pd.read_csv(os.path.join(config['data']['path_to_workdir'],'labels/UKB/cortical_metrics/segmentation/half/val.csv')).ids
    for i, id in enumerate(val_ids):
        save_label_UKB(config['data']['path_to_data'],predictions[i],os.path.join(folder_to_save,'{}_{}.label.gii'.format(str(id).split('.')[0],epoch)))

def save_segmentation_results_UKB_test(config,predictions,folder_to_save, epoch):

    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass

    test_ids = pd.read_csv(os.path.join(config['data']['path_to_workdir'],'labels/UKB/cortical_metrics/segmentation/half/test.csv')).ids
    for i, id in enumerate(test_ids):
        save_label_UKB(config['data']['path_to_data'],predictions[i],os.path.join(folder_to_save,'{}_{}.label.gii'.format(str(id).split('.')[0],epoch)))


def save_segmentation_results_MindBoggle(config,predictions,folder_to_save, epoch):

    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass
    
    val_ids = pd.read_csv(os.path.join(config['data']['path_to_workdir'],'labels/MindBoggle/cortical_metrics/segmentation/half/val.csv')).ids
    
    for i, id in enumerate(val_ids):
        save_label_MindBoggle(config['data']['path_to_data'],predictions[i],os.path.join(folder_to_save,'{}_{}.label.gii'.format(str(id).split('.')[0],epoch)))


def save_segmentation_results_MindBoggle_test(config,predictions,folder_to_save, epoch):

    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass
    
    test_ids = pd.read_csv(os.path.join(config['data']['path_to_workdir'],'labels/MindBoggle/cortical_metrics/segmentation/half/test.csv')).ids
    
    for i, id in enumerate(test_ids):
        save_label_MindBoggle(config['data']['path_to_data'],predictions[i],os.path.join(folder_to_save,'{}_{}.label.gii'.format(str(id).split('.')[0],epoch)))

def save_label_UKB(path_to_data, data, filename):
    label =nb.load(os.path.join(path_to_data,'UKB/metrics/resample_segmentation_maps/1033131.L.aparc.ico6_fs_LR.label.gii'))
    label.darrays[0].data = data
    nb.save(label,filename)

def save_label_MindBoggle(path_to_data,data, filename):
    label =nb.load(os.path.join(path_to_data,'MindBoggle/mindboggle_resample_labels_ico6/lh.labels.HLN-12-5.ico6.DKT31.manual.label.gii'))
    label.darrays[0].data = data
    nb.save(label,filename)



def load_weights_imagenet(state_dict,state_dict_imagenet,nb_layers):

    state_dict['mlp_head.0.weight'] = state_dict_imagenet['norm.weight'].data
    state_dict['mlp_head.0.bias'] = state_dict_imagenet['norm.bias'].data

    # transformer blocks
    for i in range(nb_layers):
        state_dict['transformer.layers.{}.0.norm.weight'.format(i)] = state_dict_imagenet['blocks.{}.norm1.weight'.format(i)].data
        state_dict['transformer.layers.{}.0.norm.bias'.format(i)] = state_dict_imagenet['blocks.{}.norm1.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.norm.weight'.format(i)] = state_dict_imagenet['blocks.{}.norm2.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.norm.bias'.format(i)] = state_dict_imagenet['blocks.{}.norm2.bias'.format(i)].data

        state_dict['transformer.layers.{}.0.fn.to_qkv.weight'.format(i)] = state_dict_imagenet['blocks.{}.attn.qkv.weight'.format(i)].data

        state_dict['transformer.layers.{}.0.fn.to_out.0.weight'.format(i)] = state_dict_imagenet['blocks.{}.attn.proj.weight'.format(i)].data
        state_dict['transformer.layers.{}.0.fn.to_out.0.bias'.format(i)] = state_dict_imagenet['blocks.{}.attn.proj.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.fn.net.0.weight'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc1.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.fn.net.0.bias'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc1.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.fn.net.3.weight'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc2.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.fn.net.3.bias'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc2.bias'.format(i)].data

    return state_dict


def save_gifti(data, filename):
    gifti_file = nb.gifti.gifti.GiftiImage()
    gifti_file.add_gifti_data_array(nb.gifti.gifti.GiftiDataArray(data))
    nb.save(gifti_file,filename)


def get_dataloaders_metrics(config, 
                    data_path):

    dataloader = config['data']['loader']

    if str(dataloader)=='metrics':
        train_loader, val_loader, test_loader = loader_metrics(data_path,config)
    else:
        raise('not implemented yet')
    
    return train_loader, val_loader, test_loader

def get_data_path(config):

    dataloader = config['data']['dataloader']
    dataset = config['data']['dataset']
    configuration = config['data']['configuration']
    modality = config['data']['modality']
    sampling = config['mesh_resolution']['sampling']
    registration = config['data']['registration']

    if str(dataloader) in ['metrics','numpy']:
        if dataset == 'dHCP':
            data_path = os.path.join(config['data']['path_to_metrics'],dataset,config['data']['folder_to_dhcp'].format(configuration))
    else:
        raise('not implemented yet')
    
    return data_path