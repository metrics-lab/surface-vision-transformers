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

import nibabel as nb
import os
import torch
from datetime import datetime
import pandas as pd

from tools.metrics_loader.dataloader import loader_metrics_segmentation

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from warmup_scheduler import GradualWarmupScheduler

def logging_ms_sit(config, pretraining=False):

    if pretraining:
        folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],'pretraining',config['data']['task'],config['SSL'],config['mesh_resolution']['ico_grid'],config['data']['configuration'])
    
    else:
        if config['data']['task'] =='segmentation':
            folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],config['data']['modality'],config['data']['task'],'{}_mask'.format(config['data']['masking_preprocess']),config['mesh_resolution']['ico_grid'],config['data']['configuration'])
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

    dataloader = config['data']['dataloader']
    dataset = config['data']['dataset']
    task = config['data']['task']
    modality = config['data']['modality']
    configuration = config['data']['configuration']

    if str(dataloader) == 'metrics':
        if dataset == 'UKB':
            if modality == 'cortical_metrics':
                if task == 'segmentation_msmall':
                    data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample_msmall/')
                    labels_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/resample_segmentation_maps')
                elif task == 'segmentation':
                    if config['data']['masking_preprocess']:
                        data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample_{}_mask/'.format(config['data']['masking_preprocess']))
                        labels_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/resample_segmentation_maps_{}_mask'.format(config['data']['masking_preprocess']))  
                    else:
                        data_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/merged_resample/')
                        labels_path = os.path.join(config['data']['path_to_data'],dataset,'metrics/resample_segmentation_maps')  
                                
        elif dataset == 'MindBoggle':
            if modality == 'cortical_metrics':
                if task == 'segmentation':
                    if config['data']['masking_preprocess']:
                        data_path = os.path.join(config['data']['path_to_data'],dataset,'{}/mindboggle_merged_metrics_{}_mask'.format(configuration,config['data']['masking_preprocess']))
                        labels_path = os.path.join(config['data']['path_to_data'],dataset,'{}/mindboggle_resample_labels_ico6_{}_mask'.format(configuration,config['data']['masking_preprocess'])) 
                    else:
                        data_path = os.path.join(config['data']['path_to_data'],dataset,'{}/mindboggle_merged_metrics'.format(configuration))
                        labels_path = os.path.join(config['data']['path_to_data'],dataset,'{}/mindboggle_resample_labels_ico6'.format(configuration)) 
                          
    else:
        raise('not implemented yet')
    
    return data_path, labels_path





def get_dataloaders_segmentation(config, 
                    data_path,
                    labels_path,):

    dataloader = config['data']['dataloader']
    sampler = config['training']['sampler']
    bs = config['training']['bs']
    bs_val = config['training']['bs_val']
    modality = config['data']['modality']

    if str(dataloader)=='metrics':
        if str(modality) == 'cortical_metrics' or str(modality) == 'memory_task':
            train_loader, val_loader, test_loader = loader_metrics_segmentation(data_path,labels_path,config)
        else:
            raise('not implemented yet')
    else:
        raise('not implemented yet')
    
    return train_loader, val_loader, test_loader


def get_dimensions(config):

    modality = config['data']['modality']
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['ico_{}_grid'.format(ico_grid)]['num_patches']
    num_vertices = config['ico_{}_grid'.format(ico_grid)]['num_vertices']

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
    
    if config['data']['hemi_part']=='all':
        val_ids = pd.read_csv(os.path.join(config['data']['path_to_workdir'],'labels/MindBoggle/cortical_metrics/segmentation/half/val.csv')).ids
    elif config['data']['hemi_part']=='left':
        val_ids = pd.read_csv(os.path.join(config['data']['path_to_workdir'],'labels/MindBoggle/cortical_metrics/segmentation/half/val_L.csv')).ids
    elif config['data']['hemi_part']=='right':
        val_ids = pd.read_csv(os.path.join(config['data']['path_to_workdir'],'labels/MindBoggle/cortical_metrics/segmentation/half/val_R.csv')).ids

    for i, id in enumerate(val_ids):
        save_label_MindBoggle(config['data']['path_to_data'],predictions[i],os.path.join(folder_to_save,'{}_{}.label.gii'.format(str(id).split('.')[0],epoch)))


def save_segmentation_results_MindBoggle_test(config,predictions,folder_to_save, epoch):

    try:
        os.makedirs(folder_to_save,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save))
    except OSError:
        pass
    
    if config['data']['hemi_part']=='all':
        test_ids = pd.read_csv(os.path.join(config['data']['path_to_workdir'],'labels/MindBoggle/cortical_metrics/segmentation/half/test.csv')).ids
    elif config['data']['hemi_part']=='left':
        test_ids = pd.read_csv(os.path.join(config['data']['path_to_workdir'],'labels/MindBoggle/cortical_metrics/segmentation/half/test_L.csv')).ids
    elif config['data']['hemi_part']=='right':
        test_ids = pd.read_csv(os.path.join(config['data']['path_to_workdir'],'labels/MindBoggle/cortical_metrics/segmentation/half/test_R.csv')).ids

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

