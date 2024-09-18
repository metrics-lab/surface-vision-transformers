# -*- coding: utf-8 -*-
# @Author: Simon Dahan
# @Last Modified time: 2022-01-12 15:37:28

'''
This file is used to preprocess data surface metrics into triangular patches
filling the entire surface. 
inputs: (M,C) - M mesh vertices; C channels
outputs: (N,L,V,C) - N subjects; L sequence lenght; V number of vertices per patch; C channels
'''

import numpy as np
import pandas as pd
import nibabel as nb

import yaml
import os
import argparse

def main(config):

    print('')
    print('#'*30)
    print('Starting: preprocessing script')
    print('#'*30)

    #### PARAMETERS #####

    ico = config['resolution']['ico']
    sub_ico = config['resolution']['sub_ico']

    configuration = config['data']['configuration']
    split = config['data']['split']
    num_channels = config['data']['channels']
    split = config['data']['split']
    task = config['data']['task']
    output_folder = config['output']['folder'].format(task,configuration)

    path_to_data = config['data']['data_path']
    label_path = config['data']['label_path']

    num_vertices = config['sub_ico_{}'.format(sub_ico)]['num_vertices']
    num_patches = config['sub_ico_{}'.format(sub_ico)]['num_patches']

    print('')
    print('Task: {} - Split: {} - Data: {}'.format(task,split,configuration))
    print('')

    ####

    ids = pd.read_csv(os.path.join(label_path, '{}/{}.csv'.format(task,split)))['ids'].to_numpy().reshape(-1)
    labels = pd.read_csv(os.path.join(label_path,'{}/{}.csv'.format(task,split)))['labels'].to_numpy().reshape(-1)
    num_subjects = ids.shape[0]

    means = np.load(os.path.join(config['data']['label_path'],'{}/{}/means.npy'.format(task,configuration)))
    stds = np.load(os.path.join(config['data']['label_path'],'{}/{}/stds.npy'.format(task,configuration)))
    
    print('')

    data = []

    for i,id in enumerate(ids):
        print(id)
        filename = os.path.join(path_to_data,'regression_{}_space_features'.format(configuration),'sub-{}_ses-{}_L.shape.gii'.format(id.split('_')[0],id.split('_')[1]))
        data.append(np.array(nb.load(filename).agg_data())[:num_channels,:])
        filename = os.path.join(path_to_data,'regression_{}_space_features'.format(configuration),'sub-{}_ses-{}_R.shape.gii'.format(id.split('_')[0],id.split('_')[1]))
        data.append(np.array(nb.load(filename).agg_data())[:num_channels,:])
        
    data = np.asarray(data)

    ## data normalisation 
    normalised_data = (data - means.reshape(1,num_channels,1))/stds.reshape(1,num_channels,1)

    indices_mesh_triangles = pd.read_csv('../utils/triangle_indices_ico_{}_sub_ico_{}.csv'.format(ico,sub_ico))

    #shape of the data is nbr_subjects * 20 trianglas * nbr_vertices_per_triangle * channels
    data = np.zeros((num_subjects*2, num_channels, num_patches, num_vertices))

    for i,id in enumerate(ids):
        print(id)
        for j in range(num_patches):
            indices_to_extract = indices_mesh_triangles[str(j)].to_numpy()
            data[i,:,j,:] = normalised_data[2*i][:,indices_to_extract]
            data[i+num_subjects,:,j,:] = normalised_data[2*i+1][:,indices_to_extract]
    
    print('')
    print('#'*30)
    print('#Saving: {} {} data'.format(split,configuration))
    print('#'*30)
    print('')

    try:
        os.makedirs(output_folder,exist_ok=False)
        print('Creating folder: {}'.format(output_folder))
    except OSError:
        print('folder already exist: {}'.format(output_folder))
    
    filename = os.path.join(output_folder,'{}_data.npy'.format(split,configuration))
    np.save(filename,data)
    filename = os.path.join(output_folder,'{}_labels.npy'.format(split,configuration))
    labels = np.concatenate((labels,labels))
    np.save(filename,labels)

    print('')
    print(data.shape,labels.shape)

if __name__ == '__main__':

    # Set up argument parser
        
    parser = argparse.ArgumentParser(description='preprocessing HCP data for patching')
    
    parser.add_argument(
                        'config',
                        type=str,
                        default='./config/hparams.yml',
                        help='path where the data is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    main(config)