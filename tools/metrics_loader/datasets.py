import os
import random
import torch

import numpy as np
import nibabel as nb
import pandas as pd


from torch.utils.data import Dataset
from logging import raiseExceptions


class dataset_cortical_surfaces_segmentation(Dataset):
    def __init__(self, 
                data_path,
                labels_path,
                config,
                split,
                ):

        super().__init__()

        ################################################
        ##############       CONFIG       ##############
        ################################################

        task = config['data']['task']
        sampling = config['mesh_resolution']['sampling']
        ico = config['mesh_resolution']['ico_mesh']
        sub_ico = config['mesh_resolution']['ico_grid']

        self.configuration = config['data']['configuration']
        self.filedir =  data_path
        self.split = split
        self.filedir_labels = labels_path
        self.dataset = config['data']['dataset']
        self.hemi = config['data']['hemi']
        self.split = split
        self.dataset = config['data']['dataset']
        self.augmentation = config['augmentation']['prob_augmentation']
        self.use_cross_val = config['training']['use_cross_validation']
        self.normalise = config['data']['normalise']
        self.clipping = config['data']['clipping']
        self.modality = config['data']['modality']
        self.path_to_workdir = config['data']['path_to_workdir']
        self.path_to_template = config['data']['path_to_template']
        self.script = config['SCRIPT']
        self.device = 'cpu'
        self.warps_ico = config['augmentation']['warp_ico']
        self.nbr_vertices = config['ico_{}_grid'.format(sub_ico)]['num_vertices']
        self.nbr_patches = config['ico_{}_grid'.format(sub_ico)]['num_patches']

        if config['MODEL'] == 'sit' or config['MODEL']=='ms-sit':
            self.patching=True
            self.channels = config['transformer']['channels']
            self.num_channels = len(self.channels)
        elif config['MODEL']== 'spherical-unet':
            self.patching=False 
            self.channels = config['spherical-unet']['channels']
            self.num_channels = len(self.channels)
        elif config['MODEL']== 'monet':
            self.patching=False 
            self.channels = config['monet']['channels']
            self.num_channels = len(self.channels)
        else:
            raiseExceptions('model not implemented yet')
    

        ################################################
        ##############       LABELS       ##############
        ################################################

        if config['data']['modality'] == 'cortical_metrics':
            if config['data']['hemi_part']=='all':
                self.data_info = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(
                                                                             config['data']['path_to_workdir'],
                                                                             self.dataset,task,
                                                                             self.hemi,
                                                                             split))
            elif config['data']['hemi_part']=='left':
                self.data_info = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}_L.csv'.format(
                                                                            config['data']['path_to_workdir'],
                                                                            self.dataset,task,
                                                                             self.hemi,
                                                                             split))
            elif config['data']['hemi_part']=='right':
                self.data_info = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}_R.csv'.format(
                                                                            config['data']['path_to_workdir'],
                                                                            self.dataset,task,
                                                                             self.hemi,
                                                                             split))

        self.filenames = self.data_info['ids']
        self.labels = self.data_info['labels']
     
        ###################################################
        ##############       NORMALISE       ##############
        ###################################################

        if self.normalise=='group-standardise' and self.use_cross_val:
        
            self.means = np.load('{}/labels/{}/cortical_metrics/cv_5/{}/{}/{}/means.npy'.format(
                                                                                config['data']['path_to_workdir'],
                                                                                self.dataset,task,
                                                                                self.hemi,
                                                                                self.configuration))
            self.stds = np.load('{}/labels/{}/cortical_metrics/cv_5/{}/{}/{}/stds.npy'.format(
                                                                                config['data']['path_to_workdir'],
                                                                                self.dataset,task,
                                                                                self.hemi,
                                                                                self.configuration))
                                                                            
        elif self.normalise=='group-standardise' and not self.use_cross_val:
        
            self.means = np.load('{}/labels/{}/cortical_metrics/{}/{}/{}/means.npy'.format(
                                                                                config['data']['path_to_workdir'],
                                                                                self.dataset,task,
                                                                                self.hemi,
                                                                                self.configuration))
            self.stds = np.load('{}/labels/{}/cortical_metrics/{}/{}/{}/stds.npy'.format(
                                                                                config['data']['path_to_workdir'],
                                                                                self.dataset,task,
                                                                                self.hemi,
                                                                                self.configuration))
        
        ########################################################################
        ##############       DATA AUGMENTATION & PROCESSING       ##############
        ########################################################################

        self.triangle_indices = pd.read_csv('{}/patch_extraction/{}/triangle_indices_ico_{}_sub_ico_{}.csv'.format(config['data']['path_to_workdir'],sampling,ico,sub_ico))
        
        #config, augmentation
        if self.augmentation:
            self.rotation = config['augmentation']['prob_rotation']
            self.max_degree_rot = config['augmentation']['max_abs_deg_rotation']
            self.shuffle = config['augmentation']['prob_shuffle']
            self.warp = config['augmentation']['prob_warping']
            self.coord_ico6 = np.array(nb.load('{}/coordinates_ico_6_L.shape.gii'.format(self.path_to_template)).agg_data()).T      
        
        self.apply_symmetry = config['augmentation']['apply_symmetry']  
        self.symmetry_angle = config['augmentation']['symmetry_angle']  


        if config['mesh_resolution']['reorder']:
            new_order_indices = np.load('{}/patch_extraction/order_patches/order_ico{}.npy'.format(config['data']['path_to_workdir'], sub_ico))
            d = {str(new_order_indices[i]):str(i) for i in range(len(self.triangle_indices.columns))}
            self.triangle_indices = self.triangle_indices[list([str(i) for i in new_order_indices])]
            self.triangle_indices = self.triangle_indices.rename(columns=d)
        
        self.masking = config['data']['masking']    
        if self.masking and self.dataset == 'dHCP' and self.configuration == 'template' : # for dHCP
            if split == 'train':
                print('Masking the cut: dHCP mask')
            self.mask = np.array(nb.load('{}/week-40_hemi-left_space-dhcpSym_dens-40k_desc-medialwallsymm_mask.shape.gii'.format(self.path_to_template)).agg_data())
        elif self.masking and self.dataset == 'dHCP' and self.configuration == 'native':
            if split == 'train':
                print('Masking the cut: dHCP - using native subject masks')
        elif self.masking and self.dataset == 'UKB': # for UKB
            if split == 'train':
                print('Masking the cut: UKB mask')
            self.mask = np.array(nb.load('{}/L.atlasroi.ico6_fs_LR.shape.gii'.format(self.path_to_template)).agg_data())
        elif self.masking and self.dataset == 'HCP': # for HCP
            if split == 'train':
                print('Masking the cut: HCP mask')
            self.mask = np.array(nb.load('{}/L.atlasroi.40k_fs_LR.shape.gii'.format(self.path_to_template)).agg_data())
        else:
            if split == 'train':
                print('Masking the cut: NO')
            

    def __len__(self):
        return len(self.filenames)
    

    def __getitem__(self,idx):

        ############
        # 1. load input metrics
        # 2. select input channels
        # 3. correct artefacts in the data
        # 4. apply symmetry to the mesh (if apply symmetry)
        # 5. clip the input data (if clipping)
        # 6. normalise the date (if normalise)
        # 7. apply augmentation
        # 8. get sequence of patches
        ############

        ###data
        path = os.path.join(self.filedir,self.filenames.iloc[idx])
        data =  np.array(nb.load(path).agg_data())
        if len(data.shape)==1:
            data = np.expand_dims(data,0)
        data = data[self.channels,:]

        ### label
        path_to_label = os.path.join(self.filedir_labels,self.labels.iloc[idx])
        label = np.array(nb.load(path_to_label).agg_data(),dtype=np.int64) 
        if self.dataset == 'UKB':
            label[label==-1]=0 # correct artefact
            label[label==35]=0 # correct artefact
        elif self.dataset == 'MindBoggle':
            label[label==-1]=0 # correct artefacts
            label[label==32]=0 # correct artefacts
            label[label==33]=0 # correct artefacts
            label[label==34]=0 # correct artefacts
            label[label==35]=0 # correct artefacts
                
        if self.apply_symmetry:

            data,label = self._apply_symmetry(data,label)
        
        ### hemisphere
        if self.hemi == 'half':

            data = self.get_half_hemi(data)

            if (self.augmentation and self.split =='train') or (self.augmentation and self.script=='test'):
                # do augmentation or not do augmentation
                if np.random.rand() > (1-self.augmentation):
                    # chose which augmentation technique to use
                    p = np.random.rand()
                    if p < self.rotation:
                        #apply rotation
                        rotation_angle = np.round(random.uniform(-self.max_degree_rot,self.max_degree_rot),2)
                        axis = random.choice(['x','y','z'])
                        data = self.apply_rotation_metrics(data,rotation_angle, axis)
                        label = self.apply_rotation_labels(label,rotation_angle, axis)

                    
                    elif self.warp:

                        if self.rotation <= p < self.rotation+self.warp:
                            #apply warp
                            id = np.random.randint(0,100)
                            data = self.apply_non_linear_warp_metrics(data,id)
                            label = self.apply_non_linear_warp_labels(label,id)

                        elif self.rotation+self.war <= p < self.rotation+self.warp+self.shuffle:
                            #apply shuffle
                            data = self.apply_shuffle(data)
                        
            if self.patching:

                sequence = self.get_sequence(data)

                if self.augmentation and self.shuffle and self.split == 'train':
                    bool_select_patch =np.random.rand(sequence.shape[1])<self.prob_shuffle
                    copy_data = sequence[:,bool_select_patch,:]
                    ind = np.where(bool_select_patch)[0]
                    np.random.shuffle(ind)  
                    sequence[:,ind,:]=copy_data
                return (torch.from_numpy(sequence).float(),torch.from_numpy(label))

            else:
                return (torch.from_numpy(data).float(),torch.from_numpy(label))

        return (torch.from_numpy(data).float(),torch.from_numpy(label))

    
    def get_half_hemi(self,data):

        #### 2. normalising - only vertices that are not maske

        if self.clipping:
                data = self.clipping_(data)

        data = self.normalise_(data)

        return data

    def clipping_(self,data):

        if self.dataset == 'dHCP':
            lower_bounds = np.array([0.0,-0.5, -0.05, -10.0])
            upper_bounds = np.array([2.2, 0.6,2.6, 10.0 ])
            for i,channel in enumerate(self.channels):
                data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel]
                )
        elif self.dataset == 'UKB':
            lower_bounds = np.array([0.0,-0.41, -0.40, -16.50])
            upper_bounds = np.array([2.41, 0.45,5.33, 15.4])
            for i,channel in enumerate(self.channels):
                data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel])

        elif self.dataset == 'HCP' and self.modality == 'memory_task':
            lower_bounds = np.array([-10.86,-9.46,-6.39])
            upper_bounds = np.array([12.48,10.82,6.68])
            for i,channel in enumerate(self.channels):
                data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel])

        elif self.dataset == 'HCP' and self.modality == 'cortical_metrics':
            lower_bounds = np.array([-0.01,-0.41,-0.23,-1.65])
            upper_bounds = np.array([2.44,0.46,5.09,1.52])
            for i,channel in enumerate(self.channels):
                data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel])

        elif self.dataset == 'MindBoggle' and self.modality == 'cortical_metrics':
            lower_bounds = np.array([-1.69,-0.40])
            upper_bounds = np.array([1.57,0.44])
            for i,channel in enumerate(self.channels):
                data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel])

        return data

    def normalise_(self,data):

        if self.masking:
            non_masked_vertices = self.mask>0
            if self.normalise=='group-standardise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - self.means[:,self.channels,:].reshape(self.num_channels,1))/self.stds[:,self.channels,:].reshape(self.num_channels,1)
            elif self.normalise=='sub-standardise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].mean(axis=1).reshape(self.num_channels,1))/data[:,non_masked_vertices].std(axis=1).reshape(self.num_channels,1)
            elif self.normalise=='sub-normalise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].min(axis=1).reshape(self.num_channels,1))/(data[:,non_masked_vertices].max(axis=1).reshape(self.num_channels,1)- data[:,non_masked_vertices].min(axis=1).reshape(self.num_channels,1))
        
        else:
            if self.normalise=='group-standardise':
                data= (data- self.means.reshape(self.num_channels,1))/self.stds.reshape(self.num_channels,1)
            elif self.normalise=='sub-standardise':
                data = (data - data.mean(axis=1).reshape(self.num_channels,1))/data.std(axis=1).reshape(self.num_channels,1)
            elif self.normalise=='normalise':
                data = (data- data.min(axis=1).reshape(self.num_channels,1))/(data.max(axis=1).reshape(self.num_channels,1)- data.min(axis=1).reshape(self.num_channels,1))
        return data
    
    ############ AUGMENTATION ############

    def get_sequence(self,data):

        sequence = np.zeros((self.num_channels, self.nbr_patches, self.nbr_vertices))
        for j in range(self.nbr_patches):
            indices_to_extract = self.triangle_indices[str(j)].to_numpy()
            sequence[:,j,:] = data[:,indices_to_extract]
        return sequence
    
    def apply_rotation_metrics(self,data,rotation_angle,axis):

        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to(self.device),device=self.device)

        r = R.from_euler(axis, rotation_angle, degrees=True)

        new_coord = np.asarray(r.apply(self.coord_ico6),dtype=np.float32)

        rotated_moving_img = bilinear_sphere_resample(torch.Tensor(new_coord),img, 100, self.device)

        return rotated_moving_img.numpy().T
    
    def apply_rotation_labels(self,data, rotation_angle, axis):

        img = lat_lon_img_labels('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data).unsqueeze(1).to(self.device),device=self.device)

        r = R.from_euler(axis, rotation_angle, degrees=True)

        new_coord = np.asarray(r.apply(self.coord_ico6),dtype=np.float32)

        rotated_moving_img = majority_sphere_resample(torch.Tensor(new_coord),torch.round(img), 100, self.device)
        #rotated_moving_img = nearest_neighbour_sphere_resample(torch.Tensor(new_coord),torch.round(img), 100, self.device)

        return torch.squeeze(rotated_moving_img).long().numpy()

    def apply_non_linear_warp_metrics(self,data,warp_id):

        # chose one warps at random between 0 - 99

        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to(self.device),device=self.device)
        warped_grid = nb.load('{}/warps/resample_ico6_ico_{}/ico_{}_{}.surf.gii'.format(self.path_to_template,self.warps_ico,self.warps_ico,warp_id)).agg_data()
        warped_moving_img = bilinear_sphere_resample(torch.Tensor(warped_grid[0]), img, 100, self.device)
        return warped_moving_img.numpy().T
    
    def apply_non_linear_warp_labels(self,data,warp_id):

        # chose one warps at random between 0 - 99
        
        img = lat_lon_img_labels('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data).unsqueeze(1).to(self.device),device=self.device)
        warped_grid = nb.load('{}/warps/resample_ico6_ico_{}/ico_{}_{}.surf.gii'.format(self.path_to_template,self.warps_ico,self.warps_ico,warp_id)).agg_data()
        warped_moving_img = majority_sphere_resample(torch.Tensor(warped_grid[0]), torch.round(img), 100, self.device)
        #warped_moving_img = nearest_neighbour_sphere_resample(torch.Tensor(warped_grid[0]), torch.round(img), 100, self.device)
        return torch.squeeze(warped_moving_img).long().numpy()

    def _apply_symmetry(self, data, label):

        data_sym = np.zeros((data.shape[0],data.shape[1]),dtype=np.float32)
        label_sym = np.zeros(label.shape[0],dtype=np.int64)
        rot_array = np.array(np.load('{}/surfaces/rotations_array.npy'.format(self.path_to_workdir)),dtype=np.int32)

        for i in range(40962):
            data_sym[:,i] = data[:,rot_array[self.symmetry_angle][i]]
            label_sym[i] = label[rot_array[self.symmetry_angle][i]]

        return data_sym, label_sym
    
    ############ LOGGING ############

    def logging(self):

        if self.split == 'train':
                print('Using {} channels'.format(self.channels))

        if self.split == 'train':
            if self.normalise == 'sub-standardise':
                print('Normalisation: Subject-wise standardised')
            elif self.normalise == 'group-standardise':
                print('Normalisation: Group-wise standardised')
            elif self.normalise == 'normalise':
                print('Normalisation: Normalised')
            else:
                print('Normalisation: Not normalised') 

        print('')
        print('#'*30)
        print('######## Augmentation ########')
        print('#'*30)
        print('')
        if self.augmentation:
            print('Augmentation: ratio {}'.format(self.augmentation))
            if self.rotation:
                print('     rotation with probability: {} and max abs degree {}'.format(self.rotation,self.max_degree_rot))
            else:
                print('     rotations: no')
            if self.shuffle:
                print('     shuffling with probability: {}'.format(self.shuffle))
            else:
                print('     shuffling: no')
            if self.warp:
                print('     non-linear warping with probability: {}'.format(self.warp))
            else:
                print('     non-linear warping: no')
        else:
            print('Augmentation: NO')
