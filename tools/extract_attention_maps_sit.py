import os
import torch
import argparse
import yaml
import datetime
import subprocess
import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import numpy as np
import nibabel as nb
import pandas as pd

from einops import rearrange
import torch.nn.functional as F

import fnmatch
from typing import Union, Optional, List

from models.sit import SiT
from tools.utils import get_dataloaders_metrics, save_gifti


class AttentionExtract(torch.nn.Module):
    # defaults should cover a significant number of timm models with attention maps.
    default_module_names = ['*attend']

    def __init__(
            self,
            model: Union[torch.nn.Module],
            names: Optional[List[str]] = None,
            mode: str = 'eval',
            method: str = 'fx',
            hook_type: str = 'forward',
    ):
        """ Extract attention maps (or other activations) from a model by name.

        Args:
            model: Instantiated model to extract from.
            names: List of concrete or wildcard names to extract. Names are nodes for fx and modules for hooks.
            mode: 'train' or 'eval' model mode.
            method: 'fx' or 'hook' extraction method.
            hook_type: 'forward' or 'forward_pre' hooks used.
        """
        super().__init__()
        assert mode in ('train', 'eval')
        if mode == 'train':
            model = model.train()
        else:
            model = model.eval()

        assert method in ('fx', 'hook')
        if method == 'hook':
            # names are module names
            assert hook_type in ('forward', 'forward_pre')
            from timm.models._features import FeatureHooks

            module_names = [n for n, m in model.named_modules()]
            matched = []
            names = names or self.default_module_names
            for n in names:
                matched.extend(fnmatch.filter(module_names, n))
            if not matched:
                raise RuntimeError(f'No module names found matching {names}.')

            self.model = model
            self.hooks = FeatureHooks(matched, model.named_modules(), default_hook_type=hook_type)

        self.names = matched
        self.mode = mode
        self.method = method

    def forward(self, x):
        if self.hooks is not None:
            self.model(x)
            output = self.hooks.get_output(device=x.device)
        return output
    

class AttentionMaps(torch.nn.Module):
    # defaults should cover a significant number of timm models with attention maps.

    def __init__(
            self,
            model,
            config,
            device,
                    ):
        super().__init__()
        self.model = model
        self.config = config
        self.device = device

    def forward(self, x):

        out = self.model(x)
        attention_maps_np = []
        for n, t in out.items():
            #print(n, t.shape)
            attention_maps_np.append(t.cpu().detach().numpy())
        
        return np.concatenate(attention_maps_np,axis=0)
        
   
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ViT')

    parser.add_argument(
                        'config',
                        type=str,
                        default='./config/hparams.yml',
                        help='path where the config file is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    path_to_ckpt =  config['path_to_ckpt']
    device = torch.device("cuda:{}".format(config['gpu']) if torch.cuda.is_available() else "cpu")

    with open(os.path.join(path_to_ckpt,'hparams.yml')) as f:
        config_ckpt = yaml.safe_load(f)
        
    sub_ico = config_ckpt['mesh_resolution']['ico_grid']
    ico = config_ckpt['mesh_resolution']['ico_mesh']
    num_patches = config_ckpt['sub_ico_{}'.format(sub_ico)]['num_patches']
    num_channels = config_ckpt['transformer']['num_channels']
    path_to_template = config_ckpt['data']['path_to_template']
    mask = np.array(nb.load('{}/week-40_hemi-left_space-dhcpSym_dens-40k_desc-medialwallsymm_mask.shape.gii'.format(config_ckpt['data']['path_to_template'] )).agg_data())

    triangle_indices = pd.read_csv('../patch_extraction/triangle_indices_ico_{}_sub_ico_{}.csv'.format(ico,sub_ico))   

    #import pdb;pdb.set_trace()
    
    ##############################################################
    # Load the data
    
    config_ckpt['training']['bs']=1
    config_ckpt['training']['bs_val']=1
    config_ckpt['data']['loader']='metrics'
    
    data_path = config_ckpt['data']['path_to_metrics'].format(config_ckpt['data']['dataset'],config_ckpt['data']['configuration'])
    train_loader, val_loader, test_loader = get_dataloaders_metrics(config_ckpt,data_path)
    
    if config['split_to_test'] == 'train':
        print('running segmentation on the train set')
        loader = train_loader
    elif config['split_to_test'] == 'val':
        print('running segmentation on the validation set')
        loader = val_loader
    elif config['split_to_test'] == 'test':
        print('running segmentation on the test set')
        loader = test_loader
    else:
        raise('Not implemented')

    ##############################################################
    # Define the model
    # Load the weights from the ckpt

    #T, N, V, use_bottleneck, bottleneck_dropout = get_dimensions(config_ckpt)

    model = SiT(dim=config_ckpt['transformer']['dim'],
                depth=config_ckpt['transformer']['depth'],
                heads=config_ckpt['transformer']['heads'],
                mlp_dim=config_ckpt['transformer']['mlp_dim'],
                pool=config_ckpt['transformer']['pool'], 
                num_patches=num_patches,
                num_classes=config_ckpt['transformer']['num_classes'],
                num_channels=num_channels,
                num_vertices=config_ckpt['sub_ico_{}'.format(sub_ico)]['num_vertices'],
                dim_head=config_ckpt['transformer']['dim_head'],
                dropout=config_ckpt['transformer']['dropout'],
                emb_dropout=config_ckpt['transformer']['emb_dropout'],
                )
    

    weights = torch.load(os.path.join(path_to_ckpt,'checkpoint.pth'),map_location=device)
    print('** Loading checkpoint **')
    model.load_state_dict(weights,strict=True)
    model.to(device)

    ##############################################################
    # Instatiate the Extract Attention class

    hook_model = AttentionExtract(model, method='hook')

    ##############################################################
    # Instantiate the Attention Maps class

    hook_attention_maps = AttentionMaps(hook_model, config_ckpt, device)

    ##############################################################
    # Logging

    output_dir = './outputs/attention_maps'
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_dir, path_to_ckpt.split('/')[-1], current_time)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    ##############################################################
    # Extract Attention
    
    list_ids = config['subject_id']
    
    with torch.no_grad():
        for id, data in enumerate(loader):
            if id in list_ids:
                print('**** subject {} ****'.format(id))
                inputs, targets = data[0].to(device), data[1].to(device)
                out_maps = hook_attention_maps(inputs) # this is after the softmax

                nh = out_maps.shape[1] # number of head
                nl = out_maps.shape[0] # number of layers
                # we keep only the output patch attention
                attentions = out_maps[:, :, 1:, 0].reshape(nh, -1,nl)
                
                ### saving all maps
                reconstructed_sphere_am = np.zeros((40962,nl,nh),dtype=np.float32)
                new_inputs = np.transpose(attentions,(1,2,0))
                for i in range(num_patches):
                    indices_to_extract = triangle_indices[str(i)].values
                    reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

                import pdb;pdb.set_trace()
                reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
                reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
                save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_all_layers.shape.gii'.format(id)))
                p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_all_layers.shape.gii'.format(id)), 'CORTEX_LEFT'])
                p1.wait()

                ### averaging maps per layer
                reconstructed_sphere_am = np.zeros((40962,1,nh),dtype=np.float32)
                new_inputs = np.transpose(attentions,(1,2,0))
                new_inputs =np.mean(new_inputs, axis=1, keepdims=True)
                for i in range(num_patches):
                    indices_to_extract = triangle_indices[str(i)].values
                    reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

                reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
                reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
                save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_avg_all_layers.shape.gii'.format(id)))
                p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers.shape.gii'.format(id)), 'CORTEX_LEFT'])
                p1.wait()

                ### MAX maps per layer
                reconstructed_sphere_am = np.zeros((40962,1,nh),dtype=np.float32)
                new_inputs = np.transpose(attentions,(1,2,0))
                new_inputs =np.max(new_inputs, axis=1, keepdims=True)
                for i in range(num_patches):
                    indices_to_extract = triangle_indices[str(i)].values
                    reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

                reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
                reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
                save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_max_all_layers_per_head.shape.gii'.format(id)))
                p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_max_all_layers_per_head.shape.gii'.format(id)), 'CORTEX_LEFT'])
                p1.wait()

                ### avg maps per head
                reconstructed_sphere_am = np.zeros((40962,nl,1),dtype=np.float32)
                new_inputs = np.transpose(attentions,(1,2,0))
                new_inputs =np.max(new_inputs, axis=2, keepdims=True)
                for i in range(num_patches):
                    indices_to_extract = triangle_indices[str(i)].values
                    reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

                reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
                reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
                save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_per_head.shape.gii'.format(id)))
                p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_per_head.shape.gii'.format(id)), 'CORTEX_LEFT'])
                p1.wait()

                ### averaging everything
                reconstructed_sphere_am = np.zeros((40962,1,1),dtype=np.float32)
                new_inputs = np.transpose(attentions,(1,2,0))
                new_inputs =np.mean(new_inputs, axis=1, keepdims=True)
                new_inputs =np.mean(new_inputs, axis=2, keepdims=True)
                for i in range(num_patches):
                    indices_to_extract = triangle_indices[str(i)].values
                    reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

                reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
                reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
                save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_avg_all.shape.gii'.format(id)))
                p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_avg_all.shape.gii'.format(id)), 'CORTEX_LEFT'])
                p1.wait()

                ### maxing everything
                reconstructed_sphere_am = np.zeros((40962,1,1),dtype=np.float32)
                new_inputs = np.transpose(attentions,(1,2,0))
                new_inputs =np.max(new_inputs, axis=1, keepdims=True)
                new_inputs =np.max(new_inputs, axis=2, keepdims=True)
                for i in range(num_patches):
                    indices_to_extract = triangle_indices[str(i)].values
                    reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

                reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
                reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
                save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_max_all.shape.gii'.format(id)))
                p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_max_all.shape.gii'.format(id)), 'CORTEX_LEFT'])
                p1.wait()


                ### save input data
                new_inputs =  np.transpose(inputs.detach().cpu().numpy(),(0,2,1,3))
                input_sphere = np.zeros((40962,num_channels),dtype=np.float32)
                for i in range(num_patches):
                    indices_to_extract = triangle_indices[str(i)].values
                    input_sphere[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()
                save_gifti(input_sphere, os.path.join(output_dir, 'subject_{}_input_metrics.shape.gii'.format(id)) )
                import pdb;pdb.set_trace()
                p1 = subprocess.Popen(['wb_command', '-set-structure', os.path.join(output_dir, 'subject_{}_input_metrics.shape.gii'.format(id)), 'CORTEX_LEFT'])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_all_layers.shape.gii'.format(id)),'{}/ico-6-L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_all_layers{}.shape.gii'.format(id,num_patches))])
                p1.wait()
                import pdb;pdb.set_trace()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_all_layers{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6-L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_all_layers_resamp.shape.gii'.format(id))])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers.shape.gii'.format(id)),'{}/ico-6-L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_avg_all_layers{}.shape.gii'.format(id,num_patches))])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6-L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_resamp.shape.gii'.format(id))])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_max_all_layers_per_head.shape.gii'.format(id)),'{}/ico-6-L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_max_all_layers_per_head{}.shape.gii'.format(id,num_patches))])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_max_all_layers_per_head{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6-L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_max_all_layers_per_head_resamp.shape.gii'.format(id))])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_per_head.shape.gii'.format(id)),'{}/ico-6-L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_per_head{}.shape.gii'.format(id,num_patches))])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_per_head{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6-L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_per_head_resamp.shape.gii'.format(id))])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all.shape.gii'.format(id)),'{}/ico-6-L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_avg_all{}.shape.gii'.format(id,num_patches))])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6-L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_avg_all_resamp.shape.gii'.format(id))])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_max_all.shape.gii'.format(id)),'{}/ico-6-L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_max_all{}.shape.gii'.format(id,num_patches))])
                p1.wait()

                p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_max_all{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6-L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'subject_{}_attention_max_all.shape.gii'.format(id))])
                p1.wait()
                
                import pdb;pdb.set_trace()
