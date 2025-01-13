#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  10 18:07:54 2022
@author: Mohamed A. Suliman
"""
import numpy as np
import torch
import math

#files_dir = '../surfaces/' 

def lat_lon_img_metrics(files_dir , moving_feat, device):
    num_ver = len(moving_feat)
    
    img_idxs = np.load(files_dir+'img_indices_'+ str(num_ver) +'.npy').astype(np.int64)
    img_weights = np.load(files_dir+'img_weights_'+ str(num_ver) +'.npy').astype(np.float32)
    
    img_idxs =torch.from_numpy(img_idxs).to(device)
    img_weights = torch.from_numpy(img_weights).to(device)    

    W = int(np.sqrt(len(img_idxs)))
    
    img = torch.sum(((moving_feat[img_idxs.flatten()]).reshape(img_idxs.shape[0], img_idxs.shape[1], moving_feat.shape[1]))*((img_weights.unsqueeze(2)).repeat(1,1,moving_feat.shape[1])),1)
    
    img = img.reshape(W, W, moving_feat.shape[1])
    
    return img
            

def bilinear_sphere_resample(rot_grid, org_img, radius, device):
        
    assert rot_grid.shape[1] == 3, "grid.shape[1] ≠ 3"
    
    rot_grid_r1 = rot_grid/radius
    
    w = org_img.shape[0]

    rot_grid_r1[:,2] = torch.clamp(rot_grid_r1[:,2].clone(), -0.9999999, 0.9999999)
    
    Theta = torch.acos(rot_grid_r1[:,2]/1.0)    
    Phi = torch.zeros_like(Theta)
    
    zero_idxs = (rot_grid_r1[:,0] == 0).nonzero(as_tuple=True)[0]
    rot_grid_r1[zero_idxs, 0] = 1e-15
    
    pos_idxs = (rot_grid_r1[:,0] > 0).nonzero(as_tuple=True)[0]
    Phi[pos_idxs] = torch.atan(rot_grid_r1[pos_idxs, 1]/rot_grid_r1[pos_idxs, 0])
    
    neg_idxs = (rot_grid_r1[:,0] < 0).nonzero(as_tuple=True)[0]
    Phi[neg_idxs] = torch.atan(rot_grid_r1[neg_idxs, 1]/rot_grid_r1[neg_idxs, 0]) + math.pi
     
    Phi = torch.remainder(Phi + 2 * math.pi, 2*math.pi)
    
    assert len(pos_idxs) + len(neg_idxs) == len(rot_grid_r1)
    
    u = Phi/(2*math.pi/(w-1))
    v = Theta/(math.pi/(w-1))
        
    v = torch.clamp(v, 0.0000001, org_img.shape[1]-1.00000001).to(device)
    u = torch.clamp(u, 0.0000001, org_img.shape[1]-1.1).to(device)
    
    u_floor = torch.floor(u)
    u_ceil = u_floor + 1
    v_floor = torch.floor(v)
    v_ceil = v_floor + 1
    
    img1 = org_img[v_floor.long(), u_floor.long()]
    img2 = org_img[v_floor.long(), u_ceil.long()]
    img3 = org_img[v_ceil.long() , u_floor.long()]     
    img4 = org_img[v_ceil.long() , u_ceil.long()]
    
    Q1 = (u_ceil-u).unsqueeze(1)*img1 + (u-u_floor).unsqueeze(1)*img2    
    Q2 = (u_ceil-u).unsqueeze(1)*img3 + (u-u_floor).unsqueeze(1)*img4    
    Q  = (v_ceil-v).unsqueeze(1)*Q1 + (v-v_floor).unsqueeze(1)*Q2
       
    return Q 

def lat_lon_img_batch(files_dir , moving_feat):
    
    num_ver = moving_feat.shape[1]
    b, v, c = moving_feat.shape
    
    img_idxs = np.load(files_dir+'img_indices_'+ str(num_ver) +'.npy').astype(np.int64)
    img_weights = np.load(files_dir+'img_weights_'+ str(num_ver) +'.npy').astype(np.float32)
    print(img_idxs.shape)
    
    img_idxs =torch.from_numpy(img_idxs)
    img_weights = torch.from_numpy(img_weights)  

    W = int(np.sqrt(len(img_idxs)))
    print(W)
    
    print(moving_feat[:,img_idxs.flatten(),:].shape)
    print(b, img_idxs.shape[0], img_idxs.shape[1], moving_feat.shape[2])
    print(img_weights.unsqueeze(2).shape)
    
    img = torch.sum(((moving_feat[:,img_idxs.flatten(),:]).reshape(b, img_idxs.shape[0], img_idxs.shape[1], moving_feat.shape[2]))*((img_weights.unsqueeze(2)).repeat(1,1,1,moving_feat.shape[2])),2)
    
    img = img.reshape(b, W, W, moving_feat.shape[2])
    
    return img

def bilinear_sphere_resample_batch(rot_grid, org_img, radius):
        
    assert rot_grid.shape[1] == 3, "grid.shape[1] ≠ 3"
    
    rot_grid_r1 = rot_grid/radius
    
    w = org_img.shape[1]

    rot_grid_r1[:,2] = torch.clamp(rot_grid_r1[:,2].clone(), -0.9999999, 0.9999999)
    
    Theta = torch.acos(rot_grid_r1[:,2]/1.0)    
    Phi = torch.zeros_like(Theta)
    
    zero_idxs = (rot_grid_r1[:,0] == 0).nonzero(as_tuple=True)[0]
    rot_grid_r1[zero_idxs, 0] = 1e-15
    
    pos_idxs = (rot_grid_r1[:,0] > 0).nonzero(as_tuple=True)[0]
    Phi[pos_idxs] = torch.atan(rot_grid_r1[pos_idxs, 1]/rot_grid_r1[pos_idxs, 0])
    
    neg_idxs = (rot_grid_r1[:,0] < 0).nonzero(as_tuple=True)[0]
    Phi[neg_idxs] = torch.atan(rot_grid_r1[neg_idxs, 1]/rot_grid_r1[neg_idxs, 0]) + math.pi
     
    Phi = torch.remainder(Phi + 2 * math.pi, 2*math.pi)
    
    assert len(pos_idxs) + len(neg_idxs) == len(rot_grid_r1)
    
    u = Phi/(2*math.pi/(w-1))
    v = Theta/(math.pi/(w-1))
        
    v = torch.clamp(v, 0.0000001, org_img.shape[2]-1.00000001)
    u = torch.clamp(u, 0.0000001, org_img.shape[2]-1.1)
    
    u_floor = torch.floor(u)
    u_ceil = u_floor + 1
    v_floor = torch.floor(v)
    v_ceil = v_floor + 1
    
    img1 = org_img[:,v_floor.long(), u_floor.long()]
    img2 = org_img[:,v_floor.long(), u_ceil.long()]
    img3 = org_img[:,v_ceil.long() , u_floor.long()]     
    img4 = org_img[:,v_ceil.long() , u_ceil.long()]
    
    Q1 = (u_ceil-u).unsqueeze(1)*img1 + (u-u_floor).unsqueeze(1)*img2    
    Q2 = (u_ceil-u).unsqueeze(1)*img3 + (u-u_floor).unsqueeze(1)*img4    
    Q  = (v_ceil-v).unsqueeze(1)*Q1 + (v-v_floor).unsqueeze(1)*Q2
       
    return Q

def nearest_neighbour_sphere_resample(rot_grid, org_img, radius, device):
        
    assert rot_grid.shape[1] == 3, "grid.shape[1] ≠ 3"
    
    rot_grid_r1 = rot_grid/radius
    
    w = org_img.shape[0]

    rot_grid_r1[:,2] = torch.clamp(rot_grid_r1[:,2].clone(), -0.9999999, 0.9999999)
    
    Theta = torch.acos(rot_grid_r1[:,2]/1.0)    
    Phi = torch.zeros_like(Theta)
    
    zero_idxs = (rot_grid_r1[:,0] == 0).nonzero(as_tuple=True)[0]
    rot_grid_r1[zero_idxs, 0] = 1e-15
    
    pos_idxs = (rot_grid_r1[:,0] > 0).nonzero(as_tuple=True)[0]
    Phi[pos_idxs] = torch.atan(rot_grid_r1[pos_idxs, 1]/rot_grid_r1[pos_idxs, 0])
    
    neg_idxs = (rot_grid_r1[:,0] < 0).nonzero(as_tuple=True)[0]
    Phi[neg_idxs] = torch.atan(rot_grid_r1[neg_idxs, 1]/rot_grid_r1[neg_idxs, 0]) + math.pi
     
    Phi = torch.remainder(Phi + 2 * math.pi, 2*math.pi)
    
    assert len(pos_idxs) + len(neg_idxs) == len(rot_grid_r1)
    
    u = Phi/(2*math.pi/(w-1))
    v = Theta/(math.pi/(w-1))
        
    v = torch.clamp(v, 0.0000001, org_img.shape[1]-1.00000001).to(device)
    u = torch.clamp(u, 0.0000001, org_img.shape[1]-1.1).to(device)
    
    u_floor = torch.floor(u)
    u_ceil = u_floor + 1
    v_floor = torch.floor(v)
    v_ceil = v_floor + 1
    
    img1 = org_img[v_floor.long(), u_floor.long()]
    img2 = org_img[v_floor.long(), u_ceil.long()]
    img3 = org_img[v_ceil.long() , u_floor.long()]     
    img4 = org_img[v_ceil.long() , u_ceil.long()]
    
    Q1 = (u_ceil-u).unsqueeze(1)*img1 + (u-u_floor).unsqueeze(1)*img2    
    Q2 = (u_ceil-u).unsqueeze(1)*img3 + (u-u_floor).unsqueeze(1)*img4    
    Q  = (v_ceil-v).unsqueeze(1)*Q1 + (v-v_floor).unsqueeze(1)*Q2
       
    return Q 