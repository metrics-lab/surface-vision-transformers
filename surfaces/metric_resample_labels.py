import numpy as np
import torch
import math
import random
import nibabel as nb
from scipy.spatial.transform import Rotation as R


def lat_lon_img_labels(files_dir , moving_feat, device):
    
    num_ver = len(moving_feat)
    
    img_idxs = np.load(files_dir+'img_indices_'+ str(num_ver) +'.npy').astype(np.int64)
    img_weights = np.load(files_dir+'img_weights_'+ str(num_ver) +'.npy').astype(np.float32)
    
    img_idxs =torch.from_numpy(img_idxs).to(device)
    img_weights = torch.from_numpy(img_weights).to(device)    

    W = int(np.sqrt(len(img_idxs)))
    
    img = torch.sum(((moving_feat[img_idxs.flatten()]).reshape(img_idxs.shape[0], img_idxs.shape[1], moving_feat.shape[1]))*((img_weights.unsqueeze(2)).repeat(1,1,moving_feat.shape[1])),1)
    
    img = img.reshape(W, W, moving_feat.shape[1])
    
    return img
            

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

    #x,y coordinates of surrounding points
    a1 = torch.concat([v_floor.unsqueeze(1),u_floor.unsqueeze(1)],axis=1)
    a2 = torch.concat([v_floor.unsqueeze(1),u_ceil.unsqueeze(1)],axis=1)
    a3 = torch.concat([v_ceil.unsqueeze(1),u_floor.unsqueeze(1)],axis=1)
    a4 = torch.concat([v_ceil.unsqueeze(1),u_ceil.unsqueeze(1)],axis=1)

    b = torch.concat([a1.unsqueeze(0),a2.unsqueeze(0),a3.unsqueeze(0),a4.unsqueeze(0)],axis=0)

    p = torch.concat([v.unsqueeze(1),u.unsqueeze(1)],axis=1)

    # compute distance between (v,u) points and all points
    dist = torch.sqrt(torch.sum(torch.pow((b-p),2),axis=2))

    # get list of indices with minimal distance to point (v,u)
    first, second = zip(*enumerate(torch.argmax(dist,dim=0)))

    coordinates_to_select = torch.transpose(b,1,0)[first,second,:]

    Q = org_img[coordinates_to_select[:,0].long(),coordinates_to_select[:,1].long()]

    return torch.round(Q)



def majority_sphere_resample(rot_grid, org_img, radius, device):

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
    u_floor_1 = u_floor - 1
    u_floor_2 = u_floor_1 -1
    u_floor_3 = u_floor_2 -1
    u_ceil = u_floor + 1
    u_ceil_1 = u_ceil + 1
    u_ceil_2 = u_ceil_1 + 1
    u_ceil_3 = u_ceil_2 + 1
    v_floor = torch.floor(v)
    v_floor_1 = v_floor - 1
    v_floor_2 = v_floor_1 - 1
    v_floor_3 = v_floor_2 -1 
    v_ceil = v_floor + 1
    v_ceil_1 = v_ceil + 1 
    v_ceil_2 = v_ceil_1 + 1 
    v_ceil_3 = v_ceil_2 +1

    v_ceil_1[v_ceil_1==512]=511
    u_ceil_1[u_ceil_1==512]=511
    v_ceil_2[v_ceil_2==512]=511
    u_ceil_2[u_ceil_2==512]=511
    v_ceil_2[v_ceil_2==513]=511
    u_ceil_2[u_ceil_2==513]=511
    v_ceil_3[v_ceil_3==512]=511
    u_ceil_3[u_ceil_3==512]=511
    v_ceil_3[v_ceil_3==513]=511
    u_ceil_3[u_ceil_3==513]=511
    v_ceil_3[v_ceil_3==514]=511
    u_ceil_3[u_ceil_3==514]=511



    
    img1 = org_img[v_floor.long(), u_floor.long()]
    img2 = org_img[v_floor.long(), u_ceil.long()]
    img3 = org_img[v_ceil.long() , u_floor.long()]     
    img4 = org_img[v_ceil.long() , u_ceil.long()]
    
    img5 = org_img[v_floor_1.long() , u_floor.long()]
    img6 = org_img[v_floor_1.long() , u_ceil.long()]
    img7 = org_img[v_ceil_1.long() , u_floor.long()]
    img8 = org_img[v_ceil_1.long() , u_ceil.long()]
    img9 = org_img[v_floor.long() , u_floor_1.long()]
    img10 = org_img[v_floor.long() , u_ceil_1.long()]
    img11 = org_img[v_ceil.long() , u_floor_1.long()]
    img12 = org_img[v_ceil.long() , u_ceil_1.long()]
    img13 = org_img[v_floor_1.long() , u_floor_1.long()]
    img14 = org_img[v_floor_1.long() , u_ceil_1.long()]
    img15 = org_img[v_ceil_1.long() , u_floor_1.long()]
    img16 = org_img[v_ceil_1.long() , u_ceil_1.long()]
    
    img17 = org_img[v_floor_2.long() , u_floor.long()]
    img18 = org_img[v_floor_2.long() , u_ceil.long()]
    img19 = org_img[v_floor_2.long() , u_floor_1.long()]
    img20 = org_img[v_floor_2.long() , u_ceil_1.long()]
    img21 = org_img[v_floor_2.long() , u_floor_2.long()]
    img22 = org_img[v_floor_2.long() , u_ceil_2.long()]
    img23 = org_img[v_ceil_2.long() , u_floor.long()]
    img24 = org_img[v_ceil_2.long() , u_ceil.long()]
    img25 = org_img[v_ceil_2.long() , u_floor_1.long()]
    img26 = org_img[v_ceil_2.long() , u_ceil_1.long()]
    img27 = org_img[v_ceil_2.long() , u_floor_2.long()]
    img28 = org_img[v_ceil_2.long() , u_ceil_2.long()]
    img29 = org_img[v_floor.long() , u_floor_2.long()]
    img30 = org_img[v_ceil.long() , u_floor_2.long()]
    img31 = org_img[v_floor_1.long() , u_floor_2.long()]
    img32 = org_img[v_ceil_1.long() , u_floor_2.long()]
    img33 = org_img[v_floor.long() , u_ceil_2.long()]
    img34 = org_img[v_ceil.long() , u_ceil_2.long()]    
    img35 = org_img[v_floor_1.long() , u_ceil_2.long()]
    img36 = org_img[v_ceil_1.long() , u_ceil_2.long()]
    
    img37 = org_img[v_floor_3.long() , u_floor.long()]
    img38 = org_img[v_floor_3.long() , u_ceil.long()]
    img39 = org_img[v_floor_3.long() , u_floor_1.long()]
    img40 = org_img[v_floor_3.long() , u_ceil_1.long()]
    img41 = org_img[v_floor_3.long() , u_floor_2.long()]
    img42 = org_img[v_floor_3.long() , u_ceil_2.long()]
    img43 = org_img[v_floor_3.long() , u_floor_3.long()]
    img44 = org_img[v_floor_3.long() , u_ceil_3.long()]
    img43 = org_img[v_ceil_3.long() , u_floor.long()]
    img44 = org_img[v_ceil_3.long() , u_ceil.long()]
    img45 = org_img[v_ceil_3.long() , u_floor_1.long()]
    img46 = org_img[v_ceil_3.long() , u_ceil_1.long()]
    img47 = org_img[v_ceil_3.long() , u_floor_2.long()]
    img48 = org_img[v_ceil_3.long() , u_ceil_2.long()]
    img49 = org_img[v_ceil_3.long() , u_floor_3.long()]
    img50 = org_img[v_ceil_3.long() , u_ceil_3.long()]
    
    img51 = org_img[v_floor.long() , u_floor_3.long()]
    img52 = org_img[v_ceil.long() , u_floor_3.long()]
    img53 = org_img[v_floor_1.long() , u_floor_3.long()]
    img54 = org_img[v_ceil_1.long() , u_floor_3.long()]
    img55 = org_img[v_floor_2.long() , u_floor_3.long()]
    img56 = org_img[v_ceil_2.long() , u_floor_3.long()]
    
    img57 = org_img[v_floor.long() , u_ceil_3.long()]
    img58 = org_img[v_ceil.long() , u_ceil_3.long()]    
    img59 = org_img[v_floor_1.long() , u_ceil_3.long()]
    img60 = org_img[v_ceil_1.long() , u_ceil_3.long()]
    img61 = org_img[v_floor_2.long() , u_ceil_3.long()]
    img62 = org_img[v_ceil_2.long() , u_ceil_3.long()]


    Q = torch.Tensor([torch.bincount(torch.Tensor([img1[i],img2[i],img3[i],img4[i],img5[i],img6[i],img7[i],img8[i], img9[i],img10[i],img11[i],img12[i], img13[i],img14[i],img15[i],img16[i], \
                                                   img17[i],img18[i],img19[i],img20[i],img21[i],img22[i],img23[i],img24[i], img25[i],img26[i],img27[i],img28[i],img29[i],img30[i],img31[i],img32[i], \
                                                       img33[i],img34[i],img35[i],img36[i], img37[i],img38[i],img39[i],img40[i], img41[i],img42[i],img43[i],img44[i], img45[i],img46[i],img47[i],img48[i], \
                                                           img49[i],img50[i],img51[i],img52[i],  img53[i],img54[i],img55[i],img56[i],  img57[i],img58[i],img59[i],img60[i],  img61[i],img62[i]]).long()).argmax() for i in range(40962)])

    return Q