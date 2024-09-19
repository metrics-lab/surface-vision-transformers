import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange

from einops import repeat, rearrange


'''
This implementation of SwinTransformer is greatly inspired by the official codebase: https://github.com/microsoft/Swin-Transformer
and adapted for surface analysis on icospheric meshes. 
'''

class MSSiTUNet_shifted(nn.Module):
    """
    Args:
        window_size (int): Number of patches to apply local-attention to.
    """
    

    def __init__(self,ico_init_resolution=4,num_channels=4,num_classes=1,
                    embed_dim=96, depths=[2,2,6,2],num_heads=[3,6,12,24],
                    window_size=80,window_size_factor=1, mlp_ratio=4,qkv_bias=True,qk_scale=True,
                    dropout=0, attention_dropout=0,dropout_path=0.1,
                    norm_layer=nn.LayerNorm, use_pos_emb=False,patch_norm=True,
                    use_confounds =False,device=0,reorder = False,path_to_workdir='',**kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.use_pos_emb = use_pos_emb
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers-1))   #number of features after the last layer
        self.mlp_ratio = mlp_ratio
        self.num_channels = num_channels
        self.device = device
        
        self.indices = pd.read_csv('{}/patch_extraction/triangle_indices_ico_6_sub_ico_{}.csv'.format(path_to_workdir,ico_init_resolution))
        
        if reorder:
            print('** reorder patches **')
            new_order_indices = np.load('{}/patch_extraction/reorder_patches/order_ico{}.npy'.format(path_to_workdir,ico_init_resolution))
            d = {str(new_order_indices[i]):str(i) for i in range(len(self.indices.columns))}
            self.indices = self.indices[list([str(i) for i in new_order_indices])]
            self.indices = self.indices.rename(columns=d)
        
        if isinstance(window_size,int):
            self.window_sizes = [window_size for i in range(self.num_layers)]
        elif isinstance(window_size,list):
            self.window_sizes = window_size
        print('window size: {}'.format(self.window_sizes))

        self.window_size_factor = window_size_factor 

        print('window size: {}'.format(self.window_sizes))
        print('window size factor: {}'.format(self.window_size_factor))

        if ico_init_resolution==4:
            self.num_patches = 5120
            self.num_vertices = 15
            patch_dim = self.num_vertices * self.num_channels
        elif ico_init_resolution==5:
            self.num_patches = 20480
            self.num_vertices = 6
            patch_dim = self.num_vertices * self.num_channels
            if len(self.window_sizes)==4 and self.window_sizes[-1]==320:
                print('**use global attention in the last stage**')
                self.use_global_att = [False, False, False, True]
            elif len(self.window_sizes)==5 and self.window_sizes[-1]==80:
                print('**use global attention in the last stage**')
                self.use_global_att = [False, False, False, False, True]
            else: 
                print('** NOT using global attention in the last stage**')
                self.use_global_att = [False for i in range(len(self.window_sizes))]
                
        
        # absolute position embedding
        if use_pos_emb:
            print('using positional embeddings')
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        #need another version with conv1d
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, self.embed_dim),
        )

        if use_confounds:
            self.proj_confound = nn.Sequential(
                                    nn.BatchNorm1d(1),
                                    nn.Linear(1,embed_dim))
        self.use_confounds = use_confounds
        
        self.pos_dropout = nn.Dropout(p=dropout)
        self.pre_norm = norm_layer(self.embed_dim)

        ### encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(hidden_dim=int(embed_dim * 2 ** i_layer),
                            depth= depths[i_layer],
                            num_heads=num_heads[i_layer],
                            window_size=self.window_sizes[i_layer],
                            window_size_factor=self.window_size_factor, 
                            mlp_ratio =mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop = dropout, attn_drop=attention_dropout, 
                            drop_path=dropout_path,
                            norm_layer=norm_layer,
                            downsample=PatchMerging if (i_layer < self.num_layers -1) else None,
                            use_global_attention = self.use_global_att[i_layer])
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        self.final_norm = norm_layer(embed_dim)

        ### decoder layers

        self.layers_up = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        for i_layer in range(self.num_layers):    #### TO DO 
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(int(embed_dim * 2 ** (self.num_layers-1-i_layer)))
            else:
                layer_up =BasicLayerUp(hidden_dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                            depth= depths[self.num_layers-1-i_layer],
                            num_heads=num_heads[self.num_layers-1-i_layer],
                            window_size=self.window_sizes[self.num_layers-1-i_layer],
                            window_size_factor=self.window_size_factor, 
                            mlp_ratio =mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop = dropout, attn_drop=attention_dropout, 
                            drop_path=dropout_path,
                            norm_layer=norm_layer,
                            upsample=PatchExpand if (i_layer < self.num_layers -1) else None,
                            use_global_attention = self.use_global_att[self.num_layers-1-i_layer])
            self.layers_up.append(layer_up)
            self.skip_proj.append(concat_linear)

        self.norm_up =norm_layer(self.embed_dim)

        ### HEAD
        self.up = nn.Linear(embed_dim, embed_dim*self.num_vertices)
        self.output = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

        ## normalise patches

    def forward_features(self,x,confounds=None):
       # input: B,C,L,V
        x = self.to_patch_embedding(x) # B, L, embed_dim=C
        b, n, _ = x.shape
        x = self.pre_norm(x)

        if self.use_pos_emb:
            x += self.absolute_pos_embed
        # deconfounding technique
        if self.use_confounds and (confounds is not None):
            confounds = self.proj_confound(confounds.view(-1,1))
            confounds = repeat(confounds, 'b d -> b n d', n=n)
            x += confounds
        x = self.pos_dropout(x)
        x_downsample = []

        #print('start encoder forward: {}'.format(x.shape))
        for i, layer in enumerate(self.layers):
            x_downsample.append(x)
            x = layer(x)
            #print('swin transformer output block {}: {}'.format(i+1, x.shape))

        x = self.norm(x) # B,L,C=int(embed_dim * 2 ** num_layer)

        return x,x_downsample

    def forward_up_features(self,x,x_downsample):
        #print('start decoder forward: {}'.format(x.shape))
        for ind, layer_up in enumerate(self.layers_up):    
            if ind == 0:
                x = layer_up(x)  
                #print('first patch merging: {}'.format(x.shape))          
            else:
                x = torch.cat([x, x_downsample[self.num_layers-1-ind]],dim=-1)
                #print('skip connection cat: {}'.format(x.shape))    
                x = self.skip_proj[ind](x) 
                #print('skip connection proj: {}'.format(x.shape))    
                x = layer_up(x)   
                #print('swin transformer block up: {}'.format(x.shape))    
        
        x = self.norm_up(x)
        return x 
    
    def repatch(self,x):
        
        B, L, V, C = x.shape
        
        batch_sphere =torch.zeros((B,40962,C),dtype=torch.float32).to(self.device)
        for i in range(self.num_patches):
            indices_to_extract =self.indices[str(i)].values
            batch_sphere[:,indices_to_extract,:] = x[:,i,:,:]
        return batch_sphere


    def proj_sphere(self,x):

        B, L, C = x.shape

        x = self.up(x)   # B, L, C*num_vertices
        x = rearrange(x, 'b l (c v) -> b l v c', v=self.num_vertices) # B, L, num_vertices, C
        x = self.repatch(x) # B, M, C
        x = self.final_norm(x)
        x = self.output(x) #B, M, num_class
        x = rearrange(x, 'b v c -> b c v') # B, num class, Vertices

        return x 

    def forward(self,x,confounds=None):
        # input: B,C,L,V
        x, x_downsample = self.forward_features(x,confounds) #B,int(embed_dim * 2 ** i_layer)
        x = self.forward_up_features(x,x_downsample)
        x = self.proj_sphere(x) #B, V, num_classes
        return x
    
    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class BasicLayerUp(nn.Module):

    """
    Basic Swin Transformer layer for one stage in the decoder
    """

    def __init__(self,hidden_dim, depth, num_heads,window_size,window_size_factor,
                    mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0, attn_drop=0,
                    drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_global_attention= False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.depth = depth
        #build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(hidden_dim=hidden_dim,
                                num_heads=num_heads, window_size=window_size,
                                window_size_factor=window_size_factor,
                                shift = 0 if (i%2 ==0) else 1,
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop,attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer, use_global_attention = use_global_attention)
            for i in range(depth)
        ])

        #merging layer
        if upsample is not None:
            self.upsample = upsample(hidden_dim=hidden_dim, norm_layer= norm_layer)
        else:
            self.upsample = None

    def forward(self,x):
        for block in self.blocks:
            x = block(x)
            #print('attention block up out: {}'.format(x.shape))
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class BasicLayer(nn.Module):

    """
    Basic Swin Transformer layer for one stage
    """

    def __init__(self,hidden_dim, depth, num_heads,window_size,window_size_factor, 
                    mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0, attn_drop=0,
                    drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_global_attention= False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.depth = depth
        #build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(hidden_dim=hidden_dim,
                                num_heads=num_heads, window_size=window_size,
                                window_size_factor = window_size_factor, 
                                shift = 0 if (i%2 ==0) else 1,
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop,attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer,use_global_attention = use_global_attention)
            for i in range(depth)
        ])

        #merging layer
        if downsample is not None:
            self.downsample = downsample(hidden_dim=hidden_dim, norm_layer= norm_layer)
        else:
            self.downsample = None

    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x



class SwinTransformerBlock(nn.Module):

    """
    Swin Transformer basic block
    """

    def __init__(self,hidden_dim, num_heads, window_size=80, window_size_factor=1, shift=0,
                    mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm,use_global_attention=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_size_factor = window_size_factor
        self.mlp_ratio = mlp_ratio
        self.shift = shift
        self.use_global_attention = use_global_attention


        self.norm1 = norm_layer(hidden_dim)
        #attention
        self.attention = WindowAttention(hidden_dim=hidden_dim,window_size=window_size,num_heads=num_heads,
                                        qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path >0 else nn.Identity()
        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim*mlp_ratio)
        self.mlp = MLP(in_features=hidden_dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)


    def forward(self,x):

        B,L,C = x.shape

        shortcut = x
        x = self.norm1(x)
        
        if self.shift and not self.use_global_attention:
            x_shifted = torch.roll(x,self.window_size//self.window_size_factor,dims=1)
            x_windows = window_partition(x_shifted,self.window_size)
        
        else:
            x_windows = window_partition(x,self.window_size)

        #attention

        attention_windows = self.attention(x_windows)

        if self.shift and not self.use_global_attention:
            attention_windows_reversed = window_reverse(attention_windows,self.window_size,L)
            x = torch.roll(attention_windows_reversed,-self.window_size//self.window_size_factor,dims=1)
        else:
            x = window_reverse(attention_windows,self.window_size,L)

        x = shortcut + self.drop_path(x)

        # FFN

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x 

     
class PatchExpand(nn.Module):  ### TO DO 

    def __init__(self, hidden_dim, norm_layer = nn.LayerNorm):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.unmerging = Rearrange('b l (h c) -> b (l h) c', h=4) ###### double check it is doing what i expect

        self.expand = nn.Linear(hidden_dim, 2*hidden_dim,bias=False)
        self.norm = norm_layer(hidden_dim//2)

    def forward(self,x):

        #print(x.shape)
        x = self.expand(x)
        #print(x.shape)

        x = self.unmerging(x)
        #print(x.shape)

        x = self.norm(x)

        return x 



     
class PatchMerging(nn.Module):

    def __init__(self, hidden_dim, norm_layer = nn.LayerNorm):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.merging = Rearrange('b (v h) n -> b v (n h)', h=4) ###### double check it is doing what i expect

        self.reduction = nn.Linear(4*hidden_dim, 2*hidden_dim,bias=False)
        self.norm = norm_layer(4*hidden_dim)

    def forward(self,x):

        B,L,C = x.shape
        #print(x.shape)
        #import pdb;pdb.set_trace()

        #x = x.view(B,-1, 4*C)
        x = self.merging(x)
        #import pdb;pdb.set_trace()
        x = self.norm(x)
        #import pdb;pdb.set_trace()
        x = self.reduction(x)

        return x 






class WindowAttention(nn.Module):

    """
    Args:
        x: input features with the sahpe of (num_windows*B, N, C)
    """


    def __init__(self,hidden_dim,window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim,hidden_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):

        Bw,L,C = x.shape

        qkv = self.qkv(x).reshape(Bw, L, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q *self.scale
        attention = (q @ k.transpose(-2,-1))

        attention = self.softmax(attention)

        x = (attention @ v).transpose(1,2).reshape(Bw,L,C)
        self.proj(x)
        self.proj_drop(x)

        return x



def window_partition(x,window_size):
    """
    Args:
        x: (B,L,C)
        window_size (int): window_size

    Returns:
        windows: (num_windows*B, window_size,C)
    
    """
    B,L,C = x.shape
    x = x.view(B,L//window_size, window_size,C)
    windows = x.permute(0,2,1,3).contiguous().view(-1,window_size,C)

    return windows


def window_reverse(windows,window_size,L):

    B = int(windows.shape[0] / (L//window_size))
    x = windows.view(B, L//window_size, window_size,-1)
    x = x.contiguous().view(B,L,-1)

    return x
    



class MLP(nn.Module):
    def __init__(self,in_features, hidden_features=None, out_features=None, act_layer = nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)

        return x

       

if __name__ == '__main__':

    model = SwinSiTUNet()

    import pdb;pdb.set_trace()

















