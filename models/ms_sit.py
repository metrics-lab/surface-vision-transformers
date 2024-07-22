import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange

from einops import repeat


'''
This implementation of SwinTransformer is greatly inspired by the official codebase: https://github.com/microsoft/Swin-Transformer
and adapted for surface analysis on icospheric meshes. 
'''

#OK

#TODO - DROPATH RATE TO IMPLEMENT
class MSSiT(nn.Module):
    """
    Args:
        window_size (int): Number of patches to apply local-attention to.
    """
    

    def __init__(self,ico_init_resolution=4,num_channels=4,num_classes=1,
                    embed_dim=96, depths=[2,2,6,2],num_heads=[3,6,12,24],
                    window_size=80,mlp_ratio=4,qkv_bias=True,qk_scale=True,
                    dropout=0, attention_dropout=0,drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, use_pos_emb=False,patch_norm=True,
                    use_confounds =False,**kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.use_pos_emb = use_pos_emb
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers-1))   #number of features after the last layer
        self.mlp_ratio = mlp_ratio
        self.num_channels = num_channels

        if isinstance(window_size,int):
            self.window_sizes = [window_size for i in range(self.num_layers)]
        elif isinstance(window_size,list):
            self.window_sizes = window_size

        print('window size: {}'.format(self.window_sizes))

        if ico_init_resolution==4:
            self.num_patches = 5120
            self.num_vertices = 15
            patch_dim = self.num_vertices * self.num_channels
        elif ico_init_resolution==5:
            self.num_patches = 20480
            self.num_vertices = 6
            patch_dim = self.num_vertices * self.num_channels
        elif ico_init_resolution==2:
            self.num_patches = 320
            self.num_vertices = 153
            patch_dim = self.num_vertices * self.num_channels
        
        # absolute position embedding
        if use_pos_emb:
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

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(hidden_dim=int(embed_dim * 2 ** i_layer),
                            input_resolution=ico_init_resolution,
                            depth= depths[i_layer],
                            num_heads=num_heads[i_layer],
                            window_size=self.window_sizes[i_layer],
                            mlp_ratio =mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop = dropout, attn_drop=attention_dropout, 
                            drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            norm_layer=norm_layer,
                            downsample=PatchMerging if (i_layer < self.num_layers -1) else None,
                            )
            self.layers.append(layer)
        
        self.pre_norm = norm_layer(self.embed_dim)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        ## normalise patches
    
    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


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

        att_list_encoder=  []

        #import pdb;pdb.set_trace()

        for i, layer in enumerate(self.layers):
            #if i==0:
            #    x, att = layer(x,return_attention=True)
            #else:
            #    x = layer(x,return_attention=False)
            x, att = layer(x,return_attention=True)
            att_list_encoder.append(att)
        

        x = self.norm(x) # B,L,C=int(embed_dim * 2 ** (num_layer-1))
        x = self.avgpool(x.transpose(1,2)) # B,C,1
        x = torch.flatten(x,1) # B,C
        return x, att_list_encoder

    def forward(self,x,confounds=None):
        # input: B,C,L,V
        x, att = self.forward_features(x,confounds) #B,int(embed_dim * 2 ** i_layer)
        x = self.head(x) #B, num_classes
        #return x, att
        return x 


#OK
class BasicLayer(nn.Module):

    """
    Basic Swin Transformer layer for one stage
    """

    def __init__(self,hidden_dim, input_resolution, depth, num_heads,window_size,
                    mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0, attn_drop=0,
                    drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution
        self.depth = depth
        #build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(hidden_dim=hidden_dim, input_resolution=input_resolution,
                                num_heads=num_heads, window_size=window_size,
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop,attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer)
            for i in range(depth)
        ])

        #merging layer
        if downsample is not None:
            self.downsample = downsample(hidden_dim=hidden_dim, norm_layer= norm_layer)
        else:
            self.downsample = None

    def forward(self,x, return_attention=False):
        att_list = []
        for i, block in enumerate(self.blocks):
            if return_attention:
                x, att = block(x,return_attention)
                att_list.append(att)
            else:
                x = block(x,return_attention=False)
            
        if self.downsample is not None:
            x = self.downsample(x)

        if return_attention:
            return x, att_list
        else:
            return x



class MLP(nn.Module):
    '''
    similar as https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    '''
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

class SwinTransformerBlock(nn.Module):

    """
    Swin Transformer basic block
    """

    def __init__(self,hidden_dim, input_resolution, num_heads, window_size=80, shift_size=0,
                    mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio


        self.norm1 = norm_layer(hidden_dim)
        #attention
        self.attention = WindowAttention(hidden_dim=hidden_dim,window_size=window_size,num_heads=num_heads,
                                        qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path >0 else nn.Identity()
        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim*mlp_ratio)
        self.mlp = MLP(in_features=hidden_dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)


    def forward(self,x, return_attention=False):

        B,L,C = x.shape

        shortcut = x
        x = self.norm1(x)

        x_windows = window_partition(x,self.window_size)

        #attention

        attention_windows, attention_matrix = self.attention(x_windows)

        x = window_reverse(attention_windows,self.window_size,L)

        x = shortcut + self.drop_path(x)

        # FFN

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attention_matrix
        else:
            return x

#OK

#OK
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

#OK
def window_reverse(windows,window_size,L):

    B = int(windows.shape[0] / (L//window_size))
    x = windows.view(B, L//window_size, window_size,-1)
    x = x.contiguous().view(B,L,-1)

    return x

#OK

#OK need to add relative position bias [done]

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


        ## relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2*window_size-1, num_heads))
        coords = torch.arange(self.window_size)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += self.window_size - 1
        self.register_buffer("relative_position_index", relative_coords)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            

    def forward(self,x):

        Bw,L,C = x.shape
        #print('batch size window: {}'.format(x.shape))

        qkv = self.qkv(x).reshape(Bw, L, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q *self.scale
        attention = (q @ k.transpose(-2,-1))

        ## relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        attention = self.softmax(attention)
        #print(attention.shape)
        #print(v.shape)
        attention = self.attn_drop(attention)

        x = (attention @ v).transpose(1,2).reshape(Bw,L,C)
        
        #print((attention @ v).transpose(1,2).shape)
        #print(x.shape)
        #print('*******')
        self.proj(x)
        self.proj_drop(x)

        return x, attention
class PatchMerging(nn.Module):

    def __init__(self, hidden_dim, norm_layer = nn.LayerNorm):
        super().__init__()

        self.hidden_dim = hidden_dim

        #self.merging = Rearrange('b (v h) n -> b v (n h)', h=4) ###### double check it is doing what i expect
        self.merging = Rearrange('b (v h) n -> b v (h n)', h=4)

        self.reduction = nn.Linear(4*hidden_dim, 2*hidden_dim,bias=False)
        self.norm = norm_layer(4*hidden_dim)

    def forward(self,x):

        #print(x.shape)
        #import pdb;pdb.set_trace()

        #x = x.view(B,-1, 4*C)
        x = self.merging(x)
        #import pdb;pdb.set_trace()
        x = self.norm(x)
        #import pdb;pdb.set_trace()
        x = self.reduction(x)

        return x 

if __name__ == '__main__':

    model = MSSiT()














