# 文件名: SSFTT_Net.py 2022
# (此版本已修复 num_tokens Bug)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.init as init

# --- Transformer Helper Classes (as per Author's code) ---

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module): # Renamed from PreNorm
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module): # Renamed from FeedForward
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """ Author's Attention Implementation """
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        inner_dim = dim # In author's code, inner_dim is just dim
        self.heads = heads
        self.scale = dim ** -0.5 # Author uses dim, not dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = True)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """ Author's Transformer Implementation """
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

# --- Main Model (SSFTT) ---
class SSFTT(nn.Module):
    
    def __init__(self, bands, num_classes, patch_size=13, dim=64, depth=1, heads=4, mlp_dim=8, num_tokens=4, dropout=0.1, emb_dropout=0.1, conv3d_out_channels=8):
        super(SSFTT, self).__init__()
        
        self.L = num_tokens 
        self.cT = dim

        # 1. Spectral-Spatial Feature Extraction
        # Expects input (B, 1, D, H, W) -> (B, 1, 30, 13, 13)
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=conv3d_out_channels, kernel_size=(3, 3, 3)), # (B, 8, 28, 11, 11)
            nn.BatchNorm3d(conv3d_out_channels),
            nn.ReLU(),
        )
        
        # Calculate in_channels for Conv2D based on Conv3D output
        conv3d_out_bands = bands - 3 + 1 # (e.g., 30 - 3 + 1 = 28)
        
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=conv3d_out_channels * conv3d_out_bands, out_channels=dim, kernel_size=(3, 3)), # (B, 64, 9, 9)
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        # 2. Author's Tokenizer
        # Two learnable weight matrices
        self.token_wA = nn.Parameter(torch.empty(1, self.L, dim), # <-- 2. 维度现在正确 (1, 4, 64)
                                     requires_grad=True)
        init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, dim, self.cT),
                                     requires_grad=True)  # (1, 64, 64)
        init.xavier_normal_(self.token_wV)

        # 3. Transformer Input
        self.pos_embedding = nn.Parameter(torch.empty(1, (self.L + 1), dim)) # <-- 3. 维度现在正确 (1, 5, 64)
        init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 4. Transformer Encoder
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        # 5. Classification Head
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        # Author uses a simple Linear layer, let's match that
        self.nn1 = nn.Linear(dim, num_classes)
        init.xavier_uniform_(self.nn1.weight)
        init.normal_(self.nn1.bias, std=1e-6)


    def forward(self, x, mask=None):
        # Input x shape: (B, 1, D, H, W) -> e.g., (64, 1, 30, 13, 13)
        
        # 3D Conv
        x = self.conv3d_features(x) 
        # (B, 8, D-2, H-2, W-2) -> (64, 8, 28, 11, 11)
        
        # Rearrange for 2D Conv
        # Author's: 'b c h w y -> b (c h) w y' (h=D, w=H, y=W)
        x = rearrange(x, 'b c d h w -> b (c d) h w') 
        # (B, 8*28, 11, 11) -> (64, 224, 11, 11)
        
        # 2D Conv
        x = self.conv2d_features(x)
        # (B, 64, H-4, W-4) -> (64, 64, 9, 9)
        
        # Reshape for Tokenizer
        # Author's: 'b c h w -> b (h w) c'
        x = rearrange(x, 'b c h w -> b (h w) c')
        # (B, 81, 64)

        # --- Author's Tokenizer Logic ---
        # (B, 81, 64) @ (1, 64, 4) -> (B, 81, 4)
        wa = rearrange(self.token_wA, 'b h w -> b w h') # (1, 64, 4)
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h') # (B, 4, 81)
        A = A.softmax(dim=-1) # (B, 4, 81)

        # (B, 81, 64) @ (1, 64, 64) -> (B, 81, 64)
        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        
        # (B, 4, 81) @ (B, 81, 64) -> (B, 4, 64)
        T = torch.einsum('bij,bjk->bik', A, VV)
        # --- End Tokenizer ---

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # (B, 1, 64)
        x = torch.cat((cls_tokens, T), dim=1) # (B, 5, 64)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x, mask) # (B, 5, 64)

        x = self.to_cls_token(x[:, 0]) # (B, 64)
        
        # Use author's final layer
        x = self.nn1(x) # (B, num_classes)

        return x
    
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


a = torch.randn(2, 1, 30, 13, 13)
model = SSFTT(30, 10)
out = model(a)
print(out.shape)