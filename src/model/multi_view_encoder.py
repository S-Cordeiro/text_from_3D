import torch
import torch.nn as nn
from .blocks import SelfAttention, AttentionBlock

class MultiViewEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mvconfig = config.multiview

        self.camera_emb = nn.Embedding(self.mvconfig.n_cameras, config.d_model)
        self.positional_emb = nn.Embedding(config.d_model, config.d_model)
        self.blocks = nn.ModuleList([
            AttentionBlock(config.d_model, self.mvconfig.n_head) for _ in range(self.mvconfig.n_layers)
        ])

    def forward(self, x):
        B, V, T, C = x.shape # B = batch, V = num of views, T = length of sequence (16*16), C = d_model

        camera_pos = torch.arange(0, self.mvconfig.n_cameras, dtype=torch.long, device=x.device)
        cam_emb = self.camera_emb(camera_pos) # (n_cameras, d_model)
        x = x + cam_emb[None, :, None, :]

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.positional_emb(pos) # (T, d_model)
        x = x + pos_emb[None, None, :, :]

        x = x.reshape(B, V*T, C)

        for block in self.blocks:
            x = block(x)
        
        return x