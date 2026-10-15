"""
light wrapper around DINOv2 ViT-S/14
"""

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import torch
import torch.nn as nn
from transformers import AutoModel


import torch
import torch.nn as nn
from transformers import AutoModel


class VisualEncoder(nn.Module):
    def __init__(self, d_model):
        """
        dinov2 parameters are frozen, but the projection's are not
        """
        super().__init__()

        self.dino = AutoModel.from_pretrained("facebook/dinov2-small")
        self.dino.requires_grad_(False)
        self.dino.eval()

        self.proj = nn.Linear(
            self.dino.config.hidden_size,  # 384
            d_model
        )
    
    def train(self, mode=True):
        """
        To ensure dino stays in eval even if the model recursively puts it in train mode 
        """
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, x):
        """
        x: (B, V (number of views), C, H, W)
        """

        B, V, C, H, W = x.shape

        x = x.reshape(B * V, C, H, W) # (B, V, C, H, W) -> (B*V, C, H, W)

        with torch.no_grad():
            outputs = self.dino(pixel_values=x)
            x = outputs.last_hidden_state[:, 1:, :] # (B*V, N_tokens, 384)

        x = self.proj(x) # projection 384 -> n_embd

        x = x.reshape(B, V, x.shape[1], x.shape[2]) # (B*V, N_tokens, n_embd) -> (B, V, N_tokens, n_embd)

        return x