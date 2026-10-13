import torch
import torch.nn as nn
from .blocks import SelfAttention, CrossAttention, MLP


class cameraDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.cdconfig = config.camera_decoder

        # self attention
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.self_attn = SelfAttention(config.d_model, self.cdconfig.n_head)

        # cross attention
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.cross_attn = CrossAttention(config.d_model, self.cdconfig.n_head)

        self.ln_3 = nn.LayerNorm(config.d_model)
        self.ffwd = MLP(config.d_model)

    def forward(self, queries, visual_tokens):
        x = self.ln_1(queries)
        attn = self.self_attn(x)
        queries = queries + attn

        x = self.ln_2(queries)
        attn = self.cross_attn(x, visual_tokens)
        queries = queries + attn

        queries = queries + self.ffwd(self.ln_3(queries))

        return queries


class cameraDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cdconfig = config.camera_decoder
        self.n_cameras = config.multiview.n_cameras

        self.camera_queries = nn.Embedding(self.n_cameras, config.d_model)
        self.blocks = nn.ModuleList([
            cameraDecoderBlock(config) for _ in range(self.cdconfig.n_layers)
        ])

    def forward(self, visual_tokens):
        B = visual_tokens.shape[0]

        queries = self.camera_queries.weight.unsqueeze(0)
        queries = queries.expand(B, -1, -1)

        for block in self.blocks:
            queries = block(queries, visual_tokens)

        return queries


class CameraPredictionHead(nn.Module):
    """Predict camera position (3D) and orientation (6D rotation)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cdconfig = config.camera_decoder

        self.position_head = nn.Linear(config.d_model, 3)
        self.pose_head = nn.Linear(config.d_model, 6)

    def forward(self, camera_queries):
        out = {
            "position": self.position_head(camera_queries),
            "pose": self.pose_head(camera_queries),
        }
        return out
