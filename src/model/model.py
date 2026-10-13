import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_view_encoder import MultiViewEncoder
from .object_decoder import ObjectDecoder, ObjectPredictionHead
from .config import SceneModelConfig
from .visual_encoder import VisualEncoder

class SceneModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.visual_encoder = VisualEncoder(config.d_model)
        self.multi_view_encoder = MultiViewEncoder(config)
        self.object_decoder = ObjectDecoder(config)
        self.object_prediction = ObjectPredictionHead(config)

    def forward(self, pixel_values):
        visual_tokens = self.visual_encoder(pixel_values)        
        x = self.multi_view_encoder(visual_tokens)
        x = self.object_decoder(x)
        out = self.object_prediction(x)

        return out



