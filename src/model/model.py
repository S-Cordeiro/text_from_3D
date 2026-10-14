import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_view_encoder import MultiViewEncoder
from .object_decoder import ObjectDecoder, ObjectPredictionHead
from .config import SceneModelConfig
from .visual_encoder import VisualEncoder
from .camera_decoder import CameraDecoder, CameraPredictionHead

class SceneModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.visual_encoder = VisualEncoder(config.d_model)
        self.multi_view_encoder = MultiViewEncoder(config)

        self.object_decoder = ObjectDecoder(config)
        self.object_prediction = ObjectPredictionHead(config)

        self.camera_decoder = CameraDecoder(config)
        self.camera_prediction = CameraPredictionHead(config)

    def forward(self, pixel_values):
        visual_tokens = self.visual_encoder(pixel_values)        
        mv_tokens = self.multi_view_encoder(visual_tokens)

        x_obj = self.object_decoder(mv_tokens)
        out_obj = self.object_prediction(x_obj)

        x_cam = self.camera_decoder(mv_tokens)
        out_cam = self.camera_prediction(x_cam)

        return out_obj | out_cam



