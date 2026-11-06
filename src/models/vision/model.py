import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_view_encoder import MultiViewEncoder
from .object_decoder import ObjectDecoder, ObjectPredictionHead
from .config import (
    SceneModelConfig, MultiViewConfig, ObjectDecoderConfig, CameraDecoderConfig,
)
from .visual_encoder import VisualEncoder
from .camera_decoder import CameraDecoder, CameraPredictionHead

class SceneModel(nn.Module):

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        """Load a src.train checkpoint in eval mode, without optimizer state.

        Construction uses the usual DINOv2 pretrained backbone (cached locally
        or downloaded by Transformers), then restores all checkpoint weights.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        config = dict(checkpoint['config']['model'])
        for name, config_type in (
            ('multiview', MultiViewConfig),
            ('object_decoder', ObjectDecoderConfig),
            ('camera_decoder', CameraDecoderConfig),
        ):
            config[name] = config_type(**config[name])
        model = cls(SceneModelConfig(**config))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device).eval()

    def __init__(self, config):
        super().__init__()
        self.config = config

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

        return {
            "objects": out_obj,
            "cameras": out_cam,
        }
