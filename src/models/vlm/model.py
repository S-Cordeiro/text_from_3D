"""LLaVA-style fusion of scene object queries and GPT text embeddings."""
import torch
from torch import nn

from ..language import GPT
from ..vision.model import SceneModel


class LLaVAModel(nn.Module):
    def __init__(self, vision_model: SceneModel, language_model: GPT):
        super().__init__()
        self.vision_model = vision_model
        self.language_model = language_model
        self.projector = nn.Linear(vision_model.config.d_model, language_model.config.n_embd)
        if vision_model.config.object_decoder.n_objects >= language_model.config.block_size:
            raise ValueError('The GPT context must leave room for text after object queries')

    @classmethod
    def from_pretrained(cls, vision_checkpoint, language_checkpoint, device='cpu'):
        """Combine existing vision/GPT checkpoints with a new trainable projector."""
        model = cls(SceneModel.from_checkpoint(vision_checkpoint),
                    GPT.from_checkpoint(language_checkpoint))
        return model.to(device).eval()

    def encode_images(self, pixel_values):
        return self.projector(self.vision_model.encode_objects(pixel_values))

    def forward(self, pixel_values, idx, targets=None):
        """Return text logits/loss; targets use GPT's already-shifted convention."""
        return self.language_model(idx, targets, prefix_embeddings=self.encode_images(pixel_values))

    @torch.no_grad()
    def generate(self, pixel_values, idx, max_new_tokens, temperature=1.0, top_k=50):
        was_training = self.training
        self.eval()
        try:
            prefix = self.encode_images(pixel_values)
            return self.language_model.generate(
                idx, max_new_tokens, temperature, top_k, prefix_embeddings=prefix)
        finally:
            self.train(was_training)
