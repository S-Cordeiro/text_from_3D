"""Flamingo-style gated visual cross-attention between frozen or trainable GPT blocks."""
import torch
from torch import nn
from torch.nn import functional as F

from ..blocks import SwiGLU
from ..language import GPT
from ..vision.model import SceneModel


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, text_dim, vision_dim, n_head, dropout=0.0):
        super().__init__()
        self.ln_attn = nn.LayerNorm(text_dim)
        self.attn = nn.MultiheadAttention(
            text_dim, n_head, kdim=vision_dim, vdim=vision_dim,
            dropout=dropout, batch_first=True)
        self.ln_ff = nn.LayerNorm(text_dim)
        self.ff = SwiGLU(text_dim)
        self.attn_gate = nn.Parameter(torch.zeros(()))
        self.ff_gate = nn.Parameter(torch.zeros(()))

    def forward(self, x, visual_tokens):
        attended, _ = self.attn(self.ln_attn(x), visual_tokens, visual_tokens,
                                need_weights=False)
        x = x + self.attn_gate.tanh() * attended
        return x + self.ff_gate.tanh() * self.ff(self.ln_ff(x))


class FlamingoModel(nn.Module):
    def __init__(self, vision_model: SceneModel, language_model: GPT):
        super().__init__()
        if language_model.config.n_layer < 2:
            raise ValueError('Flamingo requires at least two GPT blocks to insert cross-attention between them')
        self.vision_model = vision_model
        self.language_model = language_model
        config = language_model.config
        self.cross_attention_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(config.n_embd, vision_model.config.d_model,
                                     config.n_head, config.dropout)
            for _ in range(config.n_layer - 1)
        ])

    @classmethod
    def from_pretrained(cls, vision_checkpoint, language_checkpoint, device='cpu'):
        """Restore both backbones and initialize new cross-attention blocks."""
        model = cls(SceneModel.from_checkpoint(vision_checkpoint),
                    GPT.from_checkpoint(language_checkpoint))
        return model.to(device).eval()

    def encode_images(self, pixel_values):
        return self.vision_model.encode_multiview(pixel_values)

    def _forward_with_vision(self, idx, visual_tokens, targets=None):
        config = self.language_model.config
        if idx.ndim != 2 or not 0 < idx.shape[1] <= config.block_size:
            raise ValueError(f'idx must have shape (batch, length) with length in [1, {config.block_size}]')
        if (visual_tokens.ndim != 3 or visual_tokens.shape[0] != idx.shape[0]
                or visual_tokens.shape[1] == 0
                or visual_tokens.shape[2] != self.vision_model.config.d_model):
            raise ValueError('Visual tokens must have shape (batch, nonempty token sequence, vision dimension)')
        transformer = self.language_model.transformer
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = transformer.wte(idx) + transformer.wpe(positions)
        for i, block in enumerate(transformer.h):
            x = block(x)
            if i < len(self.cross_attention_blocks):
                x = self.cross_attention_blocks[i](x, visual_tokens)
        logits = self.language_model.lm_head(transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def forward(self, pixel_values, idx, targets=None):
        """Use GPT's already-shifted text targets, including optional -100 masks."""
        return self._forward_with_vision(idx, self.encode_images(pixel_values), targets)

    @torch.no_grad()
    def generate(self, pixel_values, idx, max_new_tokens, temperature=1.0, top_k=50):
        if temperature <= 0 or max_new_tokens < 0 or (top_k is not None and top_k <= 0):
            raise ValueError('Invalid sampling parameters')
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError('Generation requires at least one text token')
        was_training = self.training
        self.eval()
        try:
            visual_tokens = self.encode_images(pixel_values)
            for _ in range(max_new_tokens):
                context = idx[:, -self.language_model.config.block_size:]
                logits, _ = self._forward_with_vision(context, visual_tokens)
                logits = logits[:, -1].float() / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], float('-inf'))
                idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        finally:
            self.train(was_training)
        return idx
