"""GPT decoder, following karpathy/build-nanogpt and reusing our shared blocks."""
import inspect

import torch
from torch import nn
from torch.nn import functional as F

from ..blocks import AttentionBlock, SwiGLU
from .config import GPTConfig


class Block(AttentionBlock):
    def __init__(self, config):
        super().__init__(config.n_embd, config.n_head, config.dropout,
                         is_causal=True, feedforward=SwiGLU(config.n_embd))
        self.attn.c_proj.NANOGPT_SCALE_INIT = 1
        self.mlp.c_proj.NANOGPT_SCALE_INIT = 1


class GPT(nn.Module):
    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        """Load configuration and weights from a training checkpoint, in eval mode.

        Optimizer, RNG and data-loader state are not restored; use --resume
        in train_language.py to continue an interrupted training run.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(GPTConfig(**checkpoint['config']))
        model.load_state_dict(checkpoint['model'])
        return model.to(device).eval()

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, *, prefix_embeddings=None):
        # targets are already shifted by the data loader: y[t] = x[t + 1].
        B, T = idx.shape
        prefix_length = 0
        x = self.transformer.wte(idx)
        if prefix_embeddings is not None:
            if (prefix_embeddings.ndim != 3 or prefix_embeddings.shape[0] != B
                    or prefix_embeddings.shape[2] != self.config.n_embd):
                raise ValueError('prefix_embeddings must have shape (batch, prefix_length, n_embd)')
            prefix_length = prefix_embeddings.shape[1]
            x = torch.cat((prefix_embeddings.to(dtype=x.dtype), x), dim=1)
        if not 0 < T or T + prefix_length > self.config.block_size:
            raise ValueError(f'Sequence length must be in [1, {self.config.block_size}]')
        pos = torch.arange(T + prefix_length, device=idx.device)
        x = x + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x[:, prefix_length:]))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {'params': [p for p in params if p.dim() >= 2], 'weight_decay': weight_decay},
            {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0},
        ]
        kwargs = {}
        if 'fused' in inspect.signature(torch.optim.AdamW).parameters:
            kwargs['fused'] = device_type == 'cuda'
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, **kwargs)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50, *, prefix_embeddings=None):
        if temperature <= 0 or max_new_tokens < 0 or (top_k is not None and top_k <= 0):
            raise ValueError('Invalid sampling parameters')
        prefix_length = 0 if prefix_embeddings is None else prefix_embeddings.shape[1]
        text_context = self.config.block_size - prefix_length
        if text_context <= 0 or idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError('Generation requires at least one text token and room after the prefix')
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -text_context:], prefix_embeddings=prefix_embeddings)
                logits = logits[:, -1, :].float() / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], float('-inf'))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                idx = torch.cat((idx, next_token), dim=1)
        finally:
            self.train(was_training)
        return idx
