from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    def __post_init__(self):
        for name in ('block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.n_embd % self.n_head:
            raise ValueError('n_embd must be divisible by n_head')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
