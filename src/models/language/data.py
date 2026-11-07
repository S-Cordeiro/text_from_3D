from pathlib import Path

import numpy as np
import torch


class DataLoaderLite:
    """Sequential .npy token shards; keep only the current batch in RAM."""
    def __init__(self, B, T, data_root, split, vocab_size=50257):
        if B <= 0 or T <= 0:
            raise ValueError('B and T must be positive')
        self.B, self.T = B, T
        self.shards = sorted(Path(data_root).glob(f'{split}_*.npy'))
        if not self.shards:
            raise ValueError(f'No {split}_*.npy shards in {data_root}')
        self.vocab_size = vocab_size
        self.reset()

    def _load_shard(self):
        self.tokens = np.load(self.shards[self.current_shard], mmap_mode='r', allow_pickle=False)
        if self.tokens.ndim != 1 or self.tokens.dtype.kind not in 'iu':
            raise ValueError('Token shards must be one-dimensional integer arrays')
        if len(self.tokens) < self.B * self.T + 1:
            raise ValueError(f'{self.shards[self.current_shard]} needs at least B*T+1 tokens')
        if self.tokens.min() < 0 or self.tokens.max() >= self.vocab_size:
            raise ValueError('Token IDs outside the model vocabulary')

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self._load_shard()

    def next_batch(self):
        B, T = self.B, self.T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_position = 0
            self._load_shard()
        buf = torch.tensor(self.tokens[self.current_position:self.current_position + B*T+1].astype(np.int64))
        self.current_position += B * T
        return buf[:-1].view(B, T), buf[1:].view(B, T)

    def state_dict(self):
        return dict(batch_size=self.B, block_size=self.T, vocab_size=self.vocab_size,
                    shards=[(p.name, p.stat().st_size) for p in self.shards],
                    current_shard=self.current_shard, current_position=self.current_position)

    def load_state_dict(self, state):
        current = self.state_dict()
        for key in ('batch_size', 'block_size', 'vocab_size', 'shards'):
            if state[key] != current[key]:
                raise ValueError(f'Cannot resume: data loader {key} changed')
        shard, position = state['current_shard'], state['current_position']
        if not 0 <= shard < len(self.shards) or position < 0 or position % (self.B * self.T):
            raise ValueError('Invalid data loader position in checkpoint')
        self.current_shard = shard
        self._load_shard()
        if position > len(self.tokens):
            raise ValueError('Checkpoint position exceeds shard length')
        self.current_position = position
