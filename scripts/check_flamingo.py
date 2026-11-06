"""CPU architecture checks: python -m scripts.check_flamingo (no downloads)."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from src.models.language import GPT, GPTConfig
from src.models.vlm import FlamingoModel


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=6)
        self.encoder = nn.Linear(3, 6)
        self.calls = 0

    def encode_multiview(self, pixels):
        self.calls += 1
        return self.encoder(pixels)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    gpt = GPT(GPTConfig(block_size=8, vocab_size=17, n_layer=3, n_head=2, n_embd=8))
    vision = TinyVision()
    model = FlamingoModel(vision, gpt)
    pixels = torch.randn(2, 12, 3)  # More visual tokens than the text context permits.
    idx = torch.randint(17, (2, 8))
    targets = torch.randint(17, (2, 8))
    targets[:, :2] = -100
    assert len(model.cross_attention_blocks) == 2
    logits, loss = model(pixels, idx, targets)
    torch.testing.assert_close(logits, gpt(idx)[0], rtol=0, atol=0)
    torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 17), targets.reshape(-1)))
    loss.backward()
    for block in model.cross_attention_blocks:
        assert block.attn_gate.grad.abs() > 0 and block.ff_gate.grad.abs() > 0

    # Once gates open, images and cross-attention weights must receive gradients.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for block in model.cross_attention_blocks:
            block.attn_gate.fill_(0.5)
            block.ff_gate.fill_(0.5)
    model(pixels, idx, targets)[1].backward()
    for parameter in (vision.encoder.weight, gpt.transformer.wte.weight,
                      model.cross_attention_blocks[0].attn.k_proj_weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert not torch.allclose(model(pixels, idx)[0], model(pixels + 2, idx)[0])
    changed = idx.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 17
    torch.testing.assert_close(model(pixels, idx)[0][:, :-1], model(pixels, changed)[0][:, :-1])
    vision.calls = 0
    output = model.generate(pixels, idx, 3, top_k=1)
    assert output.shape == (2, 11) and vision.calls == 1
    assert model.training and gpt.training
    torch.testing.assert_close(output[:, :8], idx)
    model.eval()
    model.generate(pixels, idx, 1)
    assert not model.training and not gpt.training
    try:
        model(pixels, torch.zeros(2, 9, dtype=torch.long))
    except ValueError:
        pass
    else:
        raise AssertionError('Context overflow was accepted')
    print('Flamingo checks passed: zero gates, gradients, visual conditioning, causality and generation')


if __name__ == '__main__':
    main()
