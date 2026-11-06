"""CPU checks without downloading a vision backbone: python -m scripts.check_vlm."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from src.models.language import GPT, GPTConfig
from src.models.vlm import LLaVAModel


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=6, object_decoder=SimpleNamespace(n_objects=2))
        self.encoder = nn.Linear(3, 6)
        self.calls = 0

    def encode_objects(self, pixels):
        self.calls += 1
        return self.encoder(pixels)


def main():
    torch.manual_seed(7)
    gpt = GPT(GPTConfig(block_size=8, vocab_size=17, n_layer=1, n_head=2, n_embd=8))
    vision = TinyVision()
    model = LLaVAModel(vision, gpt)
    pixels = torch.randn(2, 2, 3)
    idx = torch.randint(17, (2, 4))
    targets = torch.randint(17, (2, 4))
    logits, loss = model(pixels, idx, targets)
    assert logits.shape == (2, 4, 17)
    torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 17), targets.reshape(-1)))
    loss.backward()
    for parameter in (model.projector.weight, vision.encoder.weight, gpt.transformer.wte.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0

    # Earlier text positions cannot see future text tokens.
    changed = idx.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 17
    torch.testing.assert_close(model(pixels, idx)[0][:, :-1], model(pixels, changed)[0][:, :-1])
    assert not torch.allclose(model(pixels, idx)[0], model(pixels + 1, idx)[0])
    # An empty prefix must preserve the text-only GPT behavior.
    torch.testing.assert_close(gpt(idx)[0], gpt(idx, prefix_embeddings=torch.empty(2, 0, 8))[0])
    try:
        model(pixels, torch.zeros(2, 7, dtype=torch.long))
    except ValueError:
        pass
    else:
        raise AssertionError('Context overflow was accepted')
    vision.calls = 0
    output = model.generate(pixels, idx, max_new_tokens=7, top_k=1)
    assert output.shape == (2, 11) and vision.calls == 1
    assert model.training and gpt.training
    torch.testing.assert_close(output[:, :4], idx)
    print('VLM checks passed: gradients, causal loss, visual conditioning, context and generation')


if __name__ == '__main__':
    main()
