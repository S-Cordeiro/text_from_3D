"""Offline regression checks: class audit, selective gradients and exact CPU resume."""
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.models.vision.visual_encoder import VisualEncoder
from src.vision_audit import audit_dataset
import src.train as training


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=384)
        self.input = nn.Linear(3, 384)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(384, 384) for _ in range(2)])
        self.layernorm = nn.LayerNorm(384)

    def forward(self, pixel_values):
        x = self.input(pixel_values.mean((-1, -2)))[:, None, :]
        for block in self.encoder.layer:
            x = torch.tanh(block(x))
        return SimpleNamespace(last_hidden_state=self.layernorm(x).expand(-1, 2, -1))


class FakeProcessor:
    def __call__(self, images, **kwargs):
        return {'pixel_values': images[None]}


def main():
    torch.set_num_threads(1)
    with patch('torch.cuda.is_available', return_value=False), patch('src.models.vision.visual_encoder.AutoModel.from_pretrained', side_effect=lambda *a, **k: FakeDino()):
        for blocks in (0, 1, 2):
            encoder = VisualEncoder(16, blocks).train()
            encoder(torch.rand(1, 8, 3, 14, 14)).sum().backward()
            assert not encoder.dino.training
            assert all(p.grad is None for p in encoder.dino.input.parameters())
            for i, layer in enumerate(encoder.dino.encoder.layer):
                assert all((p.grad is not None) == (blocks > 0 and i >= 2 - blocks) for p in layer.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'data'
            identity = torch.eye(4).tolist()
            for split, count in [('train', 5), ('val', 2)]:
                for i in range(count):
                    folder = dataset / split / f'scene_{i:04d}'
                    folder.mkdir(parents=True)
                    Image.new('RGB', (14, 14), (i * 25, 100, 50)).save(folder / 'view.png')
                    truth = dict(objects=[dict(class_id=2, type='cylinder', position=[0, 0, 1],
                        rotation_matrix=torch.eye(3).tolist())],
                        cameras=[dict(image='view.png', camera_to_reference=identity) for _ in range(8)])
                    (folder / 'ground_truth.json').write_text(json.dumps(truth))
            audit = audit_dataset(dataset)
            assert audit['n_classes'] == 3 and audit['max_objects'] == 1
            original_step = training.AdamW.step
            def loader(*args, **kwargs):
                kwargs['num_workers'] = 0
                return DataLoader(*args, **kwargs)
            def run(name, resume=False, interrupt=False, extra=()):
                argv = ['train', '--dataset', str(dataset), '--run-dir', str(root / name),
                    '--epochs', '2', '--d-model', '16', '--mv-layers', '1', '--batch-size', '2',
                    '--micro-batch-size', '1', '--warmup-steps', '1', '--eval-every', '2', '--no-compile']
                argv += list(extra)
                if resume:
                    argv += ['--resume', str(root / name / 'last.pt')]
                triggered = False
                def step(opt, *args, **kwargs):
                    nonlocal triggered
                    result = original_step(opt, *args, **kwargs)
                    if interrupt and not triggered:
                        triggered = True
                        signal.raise_signal(signal.SIGUSR1)
                    return result
                with patch.object(sys, 'argv', argv), patch.object(training, 'DataLoader', side_effect=loader), \
                     patch.object(training.AutoImageProcessor, 'from_pretrained', return_value=FakeProcessor()), \
                     patch.object(training.AdamW, 'step', step):
                    training.main()
            run('full')
            run('resumed', interrupt=True)
            stopped = json.loads((root / 'resumed/status.json').read_text())
            assert not stopped['completed'] and stopped['step'] == 1
            run('resumed', resume=True)
            full = torch.load(root / 'full/last.pt', weights_only=True)
            resumed = torch.load(root / 'resumed/last.pt', weights_only=True)
            assert full['step'] == resumed['step'] == 6
            for key in full['model_state_dict']:
                torch.testing.assert_close(full['model_state_dict'][key], resumed['model_state_dict'][key], rtol=0, atol=0)
            assert json.loads((root / 'resumed/status.json').read_text())['completed']
            # Data-scaling options: a nested subset and a step budget ending mid-epoch,
            # interrupted then resumed; best.pt follows the object loss.
            subset = ['--train-scenes', '3', '--max-steps', '3']
            run('subset', interrupt=True, extra=subset)
            run('subset', resume=True, extra=subset)
            status = json.loads((root / 'subset/status.json').read_text())
            config = json.loads((root / 'subset/config.json').read_text())
            assert status['completed'] and status['step'] == 3 and status['selection_metric'] == 'objects_total'
            assert config['train_scenes'] == 3 and config['steps_per_epoch'] == 2 and config['total_steps'] == 3
            assert config['last_train_scene'] == 'scene_0002'
            records = [json.loads(l) for l in (root / 'subset/metrics.jsonl').read_text().splitlines()]
            assert max(r['step'] for r in records) == 3
            assert all('objects_total' in r for r in records if r['split'] in ('val', 'final_val'))
            best = torch.load(root / 'subset/best.pt', weights_only=True)
            assert best['best_val_loss'] == min(r['objects_total'] for r in records if r['split'] in ('val', 'final_val'))
            # The real audit must reject two declared classes for label 2.
            with patch.object(sys, 'argv', ['train', '--dataset', str(dataset), '--n-classes', '2']):
                try:
                    training.main()
                except SystemExit as exc:
                    assert exc.code == 2
                else:
                    raise AssertionError('Conflicting classes accepted')
    print('PASS: label collision rejected; gradients selective; interrupted/resumed weights bit-identical; '
          'train-scenes/max-steps/object selection OK.')


if __name__ == '__main__':
    main()
