"""Check an actual CUDA stop/resume against uninterrupted training before a long run."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='language-resume-') as directory:
        root = Path(directory)
        common = ['--data-root', str(args.data_root), '--batch-size', '1', '--block-size', '32',
                  '--total-batch-size', '32', '--n-layer', '1', '--n-head', '2', '--n-embd', '32',
                  '--max-steps', '3', '--warmup-steps', '0', '--eval-every', '3',
                  '--eval-steps', '1', '--save-every', '1', '--max-seconds', '120']
        subprocess.run([sys.executable, '-u', '-m', 'src.train_language',
                        '--out-dir', str(root / 'full'), *common], check=True)
        interrupt = '''import os, signal
import src.train_language as train
original = train.save_checkpoint
def save(*args, **kwargs):
    original(*args, **kwargs)
    if args[1]['step'] == 1:
        os.kill(os.getpid(), signal.SIGUSR1)
train.save_checkpoint = save
train.main()
'''
        subprocess.run([sys.executable, '-u', '-c', interrupt,
                        '--out-dir', str(root / 'split'), *common], check=True)
        stopped = torch.load(root / 'split/last.pt', map_location='cpu', weights_only=True)
        assert stopped['step'] == 1 and stopped['optimizer']['state']
        assert stopped['train_loader']['current_position'] == 32
        subprocess.run([sys.executable, '-u', '-m', 'src.train_language',
                        '--resume', str(root / 'split/last.pt'), '--max-seconds', '120'], check=True)
        full = torch.load(root / 'full/last.pt', map_location='cpu', weights_only=True)
        resumed = torch.load(root / 'split/last.pt', map_location='cpu', weights_only=True)
        assert full['step'] == resumed['step'] == 3
        assert full['train_loader'] == resumed['train_loader']
        assert full['scaler'] == resumed['scaler']
        for name, value in full['model'].items():
            torch.testing.assert_close(value, resumed['model'][name], rtol=0, atol=0)
        for param, state in full['optimizer']['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, resumed['optimizer']['state'][param][name], rtol=0, atol=0)
        print('CUDA resume verified: signal stop, optimizer, data position and weights match.', flush=True)


if __name__ == '__main__':
    main()
