"""Tokenize separate UTF-8 training and validation corpora into .npy files."""
import argparse
from pathlib import Path

import numpy as np

from src.models.language.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-text', type=Path, required=True)
    parser.add_argument('--val-text', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('data/language'))
    args = parser.parse_args()
    if args.train_text.resolve() == args.val_text.resolve():
        parser.error('Use separate training and validation corpora')
    tokenizer = Tokenizer()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, path in [('train', args.train_text), ('val', args.val_text)]:
        tokens = tokenizer.encode(path.read_text(encoding='utf-8'), add_eot=True)
        np.save(args.output_dir / f'{split}_000000.npy', np.asarray(tokens, dtype=np.uint16))
        print(f'{split}: {len(tokens):,} tokens')


if __name__ == '__main__':
    main()
