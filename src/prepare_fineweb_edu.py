"""Download a bounded FineWeb-Edu sample and write GPT-2 uint16 token shards.

Only one Parquet file and a bounded batch of documents are processed at a time.
Validation uses separate documents, before the training documents.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import HfApi, hf_hub_download


class ShardWriter:
    def __init__(self, root, split, target, shard_size):
        self.root, self.split, self.target = root, split, target
        self.buffer = np.empty(min(shard_size, target), dtype=np.uint16)
        self.position = self.total = self.index = 0
        self.shards = []

    def append(self, tokens):
        tokens = np.asarray(tokens[:self.target - self.total], dtype=np.uint16)
        offset = 0
        while offset < len(tokens):
            count = min(len(tokens) - offset, len(self.buffer) - self.position)
            self.buffer[self.position:self.position + count] = tokens[offset:offset + count]
            self.position += count
            self.total += count
            offset += count
            if self.position == len(self.buffer):
                self.flush()

    def flush(self):
        if not self.position:
            return
        name = f'{self.split}_{self.index:06d}.npy'
        temporary = self.root / (name + '.partial')
        with temporary.open('wb') as handle:
            np.save(handle, self.buffer[:self.position], allow_pickle=False)
        temporary.replace(self.root / name)
        self.shards.append({'file': name, 'tokens': self.position})
        print(f'{self.split}: {self.total:,}/{self.target:,} tokens | {name}', flush=True)
        self.position = 0
        self.index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/language'))
    parser.add_argument('--train-tokens', type=int, default=1_921_152_000)
    parser.add_argument('--val-tokens', type=int, default=10_000_000)
    parser.add_argument('--shard-size', type=int, default=100_000_000)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    if min(args.train_tokens, args.val_tokens, args.shard_size, args.threads) <= 0:
        parser.error('Token counts, shard size and thread count must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent runs or overwriting an existing dataset.
    lock = args.output_dir / 'preparation.lock'
    with lock.open('x') as handle:
        handle.write(f'{os.getpid()}\n')
    if any(args.output_dir.glob('*.npy')) or (args.output_dir / 'manifest.json').exists():
        parser.error('Output already contains a dataset; choose a new output directory')

    repo = 'HuggingFaceFW/fineweb-edu'
    api = HfApi()
    revision = api.dataset_info(repo, revision=args.revision).sha
    files = sorted(name for name in api.list_repo_files(repo, repo_type='dataset', revision=revision)
                   if name.startswith('sample/10BT/') and name.endswith('.parquet'))
    if not files:
        raise RuntimeError('No sample-10BT Parquet files found')
    enc = tiktoken.get_encoding('gpt2')
    assert enc.n_vocab <= np.iinfo(np.uint16).max + 1
    val = ShardWriter(args.output_dir, 'val', args.val_tokens, args.shard_size)
    train = ShardWriter(args.output_dir, 'train', args.train_tokens, args.shard_size)
    start = time.monotonic()
    manifest = dict(dataset=repo, subset='sample-10BT', revision=revision,
                    source_url=f'https://huggingface.co/datasets/{repo}', license='odc-by',
                    tokenizer='gpt2', dtype='uint16', eot_token=enc.eot_token,
                    train_target=args.train_tokens, val_target=args.val_tokens,
                    split_policy='First documents for validation; subsequent documents for training. '
                                 'Discard the remainder of the last validation document.',
                    files=[], documents=0, status='running')

    def save_manifest():
        manifest.update(train_tokens=train.total, val_tokens=val.total,
                        shards=val.shards + train.shards,
                        elapsed_seconds=round(time.monotonic() - start, 1))
        temp = args.output_dir / 'manifest.json.partial'
        temp.write_text(json.dumps(manifest, indent=2) + '\n')
        temp.replace(args.output_dir / 'manifest.json')

    save_manifest()
    for filename in files:
        print(f'Downloading {filename} at {revision}', flush=True)
        path = hf_hub_download(repo, filename, repo_type='dataset', revision=revision)
        manifest['files'].append(filename)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=['text']):
            token_lists = enc.encode_ordinary_batch(batch.column(0).to_pylist(), num_threads=args.threads)
            for tokens in token_lists:
                writer = val if val.total < val.target else train
                writer.append(tokens + [enc.eot_token])
                manifest['documents'] += 1
                if train.total == train.target:
                    break
            save_manifest()
            if train.total == train.target:
                break
        parquet.close()
        if train.total == train.target:
            break
    if train.total != train.target or val.total != val.target:
        raise RuntimeError('Source exhausted before reaching the requested token counts')
    val.flush()
    train.flush()
    # Read back every shard to verify counts, data type and token IDs.
    for shard in val.shards + train.shards:
        array = np.load(args.output_dir / shard['file'], mmap_mode='r', allow_pickle=False)
        if array.shape != (shard['tokens'],) or array.dtype != np.uint16 or array.max() >= enc.n_vocab:
            raise RuntimeError(f'Invalid shard: {shard}')
    manifest['status'] = 'complete'
    save_manifest()
    lock.unlink()
    print(f'COMPLETE: {train.total:,} training + {val.total:,} validation tokens', flush=True)


if __name__ == '__main__':
    main()
