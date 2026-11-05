"""Single-CUDA-GPU GPT training, inspired by karpathy/build-nanogpt."""
import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.models.language import GPT, GPTConfig
from src.models.language.data import DataLoaderLite


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    ratio = min(1.0, (step - warmup_steps) / max(1, max_steps - 1 - warmup_steps))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/language'))
    parser.add_argument('--out-dir', type=Path, default=Path('runs/language'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--total-batch-size', type=int, default=32768, help='Tokens per optimizer step')
    parser.add_argument('--n-layer', type=int, default=12)
    parser.add_argument('--n-head', type=int, default=12)
    parser.add_argument('--n-embd', type=int, default=768)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--max-lr', type=float, default=6e-4)
    parser.add_argument('--min-lr', type=float, default=6e-5)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--eval-steps', type=int, default=20)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    for name in ('batch_size', 'block_size', 'total_batch_size', 'max_steps', 'eval_every', 'eval_steps', 'save_every'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if not 0 <= args.warmup_steps < args.max_steps:
        parser.error('Require 0 <= warmup-steps < max-steps')
    if not 0 <= args.min_lr <= args.max_lr or args.max_lr <= 0 or args.weight_decay < 0:
        parser.error('Invalid learning rates or weight decay')
    B, T = args.batch_size, args.block_size
    if args.total_batch_size % (B * T):
        parser.error('total-batch-size must be divisible by batch-size * block-size')
    if not torch.cuda.is_available():
        raise RuntimeError('This training script requires one CUDA GPU')
    device = 'cuda:0'
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    grad_accum_steps = args.total_batch_size // (B * T)
    config = GPTConfig(block_size=T, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    train_loader = DataLoaderLite(B, T, args.data_root, 'train', config.vocab_size)
    val_loader = DataLoaderLite(B, T, args.data_root, 'val', config.vocab_size)
    raw_model = GPT(config).to(device)
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.max_lr, 'cuda')
    model = torch.compile(raw_model) if args.compile else raw_model
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'{sum(p.numel() for p in raw_model.parameters()):,} parameters | {grad_accum_steps} accumulation steps | {dtype}')
    val_loss = None
    for step in range(args.max_steps):
        model.train()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            with torch.autocast(device_type='cuda', dtype=dtype):
                _, loss = model(x.to(device), y.to(device))
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        lr = get_lr(step, args.max_lr, args.min_lr, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        completed_steps = step + 1
        line = (f'step {completed_steps:5d} | loss {loss_accum.item():.6f} | lr {lr:.4e} '
                f'| norm {norm.item():.4f} | tok/sec {args.total_batch_size / dt:.0f}')
        print(line)
        with (args.out_dir / 'log.txt').open('a') as f:
            f.write(line + '\n')
        last_step = completed_steps == args.max_steps
        if completed_steps % args.eval_every == 0 or last_step:
            model.eval()
            val_loader.reset()
            val_loss = torch.zeros((), device=device)
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    x, y = val_loader.next_batch()
                    with torch.autocast(device_type='cuda', dtype=dtype):
                        _, loss = model(x.to(device), y.to(device))
                    val_loss += loss.detach() / args.eval_steps
            val_loss = val_loss.item()
            print(f'validation loss: {val_loss:.6f}')
            with (args.out_dir / 'log.txt').open('a') as f:
                f.write(f'step {completed_steps} | val_loss {val_loss:.6f}\n')
        if completed_steps % args.save_every == 0 or last_step:
            torch.save(dict(model=raw_model.state_dict(), config=asdict(config),
                            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            step=completed_steps, val_loss=val_loss,
                            tokenizer='gpt2', args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}),
                       args.out_dir / f'model_{completed_steps:06d}.pt')


if __name__ == '__main__':
    main()
