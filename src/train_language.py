"""Single-CUDA-GPU GPT training, inspired by karpathy/build-nanogpt."""
import argparse
import math
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.models.language import GPT, GPTConfig
from src.models.language.data import DataLoaderLite
from src.language_schedule import night_seconds


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    ratio = min(1.0, (step - warmup_steps) / max(1, max_steps - 1 - warmup_steps))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def save_checkpoint(out_dir, payload, keep=3):
    """Publish atomically; last.pt and the numbered checkpoint share disk storage."""
    path = out_dir / f"model_{payload['step']:06d}.pt"
    temporary = out_dir / 'checkpoint.tmp'
    torch.save(payload, temporary)
    temporary.replace(path)
    last_tmp = out_dir / 'last.pt.tmp'
    last_tmp.unlink(missing_ok=True)
    os.link(path, last_tmp)
    last_tmp.replace(out_dir / 'last.pt')
    for old in sorted(out_dir.glob('model_*.pt'))[:-keep]:
        old.unlink()
    print(f'checkpoint saved: {path} (resume via {out_dir / "last.pt"})', flush=True)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--resume', type=Path, help='Restore model, optimizer, RNG and data position')
    parser.add_argument('--max-seconds', type=float, help='Stop and checkpoint after this session budget')
    parser.add_argument('--night-only', action='store_true', help='Only run 18:00-08:00 Europe/Paris')
    parser.add_argument('--save-seconds', type=float, default=900, help='Checkpoint at least this often')
    parser.add_argument('--keep-checkpoints', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--total-batch-size', type=int, default=32768, help='Tokens per optimizer step')
    parser.add_argument('--n-layer', type=int, default=8)
    parser.add_argument('--n-head', type=int, default=8)
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
    if args.night_only:
        try:
            remaining = night_seconds() - 300  # Reserve five minutes for the final checkpoint.
        except ValueError as exc:
            parser.error(str(exc))
        if remaining <= 0:
            parser.error('Too little time remains before 08:00 to start training safely')
        args.max_seconds = min(args.max_seconds or remaining, remaining)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if not all(k in checkpoint for k in ('train_loader', 'rng_cpu', 'rng_cuda')):
            parser.error('This checkpoint lacks data/RNG state required for a faithful resume')
        # Preserve the original training schedule across separately timed sessions.
        for key in ('batch_size', 'block_size', 'total_batch_size', 'n_layer', 'n_head', 'n_embd',
                    'max_steps', 'warmup_steps', 'max_lr', 'min_lr', 'weight_decay', 'seed', 'compile'):
            setattr(args, key, checkpoint['args'][key])
    args.data_root = args.data_root or Path(checkpoint['args']['data_root'] if checkpoint else 'data/language')
    args.out_dir = args.out_dir or (args.resume.parent if args.resume else Path('runs/language'))
    if (args.max_seconds is not None and args.max_seconds <= 0) or args.save_seconds <= 0 or args.keep_checkpoints <= 0:
        parser.error('Time budgets and checkpoint retention must be positive')
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
    config = GPTConfig(**checkpoint['config']) if checkpoint else GPTConfig(
        block_size=T, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    train_loader = DataLoaderLite(B, T, args.data_root, 'train', config.vocab_size)
    val_loader = DataLoaderLite(B, T, args.data_root, 'val', config.vocab_size)
    raw_model = GPT(config).to(device)
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.max_lr, 'cuda')
    model = torch.compile(raw_model) if args.compile else raw_model
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint and ((args.out_dir / 'last.pt').exists() or any(args.out_dir.glob('model_*.pt'))):
        parser.error('Output already contains checkpoints; use --resume or a new --out-dir')
    print(f'{sum(p.numel() for p in raw_model.parameters()):,} parameters | {grad_accum_steps} accumulation steps | {dtype}')
    start_step = checkpoint['step'] if checkpoint else 0
    val_loss = checkpoint['val_loss'] if checkpoint else None
    if checkpoint:
        raw_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        train_loader.load_state_dict(checkpoint['train_loader'])
        torch.set_rng_state(checkpoint['rng_cpu'])
        torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        print(f'Resumed step {start_step}; data position: shard {train_loader.current_shard}, '
              f'token {train_loader.current_position}', flush=True)
        del checkpoint
    stop_reason = None

    def request_stop(signum, frame):
        nonlocal stop_reason
        stop_reason = f'signal {signal.Signals(signum).name}'

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, request_stop)

    last_saved = time.monotonic()

    def save(completed_steps):
        nonlocal last_saved
        save_checkpoint(args.out_dir, dict(
            model=raw_model.state_dict(), config=asdict(config),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            step=completed_steps, val_loss=val_loss, tokenizer='gpt2',
            train_loader=train_loader.state_dict(), rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all(),
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}),
            args.keep_checkpoints)
        last_saved = time.monotonic()

    completed_steps = start_step
    save(completed_steps)
    for step in range(start_step, args.max_steps):
        if args.max_seconds is not None and time.monotonic() - started >= args.max_seconds:
            stop_reason = 'session time budget reached'
        if stop_reason:
            break
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
        if args.max_seconds is not None and time.monotonic() - started >= args.max_seconds:
            stop_reason = 'session time budget reached'
        if (completed_steps % args.eval_every == 0 or last_step) and not stop_reason:
            model.eval()
            val_loader.reset()
            val_loss = torch.zeros((), device=device)
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    if stop_reason or (args.max_seconds is not None and time.monotonic() - started >= args.max_seconds):
                        stop_reason = stop_reason or 'session time budget reached'
                        break
                    x, y = val_loader.next_batch()
                    with torch.autocast(device_type='cuda', dtype=dtype):
                        _, loss = model(x.to(device), y.to(device))
                    val_loss += loss.detach() / args.eval_steps
            val_loss = None if stop_reason else val_loss.item()
            print(f'validation loss: {val_loss}')
            with (args.out_dir / 'log.txt').open('a') as f:
                f.write(f'step {completed_steps} | val_loss {val_loss}\n')
        if completed_steps % args.save_every == 0 or last_step or time.monotonic() - last_saved >= args.save_seconds:
            save(completed_steps)
        if stop_reason:
            break
    save(completed_steps)
    print(f'Stopped at step {completed_steps}: {stop_reason or "training complete"}', flush=True)


if __name__ == '__main__':
    main()
