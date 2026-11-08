"""Train with periodic validation: python -m src.train --eval-every 100."""

import argparse
import math
import json
import logging
import signal
import time
import random
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor

from src.dataset.scene_dataset import SceneDataset, collate_scenes
from src.losses.scene_loss import SceneLoss
from src.models.vision.config import SceneModelConfig
from src.vision_audit import audit_dataset
from src.models.vision.model import SceneModel, without_scale_head


def predict(model, images, device):
    """Run the model in bf16 and keep geometry losses in float32."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        predictions = model(images.to(device))
    return {
        group: {name: value.float() for name, value in values.items()}
        for group, values in predictions.items()
    }


OBJECT_TERMS = ('object_class', 'object_position', 'object_rotation')


def objects_total(metrics, weights):
    """Weighted object part of the loss: the checkpoint selection criterion.

    The full total is dominated by camera position, which says little about
    the objects a VLM is asked about.
    """
    return sum(weights[name] * metrics[name] for name in OBJECT_TERMS)


def format_metrics(metrics):
    return (
        f"loss={metrics['total']:.4f}\n"
        f"  objects  | class={metrics['object_class']:.4f}"
        f" | position={metrics['object_position']:.4f}"
        f" | object rotation={metrics['object_rotation']:.4f}"
        f" | correct matches={100 * metrics['matching_accuracy']:.1f}%\n"
        f"  cameras | position={metrics['camera_position']:.4f}"
        f" | rotation={metrics['camera_rotation']:.4f}"
        f" | rotation error={math.degrees(metrics['camera_rotation_angle']):.2f}°"
    )


@torch.no_grad()
def evaluate(model, scene_loss, loader, device, by_count=False, should_stop=None):
    was_training = model.training
    model.eval()
    totals = {}
    weights = {}
    grouped_totals = {}
    grouped_weights = {}
    scene_counts = {}
    try:
        for batch in loader:
            if should_stop and should_stop():
                return None
            if by_count and len(batch['targets']) != 1:
                raise ValueError('Grouped evaluation requires batch_size=1')
            metrics = scene_loss(predict(model, batch['images'], device), batch['targets'])
            scene_count = len(batch['targets'])
            object_count = sum(t['objects']['classes'].numel() for t in batch['targets'])
            if by_count:
                scene_counts[object_count] = scene_counts.get(object_count, 0) + scene_count
                grouped_totals.setdefault(object_count, {})
                grouped_weights.setdefault(object_count, {})
            for name, value in metrics.items():
                # Object geometry and accuracy are averaged over objects;
                # other losses are averaged over scenes.
                weight = object_count if name in (
                    'object_position', 'object_rotation',
                    'object_rotation_angle', 'matching_accuracy',
                ) else scene_count
                if name == 'object_rotation_angle':
                    weight = sum(int((t['objects']['classes'] != 1).sum()) for t in batch['targets'])
                totals[name] = totals.get(name, 0.0) + value.item() * weight
                weights[name] = weights.get(name, 0) + weight
                if by_count:
                    group = grouped_totals[object_count]
                    group_weights = grouped_weights[object_count]
                    group[name] = group.get(name, 0.0) + value.item() * weight
                    group_weights[name] = group_weights.get(name, 0) + weight
    finally:
        model.train(was_training)
    overall = {name: value / max(weights[name], 1) for name, value in totals.items()}
    overall['objects_total'] = objects_total(overall, scene_loss.weights)
    if not by_count:
        return overall
    groups = {}
    for count, group in sorted(grouped_totals.items()):
        groups[count] = {
            'scene_count': scene_counts[count],
            **{name: value / max(grouped_weights[count][name], 1)
               for name, value in group.items()},
        }
        groups[count]['objects_total'] = objects_total(groups[count], scene_loss.weights)
    return overall, groups


def create_run_directory(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    number = max(
        (int(path.name[4:]) for path in root.iterdir()
         if path.name.startswith('run_') and path.name[4:].isdigit()),
        default=0,
    ) + 1
    while True:
        directory = root / f'run_{number}'
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            number += 1


def save_checkpoint(path, model, optimizer, epoch, step, best_loss, config, metrics, scheduler, progress=None):
    checkpoint = {
        'progress': progress,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'best_val_loss': best_loss,
        'config': config,
        'validation_metrics': metrics,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    # parsing
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='dataset_single_cube')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max-steps', type=int,
                        help='Total optimizer steps; overrides --epochs (data-scaling runs share it)')
    parser.add_argument('--train-scenes', type=int,
                        help='Train on the first N scenes of the train split (nested subsets)')
    parser.add_argument('--select-on', choices=('objects', 'total'), default='objects',
                        help='Validation loss used to keep best.pt')
    parser.add_argument('--n-classes', type=int, default=None,
                        help='Number of object classes, excluding the no-object class')
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--no-checkpoints', action='store_true',
                        help='Keep logs and metrics without saving model or optimizer states')
    parser.add_argument('--extra-val-dataset', help='Additional dataset root containing a val split')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--mv-layers', type=int, default=4)
    parser.add_argument('--dino-unfreeze-blocks', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=3e-6)
    parser.add_argument('--dino-lr', type=float, default=1e-6)
    parser.add_argument('--warmup-steps', type=int, default=750)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--micro-batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--save-seconds', type=float, default=900)
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--dataset-audit', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    stop_reason = None
    def request_stop(signum, frame):
        nonlocal stop_reason
        stop_reason = f'signal {signum}'
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    def should_stop():
        nonlocal stop_reason
        if args.max_seconds and time.monotonic() - started >= args.max_seconds:
            stop_reason = stop_reason or 'session time budget reached'
        return stop_reason is not None
    if args.resume and args.init_checkpoint:
        parser.error('Use resume OR init-checkpoint')
    if args.no_checkpoints and (args.resume or args.max_seconds):
        parser.error('Timed/resumable runs require checkpoints')
    if min(args.batch_size, args.micro_batch_size, args.mv_layers, args.d_model) < 1:
        parser.error('Batch sizes and model dimensions must be positive')
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error('max-seconds must be positive')
    if args.save_seconds <= 0 or args.warmup_steps < 0:
        parser.error('Invalid checkpoint interval or warmup')
    if args.batch_size % args.micro_batch_size:
        parser.error('batch-size must be divisible by micro-batch-size')
    if not 0 < args.min_lr <= args.lr or args.dino_lr <= 0:
        parser.error('Invalid learning rates')
    if args.dataset_audit:
        audit = json.loads(args.dataset_audit.read_text())
        if Path(audit['root']).resolve() != Path(args.dataset).resolve():
            parser.error('Dataset audit belongs to another dataset')
    else:
        audit = audit_dataset(args.dataset)
    if args.n_classes is None:
        args.n_classes = audit['n_classes']
    if args.n_classes < audit['n_classes']:
        parser.error('n-classes conflicts with actual dataset labels (including no-object class)')
    if audit['max_objects'] > SceneModelConfig().object_decoder.n_objects:
        parser.error('Dataset contains more objects than object queries')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision('high')
    if min(args.epochs, args.eval_every, args.log_every, args.max_steps or 1, args.train_scenes or 1) < 1:
        parser.error('epochs, max-steps, train-scenes, eval-every and log-every must be positive')
    if args.n_classes < 1:
        parser.error('n-classes must be positive')


    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')

    def transform(image):
        return processor(
            images=image, do_rescale=False, do_resize=False,
            do_center_crop=False, return_tensors='pt',
        )['pixel_values'][0]

    full_train_dataset = SceneDataset(args.dataset, split='train', transform=transform)
    train_dataset = full_train_dataset
    if args.train_scenes is not None:
        if args.train_scenes > len(full_train_dataset):
            parser.error(f'train-scenes exceeds the {len(full_train_dataset)} available scenes')
        # Scene paths are sorted, so smaller subsets are nested in larger ones.
        train_dataset = Subset(full_train_dataset, range(args.train_scenes))
    val_dataset = SceneDataset(args.dataset, split='val', transform=transform)

    if not len(train_dataset) or not len(val_dataset):
        raise ValueError('The train and val splits must both contain at least one scene')

    train_loader = DataLoader(
        train_dataset, batch_size=args.micro_batch_size, shuffle=True,
        generator=torch.Generator(),
        collate_fn=collate_scenes, num_workers=4,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_scenes, generator=torch.Generator().manual_seed(args.seed))
    validation_loaders = {'val': val_loader}
    if args.extra_val_dataset:
        extra_val = SceneDataset(args.extra_val_dataset, split='val', transform=transform)
        if not len(extra_val):
            raise ValueError('The additional val split must contain at least one scene')
        validation_loaders['val_2_3'] = DataLoader(
            extra_val, batch_size=1, collate_fn=collate_scenes,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This CUDA device does not support bf16 mixed precision')
    run_dir = args.run_dir or create_run_directory(args.runs_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'config.json').exists() and not args.resume:
        parser.error('Run directory already used; resume or choose a new directory')
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(run_dir / 'train.log', encoding='utf-8'))
    model_config = SceneModelConfig(n_classes=args.n_classes, d_model=args.d_model,
                                    dino_unfreeze_blocks=args.dino_unfreeze_blocks)
    model_config.multiview.n_layers = args.mv_layers
    for split in audit['splits'].values():
        if {int(v) for v in split['view_counts']} != {model_config.multiview.n_cameras}:
            parser.error('Actual view counts do not match model configuration')
    lr_max, lr_min = args.lr, args.min_lr
    accumulation = args.batch_size // args.micro_batch_size
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    if args.max_steps is not None:
        args.epochs = math.ceil(args.max_steps / steps_per_epoch)
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_steps, max(0, total_steps - 1))
    run_config = {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'model': asdict(model_config),
        'dataset_path': str(Path(args.dataset).resolve()),
        'train_scenes': len(train_dataset),
        'train_scenes_available': len(full_train_dataset),
        'last_train_scene': full_train_dataset.scene_paths[len(train_dataset) - 1].name,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'validation_scenes': len(val_dataset),
        'validation_sets': {
            name: {'path': str(loader.dataset.directory.resolve()),
                   'scenes': len(loader.dataset)}
            for name, loader in validation_loaders.items()
        },
        'checkpoint_selection_split': 'val',
        'checkpoint_selection_metric': 'objects_total' if args.select_on == 'objects' else 'total',
        'processor': 'facebook/dinov2-small',
        'optimizer': {'name': 'AdamW', 'lr': lr_max},
        'scheduler': {
            'name': 'SequentialLR (LinearLR + CosineAnnealingLR)',
            'warmup_steps': warmup_steps,
            'total_steps': total_steps,
            'lr_max': lr_max,
            'lr_min': lr_min,
            'warmup_start_lr': lr_max / max(1, warmup_steps),
        },
        'batch_size': args.batch_size,
        'micro_batch_size': args.micro_batch_size,
        'dataset_audit': audit,
        'device': str(device),
        'compile': not args.no_compile,
        'mixed_precision': 'bf16',
        'started_at': datetime.now(timezone.utc).isoformat(),
    }
    logger.info('Run: %s | Device: %s', run_dir, device)

    model = SceneModel(model_config).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        initial_config = dict(initial['config']['model'])
        initial_config.pop('dino_unfreeze_blocks', None)
        target_config = asdict(model_config)
        target_config.pop('dino_unfreeze_blocks')
        if initial_config != target_config:
            raise ValueError('Initialization checkpoint architecture does not match')
        model.load_state_dict(without_scale_head(initial['model_state_dict']))
        del initial
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable > 15_000_000:
        raise ValueError(f'Trainable parameter budget exceeded: {trainable}')
    run_config['parameters'] = {'trainable': trainable,
        'total': sum(p.numel() for p in model.parameters()),
        'dino_trainable': sum(p.numel() for p in model.visual_encoder.dino.parameters() if p.requires_grad)}
    run_config['environment'] = dict(torch=str(torch.__version__), python=platform.python_version(),
        gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
        git_revision=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip())
    logger.info('Parameters: %s', run_config['parameters'])
    # Compile in place so checkpoints retain the original state_dict keys.
    if not args.no_compile:
        model.compile()
    scene_loss = SceneLoss().to(device)
    dino_params = [p for p in model.visual_encoder.dino.parameters() if p.requires_grad]
    dino_ids = {id(p) for p in dino_params}
    groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in dino_ids],
               'lr': lr_max}]
    if dino_params:
        groups.append({'params': dino_params, 'lr': args.dino_lr})
    optimizer = AdamW(groups, weight_decay=0.01)
    run_config['optimizer'].update(weight_decay=0.01, dino_lr=args.dino_lr if dino_params else None)
    # One multiplicative schedule preserves the lower DINO LR throughout training.
    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        fraction = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
        return lr_min / lr_max + (1 - lr_min / lr_max) * (1 + math.cos(math.pi * fraction)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    run_config['scheduler']['name'] = 'LambdaLR (linear warmup + cosine, proportional group LRs)'

    run_config['loss'] = {
        'weights': scene_loss.weights,
        'no_object_weight': scene_loss.no_object_weight,
        'class_cost': scene_loss.matcher.class_cost,
        'position_cost': scene_loss.matcher.position_cost,
    }
    config_path = run_dir / 'config.json'
    if not args.resume:
        config_path.write_text(json.dumps(run_config, indent=2), encoding='utf-8')

    def record_metrics(split, epoch, step, metrics):
        with (run_dir / 'metrics.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps({
                'split': split, 'epoch': epoch, 'step': step, **metrics,
            }) + '\n')

    selection_key = 'objects_total' if args.select_on == 'objects' else 'total'
    best_loss = math.inf
    validation_metrics = None
    step, next_epoch, next_batch = 0, 1, 0
    elapsed_before = 0.0
    peak_before = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        old_config = checkpoint['config']
        for key in ('model', 'batch_size', 'micro_batch_size', 'dataset_audit'):
            if old_config[key] != run_config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        for key in ('epochs', 'max_steps', 'train_scenes', 'select_on', 'seed', 'lr', 'min_lr',
                    'dino_lr', 'warmup_steps', 'eval_every'):
            if old_config['arguments'].get(key) != run_config['arguments'].get(key):
                raise ValueError(f'Resume argument mismatch: {key}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        step, best_loss = checkpoint['step'], checkpoint['best_val_loss']
        validation_metrics = checkpoint['validation_metrics']
        progress = checkpoint['progress']
        next_epoch, next_batch = progress['next_epoch'], progress['next_batch']
        elapsed_before = progress['elapsed_seconds']
        peak_before = progress.get('peak_cuda_bytes', 0)
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        run_config = old_config
        del checkpoint
        logger.info('Resuming optimizer step %s, epoch %s, microbatch %s', step, next_epoch, next_batch)

    last_saved = time.monotonic()
    def save(path):
        nonlocal last_saved
        progress = dict(next_epoch=next_epoch, next_batch=next_batch,
                        elapsed_seconds=elapsed_before + time.monotonic() - started,
                        peak_cuda_bytes=max(peak_before, torch.cuda.max_memory_allocated()) if device.type == 'cuda' else 0)
        save_checkpoint(path, model, optimizer, next_epoch, step, best_loss,
                        run_config, validation_metrics, scheduler, progress)
        last_saved = time.monotonic()

    model.train()
    for epoch in range(next_epoch, args.epochs + 1):
        # A run resumed after its last step only needs its final validation.
        if step >= total_steps:
            break
        train_loader.generator.manual_seed(args.seed + epoch)
        resume_batch = next_batch if epoch == next_epoch else 0
        metrics_accum, accumulated_scenes = {}, 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < resume_batch:
                continue
            # Stop only between optimizer updates, so accumulation never gets lost.
            if accumulated_scenes == 0 and should_stop():
                break
            cycle_start = (batch_idx // accumulation) * accumulation
            cycle_scenes = min(args.batch_size, len(train_dataset) - cycle_start * args.micro_batch_size)
            batch_scenes = len(batch['targets'])
            pred = predict(model, batch['images'], device)
            losses = scene_loss(pred, batch['targets'])
            (losses['total'] * batch_scenes / cycle_scenes).backward()
            accumulated_scenes += batch_scenes
            for name, value in losses.items():
                metrics_accum[name] = metrics_accum.get(name, 0.0) + value.detach().item() * batch_scenes
            if (batch_idx + 1) % accumulation and batch_idx + 1 < len(train_loader):
                continue
            learning_rate = optimizer.param_groups[0]['lr']
            dino_learning_rate = optimizer.param_groups[-1]['lr'] if dino_params else 0
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = {name: value / accumulated_scenes for name, value in metrics_accum.items()}
            metrics.update(lr=learning_rate, dino_lr=dino_learning_rate,
                elapsed_seconds=elapsed_before + time.monotonic() - started,
                peak_cuda_bytes=max(peak_before, torch.cuda.max_memory_allocated()) if device.type == 'cuda' else 0,
                scenes_seen=(epoch - 1) * len(train_dataset) + min((batch_idx + 1) * args.micro_batch_size, len(train_dataset)))
            metrics_accum, accumulated_scenes = {}, 0
            next_epoch, next_batch = epoch, batch_idx + 1
            if next_batch == len(train_loader):
                next_epoch, next_batch = epoch + 1, 0
            record_metrics('train', epoch, step, metrics)
            if step % args.log_every == 0:
                logger.info('[TRAIN] epoch %s | step %s | %s', epoch, step, format_metrics(metrics))
            is_final_step = step == total_steps
            if not should_stop() and (step % args.eval_every == 0 or is_final_step):
                complete_validation = True
                for split, loader in validation_loaders.items():
                    evaluated = evaluate(model, scene_loss, loader, device, by_count=True, should_stop=should_stop)
                    if evaluated is None:
                        complete_validation = False
                        break
                    overall, grouped = evaluated
                    record_metrics(split, epoch, step, overall)
                    logger.info('[%s] step %s | %s', split, step, format_metrics(overall))
                    for count, group_metrics in grouped.items():
                        record_metrics(f'{split}_{count}_objects', epoch, step, group_metrics)
                    if split == 'val':
                        validation_metrics = overall
                if complete_validation and validation_metrics[selection_key] < best_loss:
                    best_loss = validation_metrics[selection_key]
                    if not args.no_checkpoints:
                        save(run_dir / 'best.pt')
                if not args.no_checkpoints:
                    save(run_dir / 'last.pt')
            if not args.no_checkpoints and time.monotonic() - last_saved >= args.save_seconds:
                save(run_dir / 'last.pt')
            if step == total_steps:
                break
        if not args.no_checkpoints:
            save(run_dir / 'last.pt')
        if should_stop() or step == total_steps:
            break
    # A stop during final validation leaves the training cursor complete. Re-run
    # validation next session before declaring the experiment complete.
    final_metrics = None
    if step == total_steps and not should_stop():
        final_metrics = evaluate(model, scene_loss, val_loader, device, by_count=True, should_stop=should_stop)
        if final_metrics is not None:
            validation_metrics, grouped = final_metrics
            record_metrics('final_val', args.epochs, step, validation_metrics)
            if validation_metrics[selection_key] < best_loss:
                best_loss = validation_metrics[selection_key]
                if not args.no_checkpoints:
                    save(run_dir / 'best.pt')
            if not args.no_checkpoints:
                save(run_dir / 'last.pt')
    status = dict(completed=step == total_steps and final_metrics is not None, step=step,
                  total_steps=total_steps, best_val_loss=best_loss, selection_metric=selection_key,
                  stop_reason=stop_reason,
                  elapsed_seconds=elapsed_before + time.monotonic() - started,
                  peak_cuda_bytes=max(peak_before, torch.cuda.max_memory_allocated()) if device.type == 'cuda' else 0)
    (run_dir / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    logger.info('Training status: %s', status)


if __name__ == '__main__':
    main()
