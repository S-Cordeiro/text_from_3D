"""Train with periodic validation: python -m src.train --eval-every 100."""

import argparse
import math
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from src.dataset.scene_dataset import SceneDataset, collate_scenes
from src.losses.scene_loss import SceneLoss
from src.model.config import SceneModelConfig
from src.model.model import SceneModel


def predict(model, images, device):
    """Run the model in bf16 and keep geometry losses in float32."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        predictions = model(images.to(device))
    return {
        group: {name: value.float() for name, value in values.items()}
        for group, values in predictions.items()
    }


def format_metrics(metrics):
    return (
        f"loss={metrics['total']:.4f}\n"
        f"  objects  | class={metrics['object_class']:.4f}"
        f" | position={metrics['object_position']:.4f}"
        f" | scale={metrics['object_scale']:.4f}"
        f" | rotation={metrics['object_rotation']:.4f}"
        f" | rotation error={math.degrees(metrics['object_rotation_angle']):.2f}°"
        f" | correct matches={100 * metrics['matching_accuracy']:.1f}%\n"
        f"  cameras | position={metrics['camera_position']:.4f}"
        f" | rotation={metrics['camera_rotation']:.4f}"
        f" | rotation error={math.degrees(metrics['camera_rotation_angle']):.2f}°"
    )


@torch.no_grad()
def evaluate(model, scene_loss, loader, device, by_count=False):
    was_training = model.training
    model.eval()
    totals = {}
    weights = {}
    grouped_totals = {}
    grouped_weights = {}
    scene_counts = {}
    try:
        for batch in loader:
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
                    'object_position', 'object_scale', 'object_rotation',
                    'object_rotation_angle', 'matching_accuracy',
                ) else scene_count
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
    if not by_count:
        return overall
    groups = {
        count: {
            'scene_count': scene_counts[count],
            **{name: value / max(grouped_weights[count][name], 1)
               for name, value in group.items()},
        }
        for count, group in sorted(grouped_totals.items())
    }
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


def save_checkpoint(path, model, optimizer, epoch, step, best_loss, config, metrics):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--no-checkpoints', action='store_true',
                        help='Keep logs and metrics without saving model or optimizer states')
    parser.add_argument('--extra-val-dataset', help='Additional dataset root containing a val split')
    args = parser.parse_args()
    if min(args.epochs, args.eval_every, args.log_every) < 1:
        parser.error('epochs, eval-every and log-every must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('lr must be finite and positive')


    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')

    def transform(image):
        return processor(
            images=image, do_rescale=False, do_resize=False,
            do_center_crop=False, return_tensors='pt',
        )['pixel_values'][0]

    train_dataset = SceneDataset(args.dataset, split='train', transform=transform)
    val_dataset = SceneDataset(args.dataset, split='val', transform=transform)

    if not len(train_dataset) or not len(val_dataset):
        raise ValueError('The train and val splits must both contain at least one scene')

    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_scenes, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_scenes)
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
    run_dir = create_run_directory(args.runs_dir)
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(run_dir / 'train.log', encoding='utf-8'))
    model_config = SceneModelConfig()
    run_config = {
        'arguments': vars(args),
        'model': asdict(model_config),
        'dataset_path': str(Path(args.dataset).resolve()),
        'train_scenes': len(train_dataset),
        'validation_scenes': len(val_dataset),
        'validation_sets': {
            name: {'path': str(loader.dataset.directory.resolve()),
                   'scenes': len(loader.dataset)}
            for name, loader in validation_loaders.items()
        },
        'checkpoint_selection_split': 'val',
        'processor': 'facebook/dinov2-small',
        'optimizer': {'name': 'AdamW', 'lr': args.lr},
        'batch_size': 1,
        'device': str(device),
        'compile': True,
        'mixed_precision': 'bf16',
        'started_at': datetime.now(timezone.utc).isoformat(),
    }
    logger.info('Run: %s | Device: %s', run_dir, device)

    model = SceneModel(model_config).to(device)
    # Compile in place so checkpoints retain the original state_dict keys.
    model.compile()
    scene_loss = SceneLoss().to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    run_config['loss'] = {
        'weights': scene_loss.weights,
        'no_object_weight': scene_loss.no_object_weight,
        'class_cost': scene_loss.matcher.class_cost,
        'position_cost': scene_loss.matcher.position_cost,
    }
    (run_dir / 'config.json').write_text(json.dumps(run_config, indent=2), encoding='utf-8')

    def record_metrics(split, epoch, step, metrics):
        with (run_dir / 'metrics.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps({
                'split': split, 'epoch': epoch, 'step': step, **metrics,
            }) + '\n')

    best_loss = math.inf
    validation_metrics = None
    step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            pred = predict(model, batch['images'], device)
            losses = scene_loss(pred, batch['targets'])
            losses['total'].backward()
            optimizer.step()
            step += 1

            metrics = {name: value.detach().item() for name, value in losses.items()}
            record_metrics('train', epoch, step, metrics)
            if step % args.log_every == 0:
                logger.info(
                    f'[TRAIN] epoch {epoch}/{args.epochs}'
                    f' | batch {batch_idx}/{len(train_loader)} | step {step}'
                    f" | {format_metrics(metrics)}",
                )
            is_final_step = epoch == args.epochs and batch_idx == len(train_loader)
            if step % args.eval_every == 0 or is_final_step:
                for split, loader in validation_loaders.items():
                    overall, groups = evaluate(model, scene_loss, loader, device, by_count=True)
                    record_metrics(split, epoch, step, overall)
                    logger.info(f'[{split}] step {step} | {format_metrics(overall)}')
                    for count, group_metrics in groups.items():
                        group_split = f'{split}_{count}_cubes'
                        record_metrics(group_split, epoch, step, group_metrics)
                        logger.info(
                            f'[{group_split}] step {step}'
                            f" | scenes={group_metrics['scene_count']}"
                            f' | {format_metrics(group_metrics)}'
                        )
                    if split == 'val':
                        validation_metrics = overall
                if validation_metrics['total'] < best_loss:
                    best_loss = validation_metrics['total']
                    if not args.no_checkpoints:
                        save_checkpoint(run_dir / 'best.pt', model, optimizer, epoch, step,
                                        best_loss, run_config, validation_metrics)
                if not args.no_checkpoints:
                    save_checkpoint(run_dir / 'last.pt', model, optimizer, epoch, step,
                                    best_loss, run_config, validation_metrics)

        # Also preserve the latest training state at every epoch boundary.
        if not args.no_checkpoints:
            save_checkpoint(run_dir / 'last.pt', model, optimizer, epoch, step,
                            best_loss, run_config, validation_metrics)
    logger.info('Training complete. Results saved in %s | Checkpoints: %s',
                run_dir, 'disabled' if args.no_checkpoints else 'enabled')


if __name__ == '__main__':
    main()
