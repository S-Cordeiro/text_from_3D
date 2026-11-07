"""Export best-checkpoint predictions and targets per validation scene for later analysis."""
import argparse
import gzip
import json
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from src.dataset.scene_dataset import SceneDataset, collate_scenes
from src.models.vision.model import SceneModel
from src.losses.scene_loss import SceneLoss
from src.train import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    interrupted = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop)
    config = json.loads((args.run / 'config.json').read_text())
    processor = AutoImageProcessor.from_pretrained(config['processor'])
    def transform(image):
        return processor(images=image, do_rescale=False, do_resize=False,
                         do_center_crop=False, return_tensors='pt')['pixel_values'][0]
    dataset = SceneDataset(config['dataset_path'], split='val', transform=transform)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_scenes, num_workers=2)
    device = torch.device('cuda')
    model = SceneModel.from_checkpoint(args.run / 'best.pt', device=device)
    loss = SceneLoss(**{k: config['loss'][k] for k in ('weights', 'no_object_weight', 'class_cost', 'position_cost')})
    output = args.run / 'best_validation_predictions.jsonl.gz'
    temporary = output.with_suffix('.tmp')
    with torch.no_grad(), gzip.open(temporary, 'wt') as file:
        for index, batch in enumerate(loader):
            if interrupted or time.monotonic() - start >= args.max_seconds:
                return 75
            predictions = predict(model, batch['images'], device)
            metrics = loss(predictions, batch['targets'])
            target = batch['targets'][0]
            classes = target['objects']['classes'].to(device)
            positions = target['objects']['positions'].to(device)
            matches = loss.matcher.forward(predictions['objects']['class_logits'],
                predictions['objects']['position'], [classes], [positions])[0]
            file.write(json.dumps(dict(scene=dataset.scene_paths[index].name,
                metrics={k: v.item() for k, v in metrics.items()},
                predictions={g: {k: v[0].tolist() for k, v in values.items()} for g, values in predictions.items()},
                targets={g: {k: v.tolist() for k, v in values.items()} for g, values in target.items()},
                matched_query_indices=matches[0].tolist(), matched_target_indices=matches[1].tolist())) + '\n')
    temporary.replace(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
