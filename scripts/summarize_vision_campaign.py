"""Collect a campaign's results in machine-readable tables (no GPU needed).

    python scripts/summarize_vision_campaign.py CAMPAIGN

Writes in CAMPAIGN:
- `summary.json` / `summary.csv`: one row per experiment: its settings, cost,
  validation losses at the selected checkpoint, and the interpretable metrics
  of `best_validation_metrics.json` (units and degrees).
- `curves.csv`: long table (one row per experiment, split and step) of every
  validation record, plus training losses averaged over each evaluation
  interval, so runs with different epoch lengths line up by step.
"""
import csv
import json
import math
from pathlib import Path
import sys

SETTINGS = ('d_model', 'mv_layers', 'unfreeze', 'train_scenes', 'max_steps', 'lr', 'dino_lr')
LOSSES = ('total', 'objects_total', 'object_class', 'object_position', 'object_rotation',
          'camera_position', 'camera_rotation', 'matching_accuracy')
# Flattened from best_validation_metrics.json: column name -> path in the JSON.
READABLE = {
    'object_position_error_mean': ('object_position_error', 'mean'),
    'object_position_error_median': ('object_position_error', 'median'),
    'cube_rotation_deg_mean': ('cube_rotation_error_deg', 'mean'),
    'cube_rotation_deg_median': ('cube_rotation_error_deg', 'median'),
    'cube_rotation_share_below_10deg': ('cube_rotation_error_deg', 'share_below_10'),
    'cylinder_axis_deg_mean': ('cylinder_axis_error_deg', 'mean'),
    'cylinder_axis_deg_median': ('cylinder_axis_error_deg', 'median'),
    'class_accuracy': ('class_accuracy',),
    'count_accuracy': ('count_accuracy',),
    'false_positives': ('false_positives',),
    'missed_objects': ('missed_objects',),
    'camera_position_error_mean': ('camera_position_error', 'mean'),
    'camera_rotation_deg_mean': ('camera_rotation_error_deg', 'mean'),
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def lookup(data, path):
    for key in path:
        if data is None:
            return None
        data = data.get(key)
    return data


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def curve_rows(name, records, eval_every):
    """Validation records as they are, training records averaged per evaluation interval."""
    rows, buckets = [], {}
    for record in records:
        if record['split'] == 'train':
            bucket = math.ceil(record['step'] / eval_every) * eval_every
            entry = buckets.setdefault(bucket, {'n': 0, 'epoch': record['epoch']})
            entry['n'] += 1
            entry['epoch'] = record['epoch']
            for key in LOSSES:
                if key in record:
                    entry[key] = entry.get(key, 0.0) + record[key]
        else:
            rows.append({'experiment': name, 'split': record['split'], 'epoch': record['epoch'],
                         'step': record['step'],
                         **{k: v for k, v in record.items() if k not in ('split', 'epoch', 'step')}})
    for step, entry in sorted(buckets.items()):
        n = entry.pop('n')
        epoch = entry.pop('epoch')
        rows.append({'experiment': name, 'split': 'train', 'epoch': epoch, 'step': step,
                     **{k: v / n for k, v in entry.items()}})
    return sorted(rows, key=lambda r: (r['split'], r['step']))


def summarize(campaign):
    campaign = Path(campaign)
    manifest = json.loads((campaign / 'manifest.json').read_text())
    rows, curves = [], []
    for experiment in manifest['experiments']:
        root = campaign / experiment['name']
        if not (root / 'metrics.jsonl').exists():
            continue
        records = read_jsonl(root / 'metrics.jsonl')
        vals = [r for r in records if r['split'] in ('val', 'final_val')]
        if not vals:
            continue
        config = json.loads((root / 'config.json').read_text())
        status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        selection = config.get('checkpoint_selection_metric', 'total')
        best = min(vals, key=lambda r: r.get(selection, math.inf))
        row = dict(name=experiment['name'],
                   **{key: experiment.get(key) for key in SETTINGS},
                   train_scenes_used=config.get('train_scenes'),
                   epochs_planned=config['arguments']['epochs'],
                   **config['parameters'], completed=status.get('completed', False),
                   elapsed_hours=(status.get('elapsed_seconds') or 0) / 3600,
                   peak_cuda_gb=(status.get('peak_cuda_bytes') or 0) / 1e9,
                   selection_metric=selection, best_step=best['step'], best_epoch=best['epoch'])
        for key in LOSSES:
            row[f'val_{key}'] = best.get(key)
        row['val_object_rotation_deg'] = math.degrees(best['object_rotation_angle'])
        row['val_camera_rotation_deg'] = math.degrees(best['camera_rotation_angle'])
        row['final_val_total'] = vals[-1]['total']
        row['final_val_objects_total'] = vals[-1].get('objects_total')
        readable_path = root / 'best_validation_metrics.json'
        readable = json.loads(readable_path.read_text()) if readable_path.exists() else None
        for column, path in READABLE.items():
            row[column] = lookup(readable, path)
        rows.append(row)
        curves += curve_rows(experiment['name'], records, experiment.get('eval_every', 300))
    (campaign / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    write_csv(campaign / 'summary.csv', rows)
    write_csv(campaign / 'curves.csv', curves)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    summarize(sys.argv[1])
