"""Collect final/best validation, cost and training curves in machine-readable tables."""
import csv
import json
import math
from pathlib import Path
import sys


def summarize(campaign):
    campaign = Path(campaign)
    manifest = json.loads((campaign / 'manifest.json').read_text())
    rows = []
    for experiment in manifest['experiments']:
        root = campaign / experiment['name']
        if not (root / 'metrics.jsonl').exists():
            continue
        records = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
        vals = [r for r in records if r['split'] in ('val', 'final_val')]
        if not vals:
            continue
        best = min(vals, key=lambda r: r['total'])
        config = json.loads((root / 'config.json').read_text())
        status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        row = dict(name=experiment['name'], **config['parameters'], completed=status.get('completed', False),
            elapsed_seconds=status.get('elapsed_seconds'), peak_cuda_bytes=status.get('peak_cuda_bytes'),
            best_step=best['step'], best_total=best['total'],
            object_rotation_deg=math.degrees(best['object_rotation_angle']),
            camera_rotation_deg=math.degrees(best['camera_rotation_angle']))
        for key in ('object_class', 'object_position', 'object_scale', 'object_rotation',
                    'camera_position', 'camera_rotation', 'matching_accuracy'):
            row[key] = best[key]
        row['final_total'] = vals[-1]['total']
        rows.append(row)
    (campaign / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    if rows:
        with (campaign / 'summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    summarize(sys.argv[1])
