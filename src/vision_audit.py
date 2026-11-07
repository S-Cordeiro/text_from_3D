"""Audit actual scene labels and geometry metadata, without trusting generator configs."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def audit_dataset(root):
    root = Path(root).resolve()
    result = {'root': str(root), 'splits': {}}
    all_classes = set()
    for split in ('train', 'val'):
        paths = sorted((root / split).glob('scene_*/ground_truth.json'))
        if not paths:
            raise ValueError(f'Empty dataset split: {root / split}')
        labels, types, counts, views, sizes = (Counter() for _ in range(5))
        digest = hashlib.sha256()
        for path in paths:
            raw = path.read_bytes()
            digest.update(str(path.relative_to(root)).encode() + b'\0' + raw)
            truth = json.loads(raw)
            counts[len(truth['objects'])] += 1
            views[len(truth['cameras'])] += 1
            for obj in truth['objects']:
                label = obj['class_id']
                if type(label) is not int or label not in (0, 1, 2):
                    raise ValueError(f'Unsupported class {label}: {path}')
                expected = {'cube': 0, 'sphere': 1, 'cylinder': 2}
                if obj.get('type') in expected and expected[obj['type']] != label:
                    raise ValueError(f'Inconsistent type/class: {path}')
                labels[label] += 1
                types[f"{obj.get('type', 'unknown')}:{label}"] += 1
            for camera in truth['cameras']:
                image = path.parent / camera['image']
                if not image.is_file():
                    raise FileNotFoundError(image)
                matrix = camera['camera_to_reference']
                if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                    raise ValueError(f'Invalid camera transform: {path}')
            if not truth['cameras']:
                raise ValueError(f'No cameras: {path}')
            with Image.open(path.parent / truth['cameras'][0]['image']) as image:
                sizes[f'{image.width}x{image.height}'] += 1
        all_classes.update(labels)
        result['splits'][split] = dict(scenes=len(paths), class_counts=dict(labels),
            type_class_counts=dict(types), object_counts=dict(counts), view_counts=dict(views),
            first_view_resolutions=dict(sizes), ground_truth_sha256=digest.hexdigest())
    if not all_classes:
        raise ValueError('Cannot infer classes from an entirely empty-object dataset')
    result['n_classes'] = max(all_classes) + 1
    result['max_objects'] = max(int(k) for s in result['splits'].values() for k in s['object_counts'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--output', required=True)
    parser.add_argument('--historic-runs', default='runs')
    args = parser.parse_args()
    report = audit_dataset(args.dataset)
    report['historic_config_checks'] = []
    for path in sorted(Path(args.historic_runs).glob('**/config.json')):
        config = json.loads(path.read_text())
        if Path(config.get('dataset_path', '')).name != Path(args.dataset).name:
            continue
        declared = config.get('model', {}).get('n_classes')
        report['historic_config_checks'].append(dict(path=str(path), n_classes=declared,
            valid=declared is not None and declared >= report['n_classes'],
            note='Historical records preserved; incompatible runs must not serve as baselines.'))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
