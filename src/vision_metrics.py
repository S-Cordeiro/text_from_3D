"""Interpretable validation metrics from exported best-checkpoint predictions.

`scripts/analyze_vision_checkpoint.py` writes one JSON line per validation
scene (predictions, targets, Hungarian matches) to
`RUN/best_validation_predictions.jsonl.gz`. This module turns it into numbers
in scene units and degrees, without GPU:

    python -m src.vision_metrics RUN [RUN ...]

Outputs in each RUN directory:
- `best_validation_metrics.json`: summary (means, medians, 90th percentiles).
- `best_validation_objects.csv`: one row per target object.
- `best_validation_cameras.csv`: one row per predicted camera (camera 0 is the
  reference frame and is excluded).

Rotation errors are taken modulo object symmetries: the 24 rotations of a
regular cube, the unoriented axis of a cylinder; spheres have none.
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import torch

from src.losses.geometric_utils import _CUBE_SYMMETRIES, rotation_6d_to_matrix

CLASS_NAMES = {0: 'cube', 1: 'sphere', 2: 'cylinder'}
# Expected error of a uniformly random rotation, for scale.
RANDOM_CUBE_ROTATION_DEG = 40.7
RANDOM_CYLINDER_AXIS_DEG = 57.3


def object_angle_deg(predicted, target, class_id):
    """Rotation error in degrees modulo the symmetries of the class, None for spheres."""
    if class_id == 1:
        return None
    if class_id == 2:
        cosine = (predicted[:, 2] * target[:, 2]).sum().abs()
    elif class_id == 0:
        relative = predicted.T @ target @ _CUBE_SYMMETRIES.to(predicted.dtype)
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).max()
    else:
        cosine = ((predicted.T @ target).trace() - 1) / 2
    return math.degrees(math.acos(float(cosine.clamp(-1, 1))))


def rotation_angle_deg(predicted, target):
    cosine = ((predicted.T @ target).trace() - 1) / 2
    return math.degrees(math.acos(float(cosine.clamp(-1, 1))))


def stats(values, threshold=None):
    """Mean, median and 90th percentile; share below a threshold if given."""
    if not values:
        return None
    ordered = torch.tensor(values, dtype=torch.float64)
    result = {
        'count': len(values),
        'mean': float(ordered.mean()),
        'median': float(ordered.median()),
        'p90': float(ordered.quantile(0.9)),
    }
    if threshold is not None:
        result[f'share_below_{threshold:g}'] = float((ordered < threshold).double().mean())
    return result


def scene_rows(record):
    """Per-object and per-camera rows of one exported scene."""
    predictions, targets = record['predictions'], record['targets']
    logits = torch.tensor(predictions['objects']['class_logits'])
    no_object = logits.shape[-1] - 1
    predicted_classes = logits.argmax(-1).tolist()
    target_classes = targets['objects']['classes']
    n_objects = len(target_classes)
    matches = dict(zip(record['matched_target_indices'], record['matched_query_indices']))
    rotations = rotation_6d_to_matrix(torch.tensor(predictions['objects']['rotation']))
    objects = []
    for index, class_id in enumerate(target_classes):
        query = matches.get(index)
        row = dict(scene=record['scene'], n_objects=n_objects, object=index,
                   target_class=CLASS_NAMES.get(class_id, str(class_id)), query=query)
        if query is None:
            row.update(predicted_class=None, detected=False, class_correct=False,
                       position_error=None, rotation_error_deg=None)
        else:
            predicted = predicted_classes[query]
            position = torch.tensor(predictions['objects']['position'][query])
            target_position = torch.tensor(targets['objects']['positions'][index])
            angle = object_angle_deg(rotations[query],
                                     torch.tensor(targets['objects']['rotations'][index]), class_id)
            row.update(predicted_class='none' if predicted == no_object else CLASS_NAMES.get(predicted, str(predicted)),
                       detected=predicted != no_object, class_correct=predicted == class_id,
                       position_error=float((position - target_position).norm()),
                       rotation_error_deg=angle)
        objects.append(row)
    matched_queries = set(matches.values())
    false_positives = sum(1 for q, c in enumerate(predicted_classes)
                          if q not in matched_queries and c != no_object)
    detected = sum(1 for c in predicted_classes if c != no_object)
    camera_positions = torch.tensor(predictions['cameras']['position'])
    camera_rotations = rotation_6d_to_matrix(torch.tensor(predictions['cameras']['pose']))
    target_positions = torch.tensor(targets['cameras']['positions'])
    target_rotations = torch.tensor(targets['cameras']['rotations'])
    cameras = []
    for k in range(1, len(target_positions)):
        cameras.append(dict(
            scene=record['scene'], camera=k,
            distance_to_reference=float(target_positions[k].norm()),
            position_error=float((camera_positions[k] - target_positions[k]).norm()),
            rotation_error_deg=rotation_angle_deg(camera_rotations[k], target_rotations[k])))
    scene = dict(scene=record['scene'], n_objects=n_objects, detected=detected,
                 false_positives=false_positives, count_correct=detected == n_objects)
    return scene, objects, cameras


def summarize(scenes, objects, cameras, target_positions):
    found = [o for o in objects if o['position_error'] is not None]
    summary = {
        'scenes': len(scenes),
        'objects': len(objects),
        'object_position_error': stats([o['position_error'] for o in found]),
        'object_position_error_by_class': {
            name: stats([o['position_error'] for o in found if o['target_class'] == name])
            for name in CLASS_NAMES.values()},
        'object_position_error_by_object_count': {
            str(n): stats([o['position_error'] for o in found if o['n_objects'] == n])
            for n in sorted({o['n_objects'] for o in objects})},
        'cube_rotation_error_deg': stats(
            [o['rotation_error_deg'] for o in found if o['target_class'] == 'cube'], threshold=10),
        'cylinder_axis_error_deg': stats(
            [o['rotation_error_deg'] for o in found if o['target_class'] == 'cylinder'], threshold=10),
        'class_accuracy': sum(o['class_correct'] for o in objects) / max(len(objects), 1),
        'count_accuracy': sum(s['count_correct'] for s in scenes) / max(len(scenes), 1),
        'count_accuracy_by_object_count': {
            str(n): (sum(s['count_correct'] for s in scenes if s['n_objects'] == n)
                     / sum(1 for s in scenes if s['n_objects'] == n))
            for n in sorted({s['n_objects'] for s in scenes})},
        # Extra detections, and target objects whose query predicts "no object".
        'false_positives': sum(s['false_positives'] for s in scenes),
        'missed_objects': sum(1 for o in objects if not o['detected']),
        'camera_position_error': stats([c['position_error'] for c in cameras]),
        'camera_rotation_error_deg': stats([c['rotation_error_deg'] for c in cameras]),
        'camera_errors_by_camera': {
            str(k): {'position_error': stats([c['position_error'] for c in cameras if c['camera'] == k]),
                     'rotation_error_deg': stats([c['rotation_error_deg'] for c in cameras if c['camera'] == k])}
            for k in sorted({c['camera'] for c in cameras})},
    }
    points = torch.tensor(target_positions, dtype=torch.float64)
    summary['references'] = {
        'mean_position_error': float((points - points.mean(0)).norm(dim=1).mean()) if len(points) else None,
        'random_cube_rotation_deg': RANDOM_CUBE_ROTATION_DEG,
        'random_cylinder_axis_deg': RANDOM_CYLINDER_AXIS_DEG,
        'median_camera_distance': stats([c['distance_to_reference'] for c in cameras])['median']
        if cameras else None,
    }
    return summary


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze_run(run):
    run = Path(run)
    source = run / 'best_validation_predictions.jsonl.gz'
    scenes, objects, cameras, positions = [], [], [], []
    with gzip.open(source, 'rt') as file:
        for line in file:
            record = json.loads(line)
            scene, scene_objects, scene_cameras = scene_rows(record)
            scenes.append(scene)
            objects.extend(scene_objects)
            cameras.extend(scene_cameras)
            positions.extend(record['targets']['objects']['positions'])
    summary = summarize(scenes, objects, cameras, positions)
    summary['source'] = source.name
    temporary = run / 'best_validation_metrics.tmp'
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(run / 'best_validation_metrics.json')
    write_csv(run / 'best_validation_objects.csv', objects)
    write_csv(run / 'best_validation_cameras.csv', cameras)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    for run in args.runs:
        summary = analyze_run(run)
        print(f"{run}: position {summary['object_position_error']['mean']:.3f} u, "
              f"cube {summary['cube_rotation_error_deg']['mean']:.1f} deg, "
              f"count {100 * summary['count_accuracy']:.1f}%")


if __name__ == '__main__':
    main()
