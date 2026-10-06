"""Supervised losses for the objects predicted in a scene."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .matching import HungarianMatcher, object_predictions, object_targets


class SceneLoss(nn.Module):
    """Compute classification and position losses for object queries.

    The loss accepts the flat output returned by ``SceneModel`` and a sequence
    of per-scene JSON or tensor targets. Scale and rotation predictions are not
    used yet. The returned component losses are unweighted; only ``total``
    applies the configured weights.
    """

    def __init__(
        self,
        weights=None,
        no_object_weight=0.1,
        class_cost=1.0,
        position_cost=1.0,
    ):
        super().__init__()

        self.weights = {
            "object_class": 1.0,
            "object_position": 1.0,
        }

        if weights is not None:
            if set(weights) - set(self.weights):
                raise ValueError("Unknown scene loss weight")

            self.weights.update(weights)

        if any(not math.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative")

        if not math.isfinite(no_object_weight) or no_object_weight <= 0:
            raise ValueError("no_object_weight must be finite and positive")

        self.no_object_weight = no_object_weight
        self.matcher = HungarianMatcher(class_cost, position_cost)

    def forward(self, pred, targets):
        """Return scalar tensors including the weighted total for backward()."""
        logits, positions = object_predictions(pred)
        matches = self.matcher.forward(pred, targets)

        # All queries represent "no object" until the matching assigns them a
        # real target class.
        no_object_class = logits.shape[-1] - 1
        labels = torch.full(
            logits.shape[:2],
            no_object_class,
            dtype=torch.long,
            device=logits.device,
        )

        predicted_points = []
        target_points = []

        # Gather matched positions and replace the labels of matched queries.
        for b, (queries, objects) in enumerate(matches):
            classes, points = object_targets(targets[b], logits, positions)

            labels[b, queries] = classes[objects]
            predicted_points.append(positions[b, queries])
            target_points.append(points[objects])

        # Downweight the final class so unmatched queries do not dominate the
        # classification term.
        class_weights = logits.new_ones(logits.shape[-1])
        class_weights[no_object_class] = self.no_object_weight

        if labels.numel():
            object_class = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                weight=class_weights,
            )
        else:
            object_class = logits.sum() * 0

        # Position supervision only applies to queries paired with an object.
        # The empty-scene branch remains connected to the prediction graph.
        if any(q.numel() for q, _ in matches):
            object_position = F.smooth_l1_loss(
                torch.cat(predicted_points),
                torch.cat(target_points),
            )
        else:
            object_position = positions.sum() * 0

        total = (
            self.weights["object_class"] * object_class
            + self.weights["object_position"] * object_position
        )

        return {
            "total": total,
            "object_class": object_class,
            "object_position": object_position,
        }
