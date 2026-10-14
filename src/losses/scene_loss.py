"""Supervised losses for the objects and cameras predicted in a scene."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .geometric_utils import geodesic_rotation_loss, rotation_6d_to_matrix
from .matching import HungarianMatcher


class SceneLoss(nn.Module):
    """Compute object and camera losses for a batch of scenes.

    ``SceneModel`` groups its predictions under ``objects`` and ``cameras``.
    Cameras are paired by view order; only objects require Hungarian matching.
    The returned component losses are unweighted; only ``total`` applies the
    configured weights.
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
            "camera_position": 1.0,
            "camera_rotation": 1.0,
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
        logits, positions = self._object_predictions(pred)

        object_targets = [
            self._object_targets(target, logits, positions)
            for target in targets
        ]
        target_classes = [classes for classes, _ in object_targets]
        target_positions = [points for _, points in object_targets]

        matches = self.matcher.forward(
            logits,
            positions,
            target_classes,
            target_positions,
        )

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
            classes, points = object_targets[b]

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

        camera_position, camera_rotation = self._camera_losses(pred, targets)

        total = (
            self.weights["object_class"] * object_class
            + self.weights["object_position"] * object_position
            + self.weights["camera_position"] * camera_position
            + self.weights["camera_rotation"] * camera_rotation
        )

        return {
            "total": total,
            "object_class": object_class,
            "object_position": object_position,
            "camera_position": camera_position,
            "camera_rotation": camera_rotation,
        }

    def _object_predictions(self, pred):
        """Read and validate the object branch returned by SceneModel."""
        objects = pred.get("objects")

        if not isinstance(objects, dict):
            raise ValueError("Predictions require an objects dictionary")

        logits = objects.get("class_logits")
        positions = objects.get("position")

        if not isinstance(logits, torch.Tensor):
            raise ValueError("Object predictions require a class_logits tensor")

        if not isinstance(positions, torch.Tensor):
            raise ValueError("Object predictions require a position tensor")

        if logits.ndim != 3 or logits.shape[-1] < 2:
            raise ValueError("Object logits must have shape (B, Q, classes + 1)")

        if positions.shape != (*logits.shape[:2], 3):
            raise ValueError("Object positions must have shape (B, Q, 3)")

        if positions.device != logits.device:
            raise ValueError("Object predictions must be on the same device")

        if not logits.is_floating_point() or not positions.is_floating_point():
            raise ValueError("Object predictions must be floating tensors")

        return logits, positions

    def _object_targets(self, target, logits, positions):
        """Read object targets from generated JSON or a tensor dictionary."""
        objects = target["objects"]

        if isinstance(objects, list):
            classes = [obj["class_id"] for obj in objects]
            points = [obj["position"] for obj in objects]

            if not objects:
                points = positions.new_empty((0, 3))
        else:
            classes = objects["classes"]
            points = objects["positions"]

        classes = torch.as_tensor(classes, device=logits.device)
        points = torch.as_tensor(
            points,
            device=positions.device,
            dtype=positions.dtype,
        )

        if classes.ndim != 1 or points.shape != (classes.numel(), 3):
            raise ValueError(
                "Target classes and positions must have shapes (N,) and (N, 3)"
            )

        invalid_class_type = (
            classes.is_floating_point()
            or classes.is_complex()
            or classes.dtype == torch.bool
        )

        if classes.numel() and invalid_class_type:
            raise ValueError("Target classes must be integer indices")

        classes = classes.long()

        if classes.numel() and (
            (classes < 0).any()
            or (classes >= logits.shape[-1] - 1).any()
        ):
            raise ValueError("Target class index is outside the real object classes")

        return classes, points

    def _camera_losses(self, pred, targets):
        """Compute camera losses while preserving the dataset view order."""
        cameras = pred.get("cameras")

        if not isinstance(cameras, dict):
            raise ValueError("Predictions require a cameras dictionary")

        predicted_positions = cameras.get("position")
        predicted_poses = cameras.get("pose")

        if not isinstance(predicted_positions, torch.Tensor):
            raise ValueError("Camera predictions require a camera_position tensor")

        if not isinstance(predicted_poses, torch.Tensor):
            raise ValueError("Camera predictions require a camera_pose tensor")

        if predicted_positions.ndim != 3 or predicted_positions.shape[-1] != 3:
            raise ValueError("Camera positions must have shape (B, V, 3)")

        expected_pose_shape = (*predicted_positions.shape[:2], 6)
        if predicted_poses.shape != expected_pose_shape:
            raise ValueError("Camera poses must have shape (B, V, 6)")

        if len(targets) != predicted_positions.shape[0]:
            raise ValueError("One target scene is required per batch element")

        if predicted_positions.device != predicted_poses.device:
            raise ValueError("Camera predictions must be on the same device")

        if (
            not predicted_positions.is_floating_point()
            or not predicted_poses.is_floating_point()
        ):
            raise ValueError("Camera predictions must be floating tensors")

        target_positions = []
        target_rotations = []

        # Extract translation and rotation from camera-to-reference transforms.
        for target in targets:
            positions, rotations = self._camera_targets(
                target,
                predicted_positions,
                predicted_poses,
            )
            target_positions.append(positions)
            target_rotations.append(rotations)

        target_positions = torch.stack(target_positions)
        target_rotations = torch.stack(target_rotations)

        camera_position = F.smooth_l1_loss(
            predicted_positions,
            target_positions,
        )

        predicted_rotations = rotation_6d_to_matrix(predicted_poses)
        camera_rotation = geodesic_rotation_loss(
            predicted_rotations,
            target_rotations,
        )

        return camera_position, camera_rotation

    def _camera_targets(self, target, predicted_positions, predicted_poses):
        """Read camera targets from generated JSON or a tensor dictionary."""
        cameras = target["cameras"]
        n_views = predicted_positions.shape[1]

        if isinstance(cameras, list):
            transforms = torch.as_tensor(
                [camera["camera_to_reference"] for camera in cameras],
                device=predicted_positions.device,
                dtype=predicted_positions.dtype,
            )

            if transforms.shape != (n_views, 4, 4):
                raise ValueError(
                    "Target camera transforms must have shape (V, 4, 4)"
                )

            positions = transforms[:, :3, 3]
            rotations = transforms[:, :3, :3]

        else:
            positions = torch.as_tensor(
                cameras["positions"],
                device=predicted_positions.device,
                dtype=predicted_positions.dtype,
            )

            rotations = torch.as_tensor(
                cameras["rotations"],
                device=predicted_poses.device,
                dtype=predicted_poses.dtype,
            )

            if positions.shape != (n_views, 3):
                raise ValueError("Target camera positions must have shape (V, 3)")

            if rotations.shape == (n_views, 6):
                rotations = rotation_6d_to_matrix(rotations)
            elif rotations.shape != (n_views, 3, 3):
                raise ValueError(
                    "Target camera rotations must have shape (V, 6) or (V, 3, 3)"
                )

        return positions, rotations
