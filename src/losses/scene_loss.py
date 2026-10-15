"""Supervised losses for the objects and cameras predicted in a scene."""

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
    Object scale is the local scaling factor, not the object's dimensions.
    Object rotation predictions concatenate the first two rotation columns.
    Scale and rotation supervision use the same class/position assignment.
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
            "object_scale": 1.0,
            "object_rotation": 1.0,
            "camera_position": 1.0,
            "camera_rotation": 1.0,
        }

        if weights is not None:
            if set(weights) - set(self.weights):
                raise ValueError("Unknown scene loss weight")

            self.weights.update(weights)

        self.no_object_weight = no_object_weight
        self.matcher = HungarianMatcher(class_cost, position_cost)

    def forward(self, pred, targets):
        """Return scalar tensors including the weighted total for backward()."""
        logits, positions, scales, rotations = self._object_predictions(pred)
        targets = self._normalize_targets(targets, logits.shape[0])

        if len(targets) != logits.shape[0]:
            raise ValueError("One target scene is required per batch element")

        if pred["cameras"]["position"].shape[0] != logits.shape[0]:
            raise ValueError("Object and camera predictions must have the same batch size")

        object_targets = [
            self._object_targets(target, logits, positions, scales, rotations)
            for target in targets
        ]
        target_classes = [target[0] for target in object_targets]
        target_positions = [target[1] for target in object_targets]

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
        predicted_scales, target_scales = [], []
        predicted_rotations, target_rotations = [], []

        # Gather matched positions and replace the labels of matched queries.
        for b, (queries, objects) in enumerate(matches):
            classes, points, sizes, matrices = object_targets[b]

            labels[b, queries] = classes[objects]
            predicted_points.append(positions[b, queries])
            target_points.append(points[objects])
            predicted_scales.append(scales[b, queries])
            target_scales.append(sizes[objects])
            predicted_rotations.append(rotations[b, queries])
            target_rotations.append(matrices[objects])

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
            object_scale = F.smooth_l1_loss(
                torch.cat(predicted_scales), torch.cat(target_scales)
            )
            object_rotation = geodesic_rotation_loss(
                rotation_6d_to_matrix(torch.cat(predicted_rotations)),
                torch.cat(target_rotations),
            )
        else:
            object_position = positions.sum() * 0
            object_scale = scales.sum() * 0
            object_rotation = rotations.sum() * 0

        camera_position, camera_rotation = self._camera_losses(pred, targets)

        total = (
            self.weights["object_class"] * object_class
            + self.weights["object_position"] * object_position
            + self.weights["object_scale"] * object_scale
            + self.weights["object_rotation"] * object_rotation
            + self.weights["camera_position"] * camera_position
            + self.weights["camera_rotation"] * camera_rotation
        )

        return {
            "total": total,
            "object_class": object_class,
            "object_position": object_position,
            "object_scale": object_scale,
            "object_rotation": object_rotation,
            "camera_position": camera_position,
            "camera_rotation": camera_rotation,
        }

    def _normalize_targets(self, targets, batch_size):
        """Convert PyTorch's default-collated targets to one dict per scene."""
        if not isinstance(targets, dict):
            return targets

        scenes = []

        for index in range(batch_size):
            scenes.append({
                group: {
                    name: value[index]
                    for name, value in targets[group].items()
                }
                for group in ("objects", "cameras")
            })

        return scenes

    def _object_predictions(self, pred):
        """Read the object branch and check its main tensor shapes."""
        objects = pred["objects"]
        logits = objects["class_logits"]
        positions = objects["position"]
        scales = objects["scale"]
        rotations = objects["rotation"]

        if logits.ndim != 3 or logits.shape[-1] < 2:
            raise ValueError("Object logits must have shape (B, Q, classes + 1)")

        query_shape = logits.shape[:2]
        if (
            positions.shape != (*query_shape, 3)
            or scales.shape != (*query_shape, 3)
            or rotations.shape != (*query_shape, 6)
        ):
            raise ValueError("Object prediction shapes are incompatible")

        return logits, positions, scales, rotations

    def _object_targets(self, target, logits, positions, scales, rotations):
        """Read object targets from generated JSON or a tensor dictionary."""
        objects = target["objects"]

        if isinstance(objects, list):
            classes = [obj["class_id"] for obj in objects]
            points = [obj["position"] for obj in objects]
            sizes = [obj["scale"] for obj in objects]
            matrices = [obj["rotation_matrix"] for obj in objects]

            if not objects:
                points = positions.new_empty((0, 3))
                sizes = scales.new_empty((0, 3))
                matrices = rotations.new_empty((0, 3, 3))
        else:
            classes = objects["classes"]
            points = objects["positions"]
            sizes = objects["scales"]
            matrices = objects["rotations"]

        classes = torch.as_tensor(classes, device=logits.device)
        classes = classes.long()
        points = torch.as_tensor(points, device=positions.device, dtype=positions.dtype)
        sizes = torch.as_tensor(sizes, device=scales.device, dtype=scales.dtype)
        matrices = torch.as_tensor(matrices, device=rotations.device, dtype=rotations.dtype)

        if matrices.shape == (classes.numel(), 6):
            matrices = rotation_6d_to_matrix(matrices)

        return classes, points, sizes, matrices

    def _camera_losses(self, pred, targets):
        """Compute camera losses while preserving the dataset view order."""
        cameras = pred["cameras"]
        predicted_positions = cameras["position"]
        predicted_poses = cameras["pose"]

        if predicted_positions.ndim != 3 or predicted_positions.shape[-1] != 3:
            raise ValueError("Camera positions must have shape (B, V, 3)")

        expected_pose_shape = (*predicted_positions.shape[:2], 6)
        if predicted_poses.shape != expected_pose_shape:
            raise ValueError("Camera poses must have shape (B, V, 6)")

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

            if rotations.shape == (n_views, 6):
                rotations = rotation_6d_to_matrix(rotations)

        if positions.shape != (n_views, 3) or rotations.shape != (n_views, 3, 3):
            raise ValueError("Target and predicted camera views are incompatible")

        return positions, rotations
