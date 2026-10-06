"""Discrete object matching for model outputs and generated scene records."""

import torch
from scipy.optimize import linear_sum_assignment


def object_predictions(pred):
    """Accept the model's flat keys, or the reference nested tensor interface."""
    objects = pred.get("objects", pred)

    logits = objects.get("class_logits", objects.get("logits"))
    positions = objects.get("position", objects.get("positions"))

    # Check the model output before building any matching cost.
    if not isinstance(logits, torch.Tensor) or not isinstance(positions, torch.Tensor):
        raise ValueError(
            "Object predictions require class_logits/logits "
            "and position/positions tensors"
        )

    if logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError("Object logits must have shape (B, Q, classes + 1)")

    if positions.shape != (*logits.shape[:2], 3):
        raise ValueError("Object positions must have shape (B, Q, 3)")

    if (
        positions.device != logits.device
        or not logits.is_floating_point()
        or not positions.is_floating_point()
    ):
        raise ValueError("Object predictions must be floating tensors on the same device")

    return logits, positions


def object_targets(target, logits, positions):
    """Read generated JSON objects (class_id, position) or tensor targets."""
    objects = target["objects"]

    # The dataset uses a list of object records. The tensor interface is also
    # accepted so the loss can be called after a custom collate function.
    if isinstance(objects, list):
        classes = [obj["class_id"] for obj in objects]
        points = [obj["position"] for obj in objects]

        if not objects:
            points = positions.new_empty((0, 3))
    else:
        classes = objects["classes"]
        points = objects["positions"]

    # Move targets next to predictions without changing position precision.
    classes = torch.as_tensor(classes, device=logits.device)
    points = torch.as_tensor(points, device=positions.device, dtype=positions.dtype)

    if classes.ndim != 1 or points.shape != (classes.numel(), 3):
        raise ValueError("Target classes and positions must have shapes (N,) and (N, 3)")

    invalid_class_type = (
        classes.is_floating_point()
        or classes.is_complex()
        or classes.dtype == torch.bool
    )

    if classes.numel() and invalid_class_type:
        raise ValueError("Target classes must be integer indices")

    classes = classes.long()

    if classes.numel() and ((classes < 0).any() or (classes >= logits.shape[-1] - 1).any()):
        raise ValueError("Target class index is outside the real object classes")

    return classes, points


class HungarianMatcher:
    """Match each scene using negative class probability and L1 position cost.

    Rectangular assignment matches min(Q, N) pairs. Returned indices live on
    the prediction device; matching never retains an autograd graph.
    """

    def __init__(self, class_cost=1.0, position_cost=1.0):
        if class_cost < 0 or position_cost < 0 or class_cost + position_cost == 0:
            raise ValueError("Matching costs must be nonnegative with at least one positive")

        self.class_cost = class_cost
        self.position_cost = position_cost

    @torch.no_grad()
    def forward(self, pred, targets):
        logits, positions = object_predictions(pred)

        if len(targets) != logits.shape[0]:
            raise ValueError("One target scene is required per batch element")

        matches = []

        # Each scene is matched independently because it may contain a
        # different number of target objects.
        for b, target in enumerate(targets):
            classes, points = object_targets(target, logits, positions)

            # There is no assignment to solve if either side is empty.
            if not classes.numel() or not logits.shape[1]:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                matches.append((empty, empty.clone()))
                continue

            # Classification favors confident predictions for the target
            # class, while position uses the pairwise Manhattan distance.
            cost = -self.class_cost * logits[b].float().softmax(-1)[:, classes]

            position_cost = torch.cdist(
                positions[b].float(),
                points.float(),
                p=1,
            )
            cost = cost + self.position_cost * position_cost

            # Hungarian assignment is discrete: detach before moving to CPU.
            rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())

            query_indices = torch.as_tensor(rows, device=logits.device)
            target_indices = torch.as_tensor(cols, device=logits.device)
            matches.append((query_indices, target_indices))

        return matches
