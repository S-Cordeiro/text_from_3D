"""Discrete matching between predicted and target objects."""

import torch
from scipy.optimize import linear_sum_assignment


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
    def forward(self, logits, positions, target_classes, target_positions):
        """Return matched query and target indices for every batch element."""
        batch_size = logits.shape[0]

        if len(target_classes) != batch_size or len(target_positions) != batch_size:
            raise ValueError("One set of object targets is required per batch element")

        matches = []

        # Each scene is matched independently because it may contain a
        # different number of target objects.
        for b, (classes, points) in enumerate(zip(target_classes, target_positions)):

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
