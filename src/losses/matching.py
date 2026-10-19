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
        """Return independent query/target assignments for a batch of scenes.

        Args:
            logits: (B, Q, C) class scores, including the final no-object class.
            positions: (B, Q, 3) predicted query positions.
            target_classes: List of B tensors (N_b,) containing target class
                indices on the prediction device.
            target_positions: List of B tensors (N_b, 3) containing target
                positions on the prediction device.

        B is the batch size, Q the number of queries, C the number of classes
        including no-object, and N_b the target count in scene b. Target counts
        may differ between scenes, including empty scenes.

        Returns:
            List of B (query_indices, target_indices) pairs on logits.device.
            Both tensors have dtype long and shape (min(Q, N_b),); corresponding
            entries identify an assigned query and target in the same scene.

        Each cost matrix has shape (Q, N_b). Flattened costs are concatenated
        for one transfer to CPU, where SciPy solves each scene independently.
        All assignment indices are then transferred together to the prediction
        device. Matching does not retain an autograd graph.
        """
        costs = []
        target_counts = []
        query_count = logits.shape[1]

        # Each scene is matched independently because it may contain a
        # different number of target objects.
        for b, (classes, points) in enumerate(zip(target_classes, target_positions)):
            target_counts.append(classes.numel())

            # There is no assignment to solve if either side is empty.
            if not classes.numel() or not query_count:
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

            costs.append(cost.reshape(-1))

        if not costs:
            return [
                (torch.empty(0, dtype=torch.long, device=logits.device),
                 torch.empty(0, dtype=torch.long, device=logits.device))
                for _ in target_counts
            ]

        # Pack variable-size matrices so the batch requires only one GPU->CPU
        # transfer. Offsets restore each scene's (Q, N_b) matrix without padding.
        cpu_costs = torch.cat(costs).cpu().numpy()
        cpu_matches = []
        match_counts = []
        offset = 0
        for target_count in target_counts:
            size = query_count * target_count
            cost = cpu_costs[offset:offset + size].reshape(query_count, target_count)
            rows, cols = linear_sum_assignment(cost)
            cpu_matches.append(torch.stack((torch.as_tensor(rows), torch.as_tensor(cols))))
            match_counts.append(len(rows))
            offset += size

        # (2, sum_b min(Q, N_b)): transfer indices once, then return per-scene views.
        indices = torch.cat(cpu_matches, dim=1).to(device=logits.device, dtype=torch.long)
        return [(scene[0], scene[1]) for scene in indices.split(match_counts, dim=1)]
