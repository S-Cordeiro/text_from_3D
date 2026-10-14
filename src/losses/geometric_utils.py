"""Differentiable helpers for 3D rotations."""

import torch
from torch.nn import functional as F


def rotation_6d_to_matrix(rotation_6d):
    """Convert two predicted rotation columns to an orthonormal matrix."""
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]

    first = F.normalize(first, dim=-1)
    projection = (first * second).sum(dim=-1, keepdim=True) * first
    second = F.normalize(second - projection, dim=-1)
    third = torch.cross(first, second, dim=-1)

    return torch.stack((first, second, third), dim=-1)


def geodesic_rotation_loss(predicted, target):
    """Return the mean relative rotation angle in radians."""
    relative = predicted.transpose(-1, -2) @ target
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = (trace - 1.0) / 2.0

    epsilon = torch.finfo(cosine.dtype).eps
    angle = torch.acos(cosine.clamp(-1.0 + epsilon, 1.0 - epsilon))

    return angle.mean()
