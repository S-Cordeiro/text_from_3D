"""Differentiable helpers for 3D rotations."""

from itertools import permutations, product

import torch
from torch.nn import functional as F


# Proper signed permutation matrices: the 24 rotations of a regular cube.
def _cube_symmetries():
    matrices = []
    for permutation in permutations(range(3)):
        inversions = sum(permutation[i] > permutation[j]
                         for i in range(3) for j in range(i + 1, 3))
        for signs in product((-1, 1), repeat=3):
            if (-1) ** inversions * signs[0] * signs[1] * signs[2] == 1:
                matrix = torch.zeros(3, 3)
                matrix[range(3), permutation] = torch.tensor(signs, dtype=matrix.dtype)
                matrices.append(matrix)
    return torch.stack(matrices)


_CUBE_SYMMETRIES = _cube_symmetries()


def rotation_6d_to_matrix(rotation_6d):
    """Convert two predicted rotation columns to an orthonormal matrix."""
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]

    first = F.normalize(first, dim=-1)
    projection = (first * second).sum(dim=-1, keepdim=True) * first
    second = F.normalize(second - projection, dim=-1)
    third = torch.cross(first, second, dim=-1)

    return torch.stack((first, second, third), dim=-1)


def matrix_rotation_loss(predicted, target):
    """Squared Frobenius distance, averaged over rotations (not coefficients)."""
    return (predicted - target).square().sum(dim=(-1, -2)).mean()


def object_rotation_loss(predicted, target, classes):
    """Mean loss by class: cube=0, sphere=1, cylinder=2.

    Cubes minimize squared Frobenius distance over their 24 proper rotations:
    equivalent targets are target @ S, with S acting in local coordinates.
    This assumes regular cubes, independently of the supervised XYZ scales.
    Cylinders compare their unoriented local Z axes; spheres contribute zero.
    Other classes retain the squared Frobenius distance.
    """
    losses = (predicted - target).square().sum(dim=(-1, -2))
    symmetries = _CUBE_SYMMETRIES.to(device=target.device, dtype=target.dtype)
    equivalent_targets = target.unsqueeze(-3) @ symmetries
    cube_losses = (predicted.unsqueeze(-3) - equivalent_targets).square().sum(
        dim=(-1, -2)
    ).min(dim=-1).values
    losses = torch.where(classes == 0, cube_losses, losses)
    # R @ z is the last column of R. Both axes have unit length.
    axis_cosine = (predicted[..., :, 2] * target[..., :, 2]).sum(dim=-1)
    cylinder_losses = 1 - axis_cosine.abs().clamp(max=1)
    losses = torch.where(classes == 2, cylinder_losses, losses)
    losses = torch.where(classes == 1, losses * 0, losses)
    return losses.mean()


def object_scale_loss(predicted, target, classes):
    """Mean per-object Smooth L1: XYZ for cubes, Z for cylinders, zero for spheres."""
    errors = F.smooth_l1_loss(predicted, target, reduction="none")
    losses = torch.where(classes == 2, errors[..., 2], errors.mean(dim=-1))
    losses = torch.where(classes == 1, losses * 0, losses)
    return losses.mean()


def object_rotation_angle(predicted, target, classes):
    """Mean angle in radians modulo object symmetries, excluding spheres."""
    keep = classes != 1
    if not keep.any():
        return predicted.new_zeros(())
    predicted, target, classes = predicted[keep], target[keep], classes[keep]
    relative = predicted.transpose(-1, -2) @ target
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    symmetries = _CUBE_SYMMETRIES.to(device=target.device, dtype=target.dtype)
    cube_relative = predicted.unsqueeze(-3).transpose(-1, -2) @ (
        target.unsqueeze(-3) @ symmetries
    )
    cube_cosine = (
        cube_relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1
    ).max(dim=-1).values / 2
    cosine = torch.where(classes == 0, cube_cosine, cosine)
    axis_cosine = (predicted[..., :, 2] * target[..., :, 2]).sum(-1).abs()
    cosine = torch.where(classes == 2, axis_cosine, cosine)
    return torch.acos(cosine.clamp(-1, 1)).mean()


def geodesic_rotation_loss(predicted, target):
    """Return the mean relative rotation angle in radians."""
    relative = predicted.transpose(-1, -2) @ target
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = (trace - 1.0) / 2.0

    epsilon = torch.finfo(cosine.dtype).eps
    angle = torch.acos(cosine.clamp(-1.0 + epsilon, 1.0 - epsilon))

    return angle.mean()
