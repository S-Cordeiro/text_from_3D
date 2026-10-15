"""Load rendered scenes and tensor targets accepted by SceneLoss."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SceneDataset(Dataset):
    """Load one scene per sample from ``root/split/scene_*/``.

    Samples contain ``images`` (V, 3, H, W) and ``targets``. The latter groups
    ``objects`` with ``classes`` (N,), ``positions`` and ``scales`` (N, 3),
    ``rotations`` (N, 3, 3), and ``cameras`` with ``positions`` (V, 3) and
    ``rotations`` (V, 3, 3). All geometric targets use the generator's
    camera-0 reference frame. Camera order follows the JSON.

    Images are float32 RGB tensors in [0, 1]. An optional ``transform``
    receives each image tensor, allowing resizing or model normalization.
    It must preserve image geometry: crops/flips would invalidate targets.
    Use ``collate_scenes`` with a DataLoader to keep variable-length targets.
    """

    def __init__(self, root, split="train", transform=None):
        self.directory = Path(root) / split
        self.transform = transform
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Dataset split does not exist: {self.directory}")
        self.scene_paths = sorted(
            path for path in self.directory.glob("scene_*") if path.is_dir()
        )
        for path in self.scene_paths:
            if not (path / "ground_truth.json").is_file():
                raise FileNotFoundError(f"Missing ground truth: {path}")

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, index):
        scene_path = self.scene_paths[index]
        with (scene_path / "ground_truth.json").open(encoding="utf-8") as file:
            truth = json.load(file)

        images = []
        for camera in truth["cameras"]:
            with Image.open(scene_path / camera["image"]) as image:
                pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
            if self.transform is not None:
                tensor = self.transform(tensor)
            images.append(tensor)

        if not images:
            raise ValueError(f"Scene has no camera views: {scene_path}")
        transforms = torch.tensor(
            [camera["camera_to_reference"] for camera in truth["cameras"]],
            dtype=torch.float32,
        )
        if transforms.shape != (len(images), 4, 4):
            raise ValueError(f"Invalid camera transforms: {scene_path}")

        return {
            "images": torch.stack(images),
            "targets": {
                "objects": {
                    "classes": torch.tensor(
                        [obj["class_id"] for obj in truth["objects"]],
                        dtype=torch.long,
                    ),
                    "positions": torch.tensor(
                        [obj["position"] for obj in truth["objects"]],
                        dtype=torch.float32,
                    ).reshape(-1, 3),
                    "scales": torch.tensor(
                        [obj["scale"] for obj in truth["objects"]],
                        dtype=torch.float32,
                    ).reshape(-1, 3),
                    "rotations": torch.tensor(
                        [obj["rotation_matrix"] for obj in truth["objects"]],
                        dtype=torch.float32,
                    ).reshape(-1, 3, 3),
                },
                "cameras": {
                    "positions": transforms[:, :3, 3].contiguous(),
                    "rotations": transforms[:, :3, :3].contiguous(),
                },
            },
        }


def collate_scenes(samples):
    """Stack images and retain one target dictionary per scene for SceneLoss."""
    return {
        "images": torch.stack([sample["images"] for sample in samples]),
        "targets": [sample["targets"] for sample in samples],
    }
