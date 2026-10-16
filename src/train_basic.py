

import torch
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader
from torch.optim import AdamW

import sys
import os

from src.dataset.scene_dataset import SceneDataset
from src.model.model import SceneModel
from src.model.config import SceneModelConfig
from src.losses.scene_loss import SceneLoss

# ----------------------------
# Dataset, DataLoader
# ----------------------------

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

def transform(image):
    return processor(
        images=image,
        do_rescale=False, 
        do_resize=False,
        do_center_crop=False,
        return_tensors="pt",
    )["pixel_values"][0]

scene_dataset = SceneDataset("dataset_single_cube", transform=transform)

train_loader = DataLoader(
    dataset=scene_dataset,
    batch_size=1,
)

config = SceneModelConfig()
model = SceneModel(config)

scene_loss = SceneLoss()

# -------------------------------
# training loop
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

# model = torch.compile(model)
model = model.to(device)
scene_loss = scene_loss.to(device)

optimizer = AdamW(
    filter(lambda p : p.requires_grad, model.parameters()),
    lr=1e-4,
)

for i in range(50):
    for batch_idx, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images = batch["images"].to(device)
        pred = model(images)
        losses = scene_loss(pred, batch["targets"])
        loss = losses["total"]

        loss.backward()

        optimizer.step()

        if batch_idx % 5 == 0:
            loss_details = " | ".join(
                f"{name}={value.detach().item():.6f}"
                for name, value in losses.items()
            )
            print(
                f"epoch {i + 1}/50 | batch {batch_idx}/{len(train_loader)} | {loss_details}",
                flush=True,
            )