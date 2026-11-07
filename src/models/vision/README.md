# Loading a vision checkpoint

Load a saved vision model for inference (CPU by default):

```python
from src.models.vision.model import SceneModel

model = SceneModel.from_checkpoint('runs/run_1/best.pt', device='cpu')
```

This restores the model configuration and all weights in evaluation mode. The
DINOv2 backbone must be cached locally or downloadable during construction.
Optimizer and scheduler states are not restored by this method.

# Verified training and DINO fine-tuning

`python -m src.train --dataset DATASET` audits actual train/validation labels and
infers `n_classes` (three for cube/sphere/cylinder). An explicit incompatible
`--n-classes` fails before training: class ID 2 must never share the no-object
output. Historical runs with two declared classes on that dataset remain
historical records and are unsuitable as corrected baselines.

Architecture flags: `--d-model 384 --mv-layers 4`. Fine-tuning flags:
`--init-checkpoint REFERENCE/best.pt --dino-unfreeze-blocks 1 --dino-lr 1e-6`.
DINO keeps evaluation behavior while the selected final blocks and final layer
norm receive gradients. Training rejects more than 15 million trainable
parameters, including any unfrozen backbone blocks.

Use `--run-dir RUN --resume RUN/last.pt` with the same architecture, dataset and
schedule to resume a new-format training run, including optimizer, scheduler,
RNG and position within the epoch. `--batch-size` is the effective batch size;
`--micro-batch-size` controls gradient accumulation and defaults to four.
See `scripts/VISION_CAMPAIGN.md` for the controlled Bentley experiment protocol.
