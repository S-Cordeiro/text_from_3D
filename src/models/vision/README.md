# Loading a vision checkpoint

Load a saved vision model for inference (CPU by default):

```python
from src.models.vision.model import SceneModel

model = SceneModel.from_checkpoint('runs/run_1/best.pt', device='cpu')
```

This restores the model configuration and all weights in evaluation mode. The
DINOv2 backbone must be cached locally or downloadable during construction.
Optimizer and scheduler states are not restored by this method.
