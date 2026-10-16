from dataclasses import dataclass, field

@dataclass
class MultiViewConfig:
    n_head : int = 4
    n_layers : int = 8

    n_cameras : int = 8
    max_tokens_per_view : int = 256  # 16 x 16 patches, independent of d_model

@dataclass
class ObjectDecoderConfig:
    n_head : int = 4
    n_layers : int = 4

    n_objects : int = 5

@dataclass
class CameraDecoderConfig:
    n_head : int = 4
    n_layers : int = 4

@dataclass 
class SceneModelConfig:
    d_model : int = 256

    multiview : MultiViewConfig = field(default_factory=MultiViewConfig)
    object_decoder : ObjectDecoderConfig = field(default_factory=ObjectDecoderConfig)

    n_classes : int = 2
    camera_decoder : CameraDecoderConfig = field(default_factory=CameraDecoderConfig)

    def __post_init__(self):
        for name in ("multiview", "object_decoder", "camera_decoder"):
            n_head = getattr(self, name).n_head
            if n_head <= 0 or self.d_model % n_head != 0:
                raise ValueError(f"d_model ({self.d_model}) must be divisible by {name}.n_head ({n_head})")
