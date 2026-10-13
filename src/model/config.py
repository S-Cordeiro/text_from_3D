from dataclasses import dataclass, field

@dataclass
class MultiViewConfig:
    n_head : int = 4
    n_layers : int = 4

    n_cameras : int = 4

@dataclass
class ObjectDecoderConfig:
    n_head : int = 4
    n_layers : int = 4

    n_objects : int = 10

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
