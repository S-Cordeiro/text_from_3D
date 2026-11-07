# Text from 3D

**Answering spatial questions about a 3D scene from several views of it.**

A personal project to learn how to build a vision-language model (VLM) from
scratch. Scenes of simple objects are rendered with Blender from eight cameras.
The model answers questions that need several views to resolve, for example:

> *The cyan sphere cannot be seen from camera 2. From that camera, is it to the
> left or to the right of the red cube?* → `left`

![A rendered dataset view showing two cubes with distinct face colors.](figures/dataset-example.png)

## Components

- **Synthetic data** ([src/generate_data](src/generate_data)): Blender scenes
  with exact ground truth, per-view occlusion, and CLEVR-style templated
  questions grouped by skill: viewpoint relations, occlusion, cross-view
  correspondence, perspective taking, metric comparisons.
- **Vision** ([src/models/vision](src/models/vision)): DETR-style model. A
  frozen DINOv2 encodes each view, attention fuses the views, and object
  queries predict each object's type and pose. Hungarian matching pairs
  predictions with targets during training. A first experiment reached an
  object rotation error of about 8° on cubes.
- **Language** ([src/models/language](src/models/language)): a 96M-parameter
  GPT pretrained on about 2B FineWeb-Edu tokens.
- **VLM** ([src/models/vlm](src/models/vlm)): two ways to connect the vision
  model to the GPT. LLaVA-style feeds visual tokens to the GPT as a prefix;
  Flamingo-style adds gated cross-attention layers.

Training instructions: [src/README.md](src/README.md).

## Bibliography

- Carion et al. *End-to-End Object Detection with Transformers* (DETR). ECCV 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- Johnson et al. *CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning*. CVPR 2017. [arXiv:1612.06890](https://arxiv.org/abs/1612.06890)
- Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning*. NeurIPS 2022. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
- Liu et al. *Visual Instruction Tuning* (LLaVA). NeurIPS 2023. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
