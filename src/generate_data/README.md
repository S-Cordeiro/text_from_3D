# text_from_3D dataset generation

From the repository root, with Blender installed:

```bash
blender --background --python-exit-code 1 --python src/generate_data/generate_dataset.py -- --train 2 --val 1 --test 1
```

This first run produces 4 scenes (16 PNG images at 224 × 224). To generate a single example:

```bash
blender --background --python-exit-code 1 --python src/generate_data/generate_dataset.py -- --train 1 --val 0 --test 0 --output dataset_preview
```

You can also run `.venv/bin/python src/generate_data/generate_dataset.py` if `bpy` is installed in that environment. Cycles uses the CPU to run without a graphical interface or GPU. Rendering time depends on your hardware; `render.samples` controls quality. `render.max_bounces` limits light bounces (2 by default; 0 for direct lighting only).

```text
dataset/
  generation_config.json
  train/
    scene_000000/
      view_0.png
      view_1.png
      view_2.png
      view_3.png
      ground_truth.json
  val/
    scene_000000/...
  test/
    scene_000000/...
```

Scene numbers are local to each split. A nonempty output directory is rejected to prevent overwriting or mixing datasets. Each scene is moved to its final directory only after all configured images and its JSON file have been written. Completed scenes remain available if generation fails; choose a new output directory to rerun generation.

## Configuration

`src/generate_data/generation_config.json` contains scene counts, the seed, and rendering, object, and camera settings. `--config path.json` loads another configuration; `--train`, `--val`, `--test`, `--seed`, and `--output` allow quick experiments. Default counts: 6 training scenes, 2 validation scenes, and 2 test scenes.

The cameras point toward the center of the scene. Cubes have uniformly sampled rotations in SO(3) and independent scaling along each axis. They rest on a ground plane, with disjoint enclosing disks to prevent intersections. If an early position blocks later objects, placement restarts for all objects, preserving their sizes and rotations. `objects.placement_attempts` limits position trials per object; optional `objects.layout_attempts` limits complete layout attempts (100 by default). This placement method is conservative: if all attempts fail, increase `position_extent` or reduce object count, size, or `min_gap`. Increasing the extent can also require adjusting camera framing. Objects can still occlude one another in an image.

This placement change alters scenes generated with previous versions, even with the same seed. Runs using the same script, configuration, and Blender version remain reproducible.

To use different views for each split, add settings such as:

```json
"split_overrides": {
  "val": {"cameras": {"azimuths_deg": [65, 155, 245, 335]}},
  "test": {"cameras": {"azimuths_deg": [85, 175, 265, 355], "height": 6.0}}
}
```

Overrides are merged recursively. Each scene's seed depends on its split and scene number: increasing scene counts preserves previously generated scenes when using the same configuration and Blender version. Identical results across Blender versions are not guaranteed.

To add a primitive, extend `OBJECT_FACTORIES` and `CLASS_IDS`, then update `objects.types`. Each factory creates an object centered at the origin with a valid bounding box; placement, materials, cameras, and export remain shared. The number of cameras is determined by `cameras.azimuths_deg`.

## Two-cube and variable-count datasets

Both presets retain the six distinct face colors, fixed cubic scale, random
positions/rotations, and eight cameras of `single_cube_config.json`.
Each produces 5,000 training scenes, 500 validation scenes, and 100 test scenes
(44,800 PNG images).

| Configuration | Train objects per scene | Val/test objects per scene |
| --- | --- | --- |
| `two_cubes_config.json` | 2 | 2 |
| `variable_cubes_config.json` | 2–3 | 3–4 |

```bash
.venv/bin/python src/generate_data/generate_dataset.py --config src/generate_data/two_cubes_config.json --output dataset_two_cubes
.venv/bin/python src/generate_data/generate_dataset.py --config src/generate_data/variable_cubes_config.json --output dataset_variable_cubes
```

`objects.count_range: [min, max]` overrides `objects.count` and samples an
integer uniformly, including both endpoints, independently for each scene using
its seed. This does not enforce an exact balance across counts. Split overrides
set the validation/test range separately.

## Ground truth

### Uniform-color cubes, spheres/ellipsoids and cylinders

From the repository root:

```bash
blender --background --python-exit-code 1 --python src/generate_data/generate_dataset.py -- --config src/generate_data/uniform_objects_config.json --output dataset_uniform_objects
```

This preset produces **80 PNG images**: 10 training scenes with eight
views each, spaced 45° apart as in `two_cubes_config.json`, at 224 × 224.
Each scene contains 2–4 objects (uniformly sampled,
inclusive); each object's type is sampled independently from `cube`, `sphere`,
and `cylinder`. All three types need not occur in every scene. Each object has
one random, uniform material color, a random SO(3) rotation, and independent
XYZ scales in [0.35, 0.7]. Spheres therefore become ellipsoids; cylinders may
have elliptical sections. Base primitives span 2 units on each local axis.
The exported class IDs are cube=0, sphere=1, cylinder=2. Curved primitives use
64 segments, with 32 rings for spheres. Objects touch the ground at their
lowest mesh vertex; enclosing disks keep them apart. Lighting still creates
shading on the uniform materials.

Models configured for a fixed number of views/classes must use eight views
and three object classes for this preset.

#### Spheres without scaling and cylinders with axial scaling only

`uniform_objects_constrained_scale_config.json` retains 2–4 objects and eight
views per scene. Spheres use scale `[1, 1, 1]` (radius 1); cylinders use scale
`[1, 1, z]`, with z sampled in [0.35, 0.7] (radius 1, height 0.7–1.4).
The cylinder axis is its **local Z**, before the random object rotation.
Cubes retain independent XYZ scales in [0.35, 0.7]. Placement extent and camera
distance are increased to accommodate the larger spheres and cylinders.

```bash
blender --background --python-exit-code 1 --python src/generate_data/generate_dataset.py -- --config src/generate_data/uniform_objects_constrained_scale_config.json --train 10 --val 2 --test 2 --output dataset_constrained_scale
```

`objects.scale_ranges_by_type` maps primitive names to three positive intervals
in local XYZ order. These override `objects.scale` and `objects.scale_range`
for that type; absent types retain the shared scale settings. Equal interval
endpoints fix an axis, for example:

```json
"scale_ranges_by_type": {
  "sphere": [[1, 1], [1, 1], [1, 1]],
  "cylinder": [[1, 1], [1, 1], [0.35, 0.7]]
}
```

### Ten scenes with one multicolored cube

```bash
blender --background --python-exit-code 1 --python src/generate_data/generate_dataset.py -- --config src/generate_data/single_cube_config.json --output dataset_single_cube
```

This preset creates 10 scenes (8 train, 1 val, 1 test), each containing one
cube with equal edge lengths and six distinct face colors, with random positions
and rotations. Each scene has eight views spaced 45° apart (80 PNG images in total).
`objects.face_colors_rgb` optionally supplies six distinct RGB triplets in [0,1],
in Blender mesh polygon order, and overrides the single material color for cubes.
When enabled, annotations include `face_colors`, with each polygon's index,
local-space normal, and material RGB. The existing `color_rgb` field remains
the base color and does not describe individual faces. Omit `face_colors_rgb`
to keep the original single-color behavior.

The reference frame is that of `camera_0` (`view_0.png`): X points right, Y points up, and **-Z points forward**, with distances in meters. The ground plane and lighting are not target objects.

Each cube (`class_id = 0`) includes:

- `position`: center in camera 0's reference frame.
- `scale`: scaling along the base cube's local axes (base edge length: 2).
- `dimensions`: edge lengths, equal to `2 * scale` for cubes.
- `rotation_quaternion_wxyz`, `rotation_matrix`: object orientation in the reference frame.
- `rotation_6d`: the first column of the rotation matrix followed by the second, for the model's 6D rotation prediction head.
- `object_to_reference`, `object_to_world`: homogeneous 4 × 4 matrices including scaling, stored as lists of rows and applied to column vectors.
- `color_rgb`: material color before lighting and color management.

The JSON also describes each camera, its image, its transforms to the world and reference frames, and its `intrinsics_cv` and `world_to_camera_cv`. These last two fields use the OpenCV convention (X right, Y down, Z forward). Intrinsics are calculated for the perspective cameras used here, with horizontal sensor fit, square pixels, and no lens shift. Resolution, seed, and Blender version are also recorded.

## Configurable diagnostic scenes

Run `blender --background --python-exit-code 1 --python src/generate_data/generate_dataset.py -- --config src/generate_data/simple_config.json --output dataset_simple`.
This preset creates ten identical scenes: one centered cube, fixed scale/color,
no world rotation, and fixed cameras. Increase position_extent or enable random
rotations to create distinct examples. The four-view model contract is preserved.

| Parameter | Meaning |
| --- | --- |
| `objects.count` | Number of objects. |
| `objects.position_extent` | XY sampled in [-extent, extent]; 0 centers a single object. Z still rests on the ground. |
| `objects.rotation_mode` | `uniform` (default, uniform SO(3)), `none`, `fixed_euler`, or `euler_ranges`. |
| `objects.rotation_euler_deg` | Fixed XYZ Euler angles in degrees for fixed_euler. |
| `objects.rotation_euler_ranges_deg` | Three intervals, e.g. [[0,0],[0,0],[-20,20]], for euler_ranges. |
| `objects.scale` | Optional fixed XYZ scale, overriding scale_range. Remove to restore random scaling. |
| `objects.color_rgb` | Optional fixed RGB in [0,1]. Remove to restore random color. |
| `cameras.azimuth_jitter_deg` | Interval added independently to each camera's base azimuth per scene. |
| `cameras.radius_jitter`, `cameras.height_jitter` | Intervals added to radius/height in meters. |
| `cameras.pitch_jitter_deg`, `cameras.yaw_jitter_deg`, `cameras.roll_jitter_deg` | Local XYZ Euler offsets after look-at rotation; Z is optical roll. |

Camera intervals default to [0,0]. For example, roll_jitter_deg: [-15,15]
varies roll per view and scene; [15,15] applies a constant offset. Randomness
is seeded per scene, independently of object sampling. Orbit changes keep the
cameras aimed at target; pitch/yaw offsets change their viewing direction.
Large offsets may move objects out of frame. All options support split_overrides.

Object rotation settings use world coordinates. Targets are recomputed relative
to the actual camera 0, so rotation_mode: none does not mean an identity target.
Camera 0's relative pose remains identity, even when its world pose varies.

## Cheap rotation diagnostics

Run `.venv/bin/python tests/diagnose_rotations.py --dataset dataset_preview`.
This CPU-only script reads training JSON files, injects exact/known rotations
through SceneLoss, and optimizes free 6D vectors with three initializations.
It does not render or load images or the visual backbone.

For the full time-limited diagnostic suite, see
[`tests/ROTATION_DIAGNOSTICS.md`](../../tests/ROTATION_DIAGNOSTICS.md).
