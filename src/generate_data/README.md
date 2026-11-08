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

`uniform_objects_constrained_scale_closeup_config.json` keeps the same objects,
placement and eight views, with a 55 mm lens instead of 40 mm. Objects appear
37.5% larger in each image dimension. Generate a two-scene preview with:

```bash
.venv/bin/python src/generate_data/generate_dataset.py --config src/generate_data/uniform_objects_constrained_scale_closeup_config.json --train 2 --val 0 --test 0 --output dataset_constrained_scale_closeup_preview
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

## Occlusions and multi-view questions

`occlusion_questions_config.json` brings low cameras (about 14° elevation)
close to a compact layout of 3–4 objects, so objects occlude one another. On
600 geometry-only scenes, each object covers about 50 × 50 px (35 × 35 px
with `random_8_views_config.json`). 21% of (object, view) pairs are partly
hidden, against 2% before, and 42% of scenes contain an object fully hidden in
at least one view. About 7.5% of the pairs touch the image border. These
figures predate the switch to fixed sizes. All objects now share the scale
`[0.6, 0.6, 0.6]`: regular cubes of edge 1.2, spheres of diameter 1.2, and
cylinders of diameter and height 1.2. The vision model therefore predicts no
scale. Objects take one of six named colors, never repeated within a type in a
scene, so every object can be named. Each scene also gets `questions.json`.

```bash
.venv/bin/python src/generate_data/generate_dataset.py --config src/generate_data/occlusion_questions_config.json --output dataset_occlusion_questions
```

| Parameter | Meaning |
| --- | --- |
| `objects.linked_scale_axes_by_type` | Scale axes copied from the first axis of each group: `{"sphere": [[0, 1, 2]]}` keeps spheres round, `{"cylinder": [[0, 1]]}` keeps cylinder sections circular. |
| `objects.color_palette` | Named RGB colors; each object draws one name, stored as `color_name`. Overrides `color_rgb`. |
| `visibility.stride` | Enables ray-cast visibility (schema_version 2). One ray per `stride` × `stride` pixel block; 2 by default, about 0.5 s per scene. |
| `occlusion.min_occluded_pairs` | Minimum number of (object, view) pairs whose `visible_fraction` is at most `occlusion.max_visible_fraction`, counting only pairs with at least `min_unoccluded_pixels`. Objects are re-placed up to `layout_attempts` times, then the last layout is kept with a warning. |
| `questions` | Enables `questions.json`; keys and defaults in `DEFAULT_SETTINGS` of `questions.py`. |

### Visibility

With a `visibility` section, each camera of `ground_truth.json` has a
`visibility` list with one record per object:

- `visible_pixels`: pixels where the object is the first surface hit.
- `unoccluded_pixels`: in-frame pixels it would cover without the other objects,
  so `visible_fraction = visible_pixels / unoccluded_pixels` measures occlusion,
  not truncation by the image border (`null` when the object is out of frame).
- `occluders`: hidden pixels per occluding object.
- `bbox_px`, `in_frame_fraction`: unclipped projected box and its share inside the image.

Visibility is computed geometrically (BVH ray casts, no rendering), so it can
reject layouts before rendering. Pixel counts are full-resolution estimates.

### Questions

`questions.py` builds templated English questions from `ground_truth.json`.
Answers are short and closed (`yes/no`, `left/right`, `in front/behind`, an
object type, `camera k`, a count, `45 degrees`), so they are scored by exact
match. Every question records its `family` and `template`, so accuracy can be
reported per skill:

| Family | Skill | Templates |
| --- | --- | --- |
| A | Relations as seen by a camera | `A_side`, `A_depth`, `A_closest`, `A_extreme` |
| B | Visibility and occlusion | `B_visible`, `B_count`, `B_hidden`, `B_partial`, `B_occluder`, `B_num_cameras` |
| D | Cross-view correspondence | `D_most_visible`, `D_shared`, `D_only`, `D_extreme_seen`, `D_closest_camera` |
| F | Perspective taking | `F_ego_side`, `F_ego_front`, `F_hidden_side` |
| G | Metric comparisons | `G_closer_to`, `G_nearest_object`, `G_closest_pair`, `G_camera_angle` |

Example question: *"The cyan sphere cannot be seen from camera 2. From that
camera, is it to the left or to the right of the red cube?"* Its answer is `left`.

Safeguards:

- Relations inside a safety margin are ambiguous and never asked. Examples: a
  horizontal gap under 5% of the image width, distances within 10–15% of each
  other, an object neither clearly visible nor clearly hidden.
- Objects are named by the shortest unambiguous expression (`the cube`, `the red
  cube`). Objects without one are never mentioned, but can still be an answer.
  Sizes are fixed, so there is no size question and no `the larger cube`.
- For each template, the most frequent answer is capped at about 1.2 / (number
  of answers) of a split's questions. This keeps the majority-answer baseline low.
- `requires_multiview` tags templates where some answers logically need another
  camera. It is a heuristic; a single-view ablation measures the actual need.

Regenerate questions without rendering again, or print per-template statistics
(counts, answer distribution, majority-answer accuracy):

```bash
python src/generate_data/questions.py --dataset dataset_occlusion_questions --overwrite
python src/generate_data/questions.py --dataset dataset_occlusion_questions --stats
```

Both paths use the same seeds and produce the same questions. Standalone
regeneration reads the `questions` section of the dataset's
`generation_config.json` and ignores `split_overrides`.
