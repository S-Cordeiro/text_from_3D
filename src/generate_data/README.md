# text_from_3D dataset generation

From the repository root, with Blender installed:

```bash
blender --background --python-exit-code 1 --python generate_data/generate_dataset.py -- --train 2 --val 1 --test 1
```

This first run produces 4 scenes (16 PNG images at 224 × 224). To generate a single example:

```bash
blender --background --python-exit-code 1 --python generate_data/generate_dataset.py -- --train 1 --val 0 --test 0 --output dataset_preview
```

You can also run `.venv/bin/python generate_data/generate_dataset.py` if `bpy` is installed in that environment. Cycles uses the CPU to run without a graphical interface or GPU. Rendering time depends on your hardware; `render.samples` controls quality. `render.max_bounces` limits light bounces (2 by default; 0 for direct lighting only).

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

Scene numbers are local to each split. A nonempty output directory is rejected to prevent overwriting or mixing datasets. Each scene is moved to its final directory only after all four images and its JSON file have been written. Completed scenes remain available if generation fails; choose a new output directory to rerun generation.

## Configuration

`generate_data/generation_config.json` contains scene counts, the seed, and rendering, object, and camera settings. `--config path.json` loads another configuration; `--train`, `--val`, `--test`, `--seed`, and `--output` allow quick experiments. Default counts: 6 training scenes, 2 validation scenes, and 2 test scenes.

The four cameras point toward the center of the scene. Cubes have uniformly sampled rotations in SO(3) and independent scaling along each axis. They rest on a ground plane, with disjoint enclosing disks to prevent intersections. This placement method is conservative: for dense scenes, increase `position_extent` or adapt the placement algorithm. Objects can still occlude one another in an image.

To use different views for each split, add settings such as:

```json
"split_overrides": {
  "val": {"cameras": {"azimuths_deg": [65, 155, 245, 335]}},
  "test": {"cameras": {"azimuths_deg": [85, 175, 265, 355], "height": 6.0}}
}
```

Overrides are merged recursively. Each scene's seed depends on its split and scene number: increasing scene counts preserves previously generated scenes when using the same configuration and Blender version. Identical results across Blender versions are not guaranteed.

To add a primitive, extend `OBJECT_FACTORIES` and `CLASS_IDS`, then update `objects.types`. Each factory creates an object centered at the origin with a valid bounding box; placement, materials, cameras, and export remain shared. This initial version uses exactly four cameras.

## Ground truth

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
