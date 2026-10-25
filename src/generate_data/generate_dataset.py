"""Generate multi-view scenes with Blender (CLI usage in the adjacent README)."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


CLASS_IDS = {"cube": 0, "sphere": 1, "cylinder": 2}


def create_cube():
    # Base cube edges have length 2: dimensions = 2 * local scale.
    bpy.ops.mesh.primitive_cube_add(size=2)
    return bpy.context.object


def create_sphere():
    # Unit radius; independent XYZ scaling also produces ellipsoids.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1)
    obj = bpy.context.object
    for face in obj.data.polygons:
        face.use_smooth = True
    return obj


def create_cylinder():
    # Unit radius and height 2, aligned with local Z before rotation.
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=1, depth=2)
    obj = bpy.context.object
    for face in obj.data.polygons:
        # Smooth the sides, keeping the two caps flat.
        face.use_smooth = len(face.vertices) == 4
    return obj


OBJECT_FACTORIES = {"cube": create_cube, "sphere": create_sphere,
                    "cylinder": create_cylinder}


def merge_config(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def scene_seed(seed, split, index):
    payload = f"{seed}:{split}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31)


def matrix_rows(matrix):
    return [list(row) for row in matrix]


def material(name, color):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1.0)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = 0.65
    return result


def reset_scene(config, seed):
    # Also release meshes/materials between scenes during long runs.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = config["render"]["samples"]
    scene.cycles.max_bounces = config["render"].get("max_bounces", 2)
    scene.cycles.seed = seed
    scene.render.resolution_x, scene.render.resolution_y = config["render"]["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.35, 0.35, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    bpy.ops.mesh.primitive_plane_add(size=200)
    bpy.context.object.name = "Ground"
    bpy.context.object.data.materials.append(material("Ground", (0.45, 0.45, 0.45)))
    bpy.ops.object.light_add(type="AREA", location=(1, -3, 8))
    bpy.context.object.data.energy = 1200
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 6
    return scene


def sample_rotation(rng):
    # Uniform rotations on SO(3), stored by Blender as quaternion (w, x, y, z).
    u1, u2, u3 = (rng.random() for _ in range(3))
    return Quaternion((math.sqrt(u1) * math.cos(2 * math.pi * u3),
                       math.sqrt(1-u1) * math.sin(2 * math.pi * u2),
                       math.sqrt(1-u1) * math.cos(2 * math.pi * u2),
                       math.sqrt(u1) * math.sin(2 * math.pi * u3)))


def object_rotation(spec, rng):
    mode = spec.get("rotation_mode", "uniform")
    if mode == "uniform":
        return sample_rotation(rng)
    if mode == "none":
        return Quaternion((1, 0, 0, 0))
    angles = spec.get("rotation_euler_deg", [0, 0, 0])
    if mode == "euler_ranges":
        angles = [rng.uniform(*bounds) for bounds in spec["rotation_euler_ranges_deg"]]
    return Euler(tuple(math.radians(a) for a in angles), "XYZ").to_quaternion()


def create_objects(config, rng):
    objects, footprints = [], []
    spec = config["objects"]
    count = rng.randint(*spec["count_range"]) if "count_range" in spec else spec["count"]
    for index in range(count):
        kind = rng.choice(spec["types"])
        obj = OBJECT_FACTORIES[kind]()
        obj.name = f"{kind}_{index:03d}"
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = object_rotation(spec, rng)
        scale_ranges = spec.get("scale_ranges_by_type", {}).get(kind)
        if scale_ranges is not None:
            obj.scale = tuple(rng.uniform(*interval) for interval in scale_ranges)
        else:
            obj.scale = tuple(spec["scale"]) if "scale" in spec else tuple(rng.uniform(*spec["scale_range"]) for _ in range(3))
        bpy.context.view_layer.update()
        local_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        radius = max(math.hypot(v.x, v.y) for v in local_corners)
        # Bounding-box disks conservatively prevent intersections. Use actual
        # mesh vertices for ground contact of curved, rotated primitives.
        min_z = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
        footprints.append((radius, -min_z))
        color = list(spec["color_rgb"]) if "color_rgb" in spec else [rng.uniform(0.15, 0.85) for _ in range(3)]
        obj.data.materials.append(material(obj.name, color))
        if "face_colors_rgb" in spec:
            obj.data.materials.clear()
            for face, face_color in zip(obj.data.polygons, spec["face_colors_rgb"]):
                face.material_index = len(obj.data.materials)
                obj.data.materials.append(material(f"{obj.name}_face_{face.index}", face_color))
        objects.append((kind, obj, color))
    # Restart the entire layout when an early position blocks later objects.
    # Keep geometry fixed so retries do not bias the sampled sizes/rotations.
    layout_attempts = spec.get("layout_attempts", 100)
    for _ in range(layout_attempts):
        occupied = []
        for radius, _ in footprints:
            for _ in range(spec["placement_attempts"]):
                x, y = (rng.uniform(-spec["position_extent"], spec["position_extent"]) for _ in range(2))
                if all(math.hypot(x-px, y-py) > radius+pr+spec["min_gap"] for px, py, pr in occupied):
                    occupied.append((x, y, radius))
                    break
            else:
                break
        if len(occupied) == len(objects):
            for (_, obj, _), (x, y, _), (_, height) in zip(objects, occupied, footprints):
                obj.location = (x, y, height)
            break
    else:
        raise ValueError(
            f"Placement impossible après {layout_attempts} dispositions "
            f"({spec['placement_attempts']} essais par objet) : "
            f"count={count}, position_extent={spec['position_extent']}, "
            f"scale_range={spec['scale_range']}, min_gap={spec['min_gap']}. "
            "Agrandir position_extent ou réduire le nombre/la taille des objets."
        )
    bpy.context.view_layer.update()
    return objects


def create_cameras(spec, rng=None):
    rng = rng if rng is not None else random.Random()
    def offset(key):
        bounds = spec.get(key, [0, 0])
        return rng.uniform(*bounds) if bounds != [0, 0] else 0.0

    cameras = []
    for index, azimuth in enumerate(spec["azimuths_deg"]):
        angle = math.radians(azimuth + offset("azimuth_jitter_deg"))
        radius = spec["radius"] + offset("radius_jitter")
        height = spec["height"] + offset("height_jitter")
        bpy.ops.object.camera_add(location=(radius*math.cos(angle),
                                           radius*math.sin(angle), height))
        camera = bpy.context.object
        camera.name = f"camera_{index}"
        camera.rotation_euler = (Vector(spec["target"])-camera.location).to_track_quat("-Z", "Y").to_euler()
        # Local XYZ offsets relative to the look-at orientation; Z is optical roll.
        angles = [offset(key) for key in
                  ("pitch_jitter_deg", "yaw_jitter_deg", "roll_jitter_deg")]
        if any(angles):
            camera.rotation_euler = (
                camera.rotation_euler.to_matrix()
                @ Euler(tuple(math.radians(a) for a in angles), "XYZ").to_matrix()
            ).to_euler()
        camera.data.type = "PERSP"
        camera.data.lens = spec["lens_mm"]
        camera.data.sensor_width = spec["sensor_width_mm"]
        camera.data.sensor_fit = "HORIZONTAL"
        cameras.append(camera)
    bpy.context.view_layer.update()
    return cameras


def ground_truth(scene, objects, cameras, seed, split, index):
    world_to_reference = cameras[0].matrix_world.inverted()
    width, height = scene.render.resolution_x, scene.render.resolution_y
    records = []
    for kind, obj, color in objects:
        transform = world_to_reference @ obj.matrix_world
        position, rotation, scale = transform.decompose()
        rotation_matrix = rotation.to_matrix()
        records.append({"id": obj.name, "type": kind, "class_id": CLASS_IDS[kind],
                        "position": list(position), "scale": list(scale),
                        "dimensions": list(obj.dimensions), "color_rgb": color,
                        "rotation_quaternion_wxyz": list(rotation),
                        "rotation_matrix": matrix_rows(rotation_matrix),
                        "rotation_6d": list(rotation_matrix.col[0]) + list(rotation_matrix.col[1]),
                        "object_to_reference": matrix_rows(transform),
                        "object_to_world": matrix_rows(obj.matrix_world)})
        if len(obj.data.materials) > 1:
            records[-1]["face_colors"] = [
                {"polygon_index": face.index, "normal_local": list(face.normal),
                 "color_rgb": list(obj.data.materials[face.material_index].diffuse_color[:3])}
                for face in obj.data.polygons
            ]
    views = []
    for i, camera in enumerate(cameras):
        focal = width * camera.data.lens / camera.data.sensor_width
        # OpenCV camera coordinates: x right, y down, z forward.
        blender_to_cv = Matrix.Diagonal((1, -1, -1, 1))
        views.append({"id": camera.name, "image": f"view_{i}.png",
                      "camera_to_world": matrix_rows(camera.matrix_world),
                      "camera_to_reference": matrix_rows(world_to_reference @ camera.matrix_world),
                      "world_to_camera_cv": matrix_rows(blender_to_cv @ camera.matrix_world.inverted()),
                      "intrinsics_cv": [[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]],
                      "lens_mm": camera.data.lens, "sensor_width_mm": camera.data.sensor_width})
    return {"schema_version": 1, "scene_id": f"scene_{index:06d}", "split": split,
            "seed": seed, "blender_version": bpy.app.version_string,
            "resolution": [width, height], "reference_camera": "camera_0",
            "coordinate_convention": "Blender camera: X right, Y up, -Z forward; units: meters",
            "world_to_reference": matrix_rows(world_to_reference),
            "objects": records, "cameras": views}


def validate_config(config):
    spec = config["objects"]
    counts = spec.get("count_range", [spec.get("count"), spec.get("count")])
    if (not isinstance(counts, (list, tuple)) or len(counts) != 2
            or any(type(v) is not int or v < 1 for v in counts) or counts[0] > counts[1]):
        raise ValueError("objects.count ou count_range doit définir des entiers strictement positifs ordonnés.")
    if not spec["types"] or set(spec["types"])-OBJECT_FACTORIES.keys():
        raise ValueError("Nombre ou types d'objets invalides.")
    if len(spec["scale_range"]) != 2 or not 0 < spec["scale_range"][0] <= spec["scale_range"][1]:
        raise ValueError("scale_range doit contenir deux valeurs positives croissantes.")
    if spec["position_extent"] < 0 or spec["min_gap"] < 0 or spec["placement_attempts"] < 1:
        raise ValueError("Paramètres de placement invalides.")
    if spec["position_extent"] == 0 and counts[1] > 1:
        raise ValueError("position_extent=0 ne permet de placer qu'un seul objet.")
    if not isinstance(spec.get("layout_attempts", 100), int) or spec.get("layout_attempts", 100) < 1:
        raise ValueError("layout_attempts doit être un entier strictement positif.")
    mode = spec.get("rotation_mode", "uniform")
    if mode not in {"uniform", "none", "fixed_euler", "euler_ranges"}:
        raise ValueError("objects.rotation_mode invalide.")
    def vector(value, name, positive=False):
        if (len(value) != 3 or any(not isinstance(v, (int, float)) or
                not math.isfinite(v) or (positive and v <= 0) for v in value)):
            raise ValueError(f"{name} doit contenir trois valeurs finies valides.")
    def bounds(value, name):
        if (len(value) != 2 or any(not isinstance(v, (int, float)) or
                not math.isfinite(v) for v in value) or value[0] > value[1]):
            raise ValueError(f"{name} doit être un intervalle fini ordonné.")
    if "scale" in spec:
        vector(spec["scale"], "objects.scale", positive=True)
    per_type = spec.get("scale_ranges_by_type", {})
    if not isinstance(per_type, dict) or set(per_type) - OBJECT_FACTORIES.keys():
        raise ValueError("objects.scale_ranges_by_type doit associer des types connus à des intervalles XYZ.")
    for kind, ranges in per_type.items():
        if not isinstance(ranges, (list, tuple)) or len(ranges) != 3:
            raise ValueError(f"scale_ranges_by_type.{kind} attend trois intervalles XYZ.")
        for interval in ranges:
            bounds(interval, f"scale_ranges_by_type.{kind}")
            if interval[0] <= 0:
                raise ValueError("Les scales par type doivent être strictement positifs.")
    if "color_rgb" in spec:
        vector(spec["color_rgb"], "objects.color_rgb")
        if any(v < 0 or v > 1 for v in spec["color_rgb"]):
            raise ValueError("objects.color_rgb doit être dans [0, 1].")
    if "face_colors_rgb" in spec:
        colors = spec["face_colors_rgb"]
        if set(spec["types"]) != {"cube"} or len(colors) != 6:
            raise ValueError("objects.face_colors_rgb attend six couleurs pour des cubes uniquement.")
        for color in colors:
            vector(color, "objects.face_colors_rgb")
            if any(v < 0 or v > 1 for v in color):
                raise ValueError("objects.face_colors_rgb doit être dans [0, 1].")
        if len({tuple(color) for color in colors}) != 6:
            raise ValueError("objects.face_colors_rgb attend six couleurs différentes.")
    vector(spec.get("rotation_euler_deg", [0, 0, 0]), "rotation_euler_deg")
    if mode == "euler_ranges":
        ranges = spec.get("rotation_euler_ranges_deg", [])
        if len(ranges) != 3:
            raise ValueError("rotation_euler_ranges_deg attend trois intervalles XYZ.")
        for interval in ranges:
            bounds(interval, "rotation_euler_ranges_deg")
    camera_spec = config["cameras"]
    for key in ("azimuth_jitter_deg", "radius_jitter", "height_jitter",
                "pitch_jitter_deg", "yaw_jitter_deg", "roll_jitter_deg"):
        bounds(camera_spec.get(key, [0, 0]), key)
    if camera_spec["radius"] + camera_spec.get("radius_jitter", [0, 0])[0] <= 0:
        raise ValueError("Le rayon minimal des caméras doit être positif.")
    azimuths = camera_spec["azimuths_deg"]
    if not azimuths or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in azimuths):
        raise ValueError("cameras.azimuths_deg attend au moins un angle fini.")
    if any(config["cameras"][key] <= 0 for key in ("radius", "lens_mm", "sensor_width_mm")):
        raise ValueError("Paramètres de caméra invalides.")
    resolution = config["render"]["resolution"]
    if len(resolution) != 2 or any(not isinstance(v, int) or v < 1 for v in resolution):
        raise ValueError("Résolution invalide.")
    if config["render"]["samples"] < 1:
        raise ValueError("samples doit être positif.")
    max_bounces = config["render"].get("max_bounces", 2)
    if not isinstance(max_bounces, int) or max_bounces < 0:
        raise ValueError("max_bounces doit être un entier positif ou nul.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("generation_config.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--train", type=int)
    parser.add_argument("--val", type=int)
    parser.add_argument("--test", type=int)
    parser.add_argument("--seed", type=int)
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    for split in ("train", "val", "test"):
        if getattr(args, split) is not None:
            config["splits"][split] = getattr(args, split)
    if set(config["splits"]) != {"train", "val", "test"}:
        parser.error("Les splits doivent être train, val et test.")
    for count in config["splits"].values():
        if not isinstance(count, int) or count < 0:
            parser.error("Les nombres de scènes doivent être des entiers positifs ou nuls.")
    effective = {split: merge_config(config, config.get("split_overrides", {}).get(split, {}))
                 for split in config["splits"]}
    for settings in effective.values():
        validate_config(settings)
    # Never mix a new run with existing data or silently overwrite scenes.
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Le dossier {args.output} n'est pas vide. Choisir un autre --output.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "generation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    for split, count in config["splits"].items():
        split_dir = args.output / split
        split_dir.mkdir(exist_ok=True)
        for index in range(count):
            seed = scene_seed(config["seed"], split, index)
            settings = effective[split]
            scene = reset_scene(settings, seed)
            objects = create_objects(settings, random.Random(seed))
            cameras = create_cameras(settings["cameras"], random.Random(seed ^ 0xCAFE))
            destination = split_dir / f"scene_{index:06d}"
            temporary = Path(tempfile.mkdtemp(prefix=".scene_", dir=split_dir))
            try:
                for view, camera in enumerate(cameras):
                    scene.camera = camera
                    scene.render.filepath = str((temporary / f"view_{view}.png").resolve())
                    bpy.ops.render.render(write_still=True)
                truth = ground_truth(scene, objects, cameras, seed, split, index)
                (temporary / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
                temporary.rename(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            print(f"Generated {destination}", flush=True)


if __name__ == "__main__":
    main()
