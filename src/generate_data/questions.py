"""Templated multi-view questions computed from a scene's ground truth.

Each question has a family (topic) so results can be broken down per skill:

    A  viewpoint-relative relations   (left/right, front/behind as seen by a camera)
    B  visibility and occlusion       (what a camera can or cannot see, and why)
    D  cross-view correspondence      (relating what different cameras see)
    F  perspective taking             (relations from a viewpoint that is not a camera)
    G  metric comparisons             (3D distances, angles between cameras)

Answers are computed from `ground_truth.json`; relations closer than a safety
margin are considered ambiguous and never asked. Families B and D, plus a few
other templates, need the per-view `visibility` written by generate_dataset.py
(schema_version 2). Pure standard library: runs inside Blender or standalone.

Standalone use (regenerates questions.json for an existing dataset):

    python src/generate_data/questions.py --dataset data/occlusion_questions
    python src/generate_data/questions.py --dataset data/occlusion_questions --stats
"""
import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random


SCHEMA_VERSION = 1

FAMILIES = {
    "A": "viewpoint-relative relations",
    "B": "visibility and occlusion",
    "D": "cross-view correspondence",
    "F": "perspective taking",
    "G": "metric comparisons",
}

DEFAULT_SETTINGS = {
    # Number of questions sampled per scene (fewer if not enough are valid).
    "per_scene": 16,
    "families": ["A", "B", "D", "F", "G"],
    "max_per_template": 2,
    # Answer balancing across a split: once a template has `balance_warmup`
    # questions, an answer whose share exceeds balance_slack / (number of
    # answers seen) is not asked again until the others catch up.
    "balance_warmup": 30,
    "balance_slack": 1.2,
    # Safety margins: relations inside these margins are ambiguous.
    "side_margin": 0.05,           # horizontal gap, fraction of image width
    "depth_ratio": 1.1,            # camera-distance ratio
    "distance_ratio": 1.15,        # object-distance ratio
    "angle_margin_deg": 15.0,      # away from the left/right or front/back boundary
    "angle_bin_deg": 45.0,
    "angle_bin_tolerance_deg": 12.0,
    # Visibility states, from ray-cast pixel counts (fractions of image area).
    "visible_min_pixels": 0.004,   # ~200 px at 224x224
    "visible_min_fraction": 0.2,   # of the unoccluded in-frame silhouette
    "hidden_max_pixels": 0.0005,   # ~25 px at 224x224
    "partial_range": [0.1, 0.75],  # visible fraction counted as "partly hidden"
    "unoccluded_min_fraction": 0.97,
    "occluded_min_fraction": 0.25,
    "occluder_share": 0.8,
    "most_visible_ratio": 1.25,
}

PLURALS = {"cube": "cubes", "sphere": "spheres", "cylinder": "cylinders"}


def validate_settings(settings):
    unknown = set(settings) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"questions: paramètres inconnus {sorted(unknown)}.")
    merged = {**DEFAULT_SETTINGS, **settings}
    if set(merged["families"]) - set(FAMILIES) or not merged["families"]:
        raise ValueError(f"questions.families doit être un sous-ensemble de {sorted(FAMILIES)}.")
    if not isinstance(merged["per_scene"], int) or merged["per_scene"] < 1:
        raise ValueError("questions.per_scene doit être un entier strictement positif.")
    if not isinstance(merged["max_per_template"], int) or merged["max_per_template"] < 1:
        raise ValueError("questions.max_per_template doit être un entier strictement positif.")
    low, high = merged["partial_range"]
    if not 0 <= low < high < merged["unoccluded_min_fraction"] <= 1:
        raise ValueError("questions.partial_range doit être croissant et sous unoccluded_min_fraction.")
    if merged["hidden_max_pixels"] >= merged["visible_min_pixels"]:
        raise ValueError("questions.hidden_max_pixels doit être inférieur à visible_min_pixels.")
    return merged


# --------------------------------------------------------------------------
# Scene facts


def translation(rows):
    return [rows[0][3], rows[1][3], rows[2][3]]


def transform(rows, point):
    return [sum(rows[r][c] * point[c] for c in range(3)) + rows[r][3] for r in range(3)]


def distance(a, b):
    return math.dist(a, b)


def capitalize(text):
    return text[0].upper() + text[1:]


class SceneFacts:
    """Geometry and visibility facts derived once per scene."""

    def __init__(self, truth, settings):
        self.s = settings
        self.width, self.height = truth["resolution"]
        self.area = self.width * self.height
        objects = truth["objects"]
        self.n = len(objects)
        self.ids = [o["id"] for o in objects]
        self.types = [o["type"] for o in objects]
        self.colors = [o.get("color_name") for o in objects]
        # World coordinates: Z up, objects resting on the ground plane.
        self.positions = [translation(o["object_to_world"]) for o in objects]
        cameras = truth["cameras"]
        self.k = len(cameras)
        self.camera_positions = [translation(c["camera_to_world"]) for c in cameras]
        # Projected centers (u, v) in pixels and depth along the optical axis.
        self.projections = []
        for camera in cameras:
            intrinsics = camera["intrinsics_cv"]
            row = []
            for position in self.positions:
                x, y, z = transform(camera["world_to_camera_cv"], position)
                if z <= 1e-6:
                    row.append(None)
                else:
                    row.append((intrinsics[0][0] * x / z + intrinsics[0][2],
                                intrinsics[1][1] * y / z + intrinsics[1][2], z))
            self.projections.append(row)
        self.has_visibility = all("visibility" in c for c in cameras)
        if self.has_visibility:
            self.visibility = [{r["object"]: r for r in c["visibility"]} for c in cameras]
        self.refs = self.referring_expressions()
        self.referable = [i for i in range(self.n) if self.refs[i] is not None]

    def referring_expressions(self):
        """Shortest unambiguous name of each object, or None.

        Tries "the cube", then "the red cube" (if colors are named). Objects
        have a fixed size per type, so two red cubes are never referable.
        """
        refs = [None] * self.n
        by_type, by_color_type = defaultdict(list), defaultdict(list)
        for i, kind in enumerate(self.types):
            by_type[kind].append(i)
            if self.colors[i] is not None:
                by_color_type[(self.colors[i], kind)].append(i)
        for i, kind in enumerate(self.types):
            if len(by_type[kind]) == 1:
                refs[i] = f"the {kind}"
            elif self.colors[i] is not None and len(by_color_type[(self.colors[i], kind)]) == 1:
                refs[i] = f"the {self.colors[i]} {kind}"
        return refs

    def record(self, k, i):
        return self.visibility[k][self.ids[i]]

    def state(self, k, i):
        """'visible', 'hidden' or None (ambiguous) for object i in camera k."""
        record = self.record(k, i)
        pixels = record["visible_pixels"] / self.area
        if pixels <= self.s["hidden_max_pixels"]:
            return "hidden"
        if (pixels >= self.s["visible_min_pixels"]
                and record["visible_fraction"] >= self.s["visible_min_fraction"]):
            return "visible"
        return None

    def states(self, k):
        """All states of camera k, or None if any of them is ambiguous."""
        result = [self.state(k, i) for i in range(self.n)]
        return None if None in result else result

    def in_frame(self, k, i):
        p = self.projections[k][i]
        return p is not None and 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def seen(self, k, i):
        """Clearly visible in camera k (projected center in frame without visibility data)."""
        if self.has_visibility:
            return self.state(k, i) == "visible"
        return self.in_frame(k, i)

    def extreme(self, k, side):
        """Index of the clearly leftmost/rightmost visible object in camera k, or None."""
        states = self.states(k)
        if states is None:
            return None
        visible = [i for i in range(self.n) if states[i] == "visible"]
        if len(visible) < 2:
            return None
        order = sorted(visible, key=lambda i: self.projections[k][i][0], reverse=side == "right")
        gap = abs(self.projections[k][order[0]][0] - self.projections[k][order[1]][0])
        return order[0] if gap >= self.s["side_margin"] * self.width else None

    def clear_extreme(self, values, mode, ratio):
        """Index of the clear minimum/maximum of `values` (dict index -> value), or None."""
        order = sorted(values, key=values.get, reverse=mode == "max")
        if len(order) < 2:
            return order[0] if order else None
        best, second = values[order[0]], values[order[1]]
        if mode == "min" and second >= ratio * best:
            return order[0]
        if mode == "max" and best >= ratio * second:
            return order[0]
        return None

    def ground_angle(self, anchor, target, subject):
        """Signed angle (degrees, counterclockwise from above) from anchor→target to anchor→subject."""
        fx, fy = target[0] - anchor[0], target[1] - anchor[1]
        dx, dy = subject[0] - anchor[0], subject[1] - anchor[1]
        return math.degrees(math.atan2(fx * dy - fy * dx, fx * dx + fy * dy))


# --------------------------------------------------------------------------
# Templates


@dataclass
class Candidate:
    variants: list
    answer: str
    params: dict
    key: tuple = field(default=())


@dataclass
class Template:
    family: str
    function: object
    needs_visibility: bool
    # True when, for some answers, information from another camera is logically
    # required. A heuristic tag; a single-view ablation measures the real need.
    requires_multiview: bool


TEMPLATES = {}


def template(name, family, needs_visibility=False, requires_multiview=False):
    def register(function):
        TEMPLATES[name] = Template(family, function, needs_visibility, requires_multiview)
        return function
    return register


def ordered_pairs(indices):
    return [(i, j) for i in indices for j in indices if i != j]


# A: viewpoint-relative relations --------------------------------------------

@template("A_side", "A")
def a_side(sc):
    out = []
    for k in range(sc.k):
        for i, j in ordered_pairs(sc.referable):
            if not (sc.seen(k, i) and sc.seen(k, j)):
                continue
            du = sc.projections[k][i][0] - sc.projections[k][j][0]
            if abs(du) < sc.s["side_margin"] * sc.width:
                continue
            x, y = sc.refs[i], sc.refs[j]
            out.append(Candidate(
                [f"From camera {k}, is {x} to the left or to the right of {y}?",
                 f"In the image from camera {k}, is {x} left or right of {y}?"],
                "left" if du < 0 else "right",
                {"camera": k, "objects": [sc.ids[i], sc.ids[j]]},
                ("A_side", k, frozenset((i, j)))))
    return out


@template("A_depth", "A")
def a_depth(sc):
    out = []
    for k in range(sc.k):
        camera = sc.camera_positions[k]
        for i, j in ordered_pairs(sc.referable):
            if not (sc.seen(k, i) and sc.seen(k, j)):
                continue
            di, dj = distance(sc.positions[i], camera), distance(sc.positions[j], camera)
            if max(di, dj) < sc.s["depth_ratio"] * min(di, dj):
                continue
            x, y = sc.refs[i], sc.refs[j]
            out.append(Candidate(
                [f"From camera {k}, is {x} in front of or behind {y}?",
                 f"Seen from camera {k}, is {x} in front of {y} or behind it?"],
                "in front" if di < dj else "behind",
                {"camera": k, "objects": [sc.ids[i], sc.ids[j]]},
                ("A_depth", k, frozenset((i, j)))))
    return out


@template("A_closest", "A")
def a_closest(sc):
    out = []
    for k in range(sc.k):
        distances = {i: distance(sc.positions[i], sc.camera_positions[k]) for i in range(sc.n)}
        for mode, text in (("min", "closest to"), ("max", "farthest from")):
            best = sc.clear_extreme(distances, mode, sc.s["depth_ratio"])
            if best is not None:
                out.append(Candidate(
                    [f"Which object is {text} camera {k}?"],
                    sc.types[best], {"camera": k, "mode": mode, "object": sc.ids[best]}))
    return out


@template("A_extreme", "A", needs_visibility=True)
def a_extreme(sc):
    out = []
    for k in range(sc.k):
        for side in ("left", "right"):
            best = sc.extreme(k, side)
            if best is not None:
                out.append(Candidate(
                    [f"Which object appears {side}most in the image from camera {k}?",
                     f"In the view from camera {k}, which object is the {side}most?"],
                    sc.types[best], {"camera": k, "side": side, "object": sc.ids[best]}))
    return out


# B: visibility and occlusion --------------------------------------------------

@template("B_visible", "B", needs_visibility=True, requires_multiview=True)
def b_visible(sc):
    out = []
    for k in range(sc.k):
        for i in sc.referable:
            state = sc.state(k, i)
            if state is None:
                continue
            x = sc.refs[i]
            out.append(Candidate(
                [f"Can {x} be seen from camera {k}?", f"Is {x} visible from camera {k}?"],
                "yes" if state == "visible" else "no",
                {"camera": k, "object": sc.ids[i]}))
    return out


@template("B_count", "B", needs_visibility=True)
def b_count(sc):
    out = []
    for k in range(sc.k):
        states = sc.states(k)
        if states is not None:
            out.append(Candidate(
                [f"How many objects can be seen from camera {k}?",
                 f"How many objects are visible in the image from camera {k}?"],
                str(states.count("visible")), {"camera": k}))
    return out


@template("B_hidden", "B", needs_visibility=True, requires_multiview=True)
def b_hidden(sc):
    out = []
    for k in range(sc.k):
        states = sc.states(k)
        if states is None or states.count("hidden") != 1:
            continue
        hidden = states.index("hidden")
        out.append(Candidate(
            [f"Which object cannot be seen from camera {k}?",
             f"Which object is missing from the image of camera {k}?"],
            sc.types[hidden], {"camera": k, "object": sc.ids[hidden]}))
    return out


@template("B_partial", "B", needs_visibility=True)
def b_partial(sc):
    out = []
    low, high = sc.s["partial_range"]
    for k in range(sc.k):
        for i in sc.referable:
            if sc.state(k, i) != "visible":
                continue
            fraction = sc.record(k, i)["visible_fraction"]
            if low <= fraction <= high:
                answer = "yes"
            elif fraction >= sc.s["unoccluded_min_fraction"]:
                answer = "no"
            else:
                continue
            x = sc.refs[i]
            out.append(Candidate(
                [f"From camera {k}, is {x} partly hidden by another object?",
                 f"Seen from camera {k}, is part of {x} hidden behind another object?"],
                answer, {"camera": k, "object": sc.ids[i]}))
    return out


@template("B_occluder", "B", needs_visibility=True, requires_multiview=True)
def b_occluder(sc):
    out = []
    for k in range(sc.k):
        for i in sc.referable:
            record = sc.record(k, i)
            unoccluded = record["unoccluded_pixels"]
            if unoccluded < sc.s["visible_min_pixels"] * sc.area:
                continue
            occluded = unoccluded - record["visible_pixels"]
            if occluded < sc.s["occluded_min_fraction"] * unoccluded:
                continue
            name, pixels = max(record["occluders"].items(), key=lambda item: item[1])
            if pixels < sc.s["occluder_share"] * occluded:
                continue
            occluder = sc.ids.index(name)
            x = sc.refs[i]
            out.append(Candidate(
                [f"From camera {k}, which object hides {x}?",
                 f"Which object blocks the view of {x} from camera {k}?"],
                sc.types[occluder], {"camera": k, "object": sc.ids[i], "occluder": name}))
    return out


@template("B_num_cameras", "B", needs_visibility=True, requires_multiview=True)
def b_num_cameras(sc):
    out = []
    for i in sc.referable:
        states = [sc.state(k, i) for k in range(sc.k)]
        if None in states:
            continue
        x = sc.refs[i]
        out.append(Candidate(
            [f"From how many cameras can {x} be seen?", f"How many cameras can see {x}?"],
            str(states.count("visible")), {"object": sc.ids[i]}))
    return out


# D: cross-view correspondence -------------------------------------------------

@template("D_most_visible", "D", needs_visibility=True, requires_multiview=True)
def d_most_visible(sc):
    out = []
    for i in sc.referable:
        pixels = {k: sc.record(k, i)["visible_pixels"] for k in range(sc.k)}
        best = sc.clear_extreme(pixels, "max", sc.s["most_visible_ratio"])
        if best is None or sc.state(best, i) != "visible":
            continue
        x = sc.refs[i]
        out.append(Candidate(
            [f"From which camera is {x} most visible?",
             f"In which camera's image does {x} take up the most space?"],
            f"camera {best}", {"object": sc.ids[i], "camera": best}))
    return out


@template("D_shared", "D", needs_visibility=True, requires_multiview=True)
def d_shared(sc):
    out = []
    for j in range(sc.k):
        for k in range(j + 1, sc.k):
            sj, sk = sc.states(j), sc.states(k)
            if sj is None or sk is None:
                continue
            count = sum(a == b == "visible" for a, b in zip(sj, sk))
            out.append(Candidate(
                [f"How many objects can be seen from both camera {j} and camera {k}?"],
                str(count), {"cameras": [j, k]}))
    return out


@template("D_only", "D", needs_visibility=True, requires_multiview=True)
def d_only(sc):
    out = []
    for j, k in ordered_pairs(range(sc.k)):
        sj, sk = sc.states(j), sc.states(k)
        if sj is None or sk is None:
            continue
        only = [i for i in range(sc.n) if sj[i] == "visible" and sk[i] == "hidden"]
        if len(only) != 1:
            continue
        out.append(Candidate(
            [f"Which object can be seen from camera {j} but not from camera {k}?"],
            sc.types[only[0]], {"cameras": [j, k], "object": sc.ids[only[0]]}))
    return out


@template("D_extreme_seen", "D", needs_visibility=True, requires_multiview=True)
def d_extreme_seen(sc):
    out = []
    for j, k in ordered_pairs(range(sc.k)):
        for side in ("left", "right"):
            best = sc.extreme(j, side)
            if best is None:
                continue
            state = sc.state(k, best)
            if state is None:
                continue
            out.append(Candidate(
                [f"Can the {side}most object in the image from camera {j} be seen from camera {k}?"],
                "yes" if state == "visible" else "no",
                {"cameras": [j, k], "side": side, "object": sc.ids[best]}))
    return out


@template("D_closest_camera", "D", requires_multiview=True)
def d_closest_camera(sc):
    out = []
    for i in sc.referable:
        distances = {k: distance(sc.positions[i], sc.camera_positions[k]) for k in range(sc.k)}
        best = sc.clear_extreme(distances, "min", sc.s["depth_ratio"])
        if best is None:
            continue
        x = sc.refs[i]
        out.append(Candidate(
            [f"Which camera is closest to {x}?"],
            f"camera {best}", {"object": sc.ids[i], "camera": best}))
    return out


# F: perspective taking --------------------------------------------------------

def ego_targets(sc, anchor):
    """Things one can face from an object: other objects and the cameras."""
    targets = [(sc.refs[t], sc.positions[t], sc.ids[t]) for t in sc.referable if t != anchor]
    targets += [(f"camera {k}", sc.camera_positions[k], f"camera_{k}") for k in range(sc.k)]
    return targets


def ego_candidates(sc, name, classify, variants):
    out = []
    for a in sc.referable:
        for target_ref, target_pos, target_id in ego_targets(sc, a):
            for x in sc.referable:
                if x == a or sc.ids[x] == target_id:
                    continue
                angle = sc.ground_angle(sc.positions[a], target_pos, sc.positions[x])
                answer = classify(angle, sc.s["angle_margin_deg"])
                if answer is None:
                    continue
                words = {"a": sc.refs[a], "t": target_ref, "x": sc.refs[x]}
                out.append(Candidate(
                    [v.format(**words) for v in variants], answer,
                    {"anchor": sc.ids[a], "facing": target_id, "object": sc.ids[x]},
                    (name, a, target_id, x)))
    return out


def classify_side(angle, margin):
    if margin <= abs(angle) <= 180 - margin:
        return "left" if angle > 0 else "right"
    return None


def classify_front(angle, margin):
    if abs(angle) <= 90 - margin:
        return "in front"
    if abs(angle) >= 90 + margin:
        return "behind"
    return None


@template("F_ego_side", "F")
def f_ego_side(sc):
    return ego_candidates(sc, "F_ego_side", classify_side, [
        "If you stood at {a} facing {t}, would {x} be on your left or on your right?",
        "Imagine standing at {a} and looking toward {t}. Is {x} on your left or on your right?"])


@template("F_ego_front", "F")
def f_ego_front(sc):
    return ego_candidates(sc, "F_ego_front", classify_front, [
        "If you stood at {a} facing {t}, would {x} be in front of you or behind you?",
        "Imagine standing at {a} and looking toward {t}. Is {x} in front of you or behind you?"])


@template("F_hidden_side", "F", needs_visibility=True, requires_multiview=True)
def f_hidden_side(sc):
    out = []
    for k in range(sc.k):
        for i, j in ordered_pairs(sc.referable):
            if sc.state(k, i) != "hidden" or sc.state(k, j) != "visible":
                continue
            if sc.projections[k][i] is None:
                continue
            du = sc.projections[k][i][0] - sc.projections[k][j][0]
            if abs(du) < sc.s["side_margin"] * sc.width:
                continue
            x, y = sc.refs[i], sc.refs[j]
            out.append(Candidate(
                [f"{capitalize(x)} cannot be seen from camera {k}. "
                 f"From that camera, is it to the left or to the right of {y}?"],
                "left" if du < 0 else "right",
                {"camera": k, "objects": [sc.ids[i], sc.ids[j]]}))
    return out


# G: metric comparisons --------------------------------------------------------

@template("G_closer_to", "G")
def g_closer_to(sc):
    out = []
    for x in sc.referable:
        for y, z in ordered_pairs([i for i in sc.referable if i != x]):
            dy, dz = distance(sc.positions[x], sc.positions[y]), distance(sc.positions[x], sc.positions[z])
            if max(dy, dz) < sc.s["distance_ratio"] * min(dy, dz):
                continue
            out.append(Candidate(
                [f"Is {sc.refs[x]} closer to {sc.refs[y]} than to {sc.refs[z]}?"],
                "yes" if dy < dz else "no",
                {"objects": [sc.ids[x], sc.ids[y], sc.ids[z]]},
                ("G_closer_to", x, frozenset((y, z)))))
    return out


@template("G_nearest_object", "G")
def g_nearest_object(sc):
    out = []
    for x in sc.referable:
        distances = {i: distance(sc.positions[x], sc.positions[i]) for i in range(sc.n) if i != x}
        best = sc.clear_extreme(distances, "min", sc.s["distance_ratio"])
        if best is None or len(distances) < 2:
            continue
        out.append(Candidate(
            [f"Which object is closest to {sc.refs[x]}?"],
            sc.types[best], {"object": sc.ids[x], "nearest": sc.ids[best]}))
    return out


@template("G_closest_pair", "G")
def g_closest_pair(sc):
    if sc.n < 3:
        return []
    pairs = {(i, j): distance(sc.positions[i], sc.positions[j])
             for i in range(sc.n) for j in range(i + 1, sc.n)}
    best = sc.clear_extreme(pairs, "min", sc.s["distance_ratio"])
    if best is None:
        return []
    answer = " and ".join(sorted(sc.types[i] for i in best))
    return [Candidate(["Which two objects are closest to each other?"],
                      answer, {"objects": [sc.ids[i] for i in best]})]


@template("G_camera_angle", "G", requires_multiview=True)
def g_camera_angle(sc):
    out = []
    step, tolerance = sc.s["angle_bin_deg"], sc.s["angle_bin_tolerance_deg"]
    for j in range(sc.k):
        for k in range(j + 1, sc.k):
            # Azimuth difference around the vertical axis through the scene center.
            aj = math.atan2(sc.camera_positions[j][1], sc.camera_positions[j][0])
            ak = math.atan2(sc.camera_positions[k][1], sc.camera_positions[k][0])
            angle = abs(math.degrees((aj - ak + math.pi) % (2 * math.pi) - math.pi))
            rounded = step * round(angle / step)
            if abs(angle - rounded) > tolerance:
                continue
            out.append(Candidate(
                [f"Around the center of the scene, what is the angle between camera {j} and camera {k}?"],
                f"{int(rounded)} degrees", {"cameras": [j, k], "angle_deg": angle}))
    return out


# --------------------------------------------------------------------------
# Sampling


class AnswerBalancer:
    """Per-template answer counts shared by all scenes of a split."""

    def __init__(self):
        self.counts = defaultdict(Counter)

    def allowed(self, name, answer, settings):
        counts = self.counts[name]
        total = sum(counts.values())
        if total < settings["balance_warmup"]:
            return True
        answers = len(counts) + (answer not in counts)
        return (counts[answer] + 1) / (total + 1) <= settings["balance_slack"] / answers

    def add(self, name, answer):
        self.counts[name][answer] += 1


def scene_candidates(truth, settings):
    sc = SceneFacts(truth, settings)
    candidates = {}
    for name, spec in TEMPLATES.items():
        if spec.family not in settings["families"]:
            continue
        if spec.needs_visibility and not sc.has_visibility:
            continue
        found = spec.function(sc)
        if found:
            candidates[name] = found
    return candidates


def generate_scene_questions(truth, settings=None, rng=None, balancer=None):
    """Sample a family, then a template, then its least frequent answer."""
    settings = validate_settings(settings or {})
    rng = rng or random.Random(truth["seed"])
    balancer = balancer if balancer is not None else AnswerBalancer()
    pools = scene_candidates(truth, settings)
    per_template, questions = Counter(), []

    def key_of(name, candidate):
        # Symmetric questions (X vs Y, Y vs X) share a key and are asked once.
        return candidate.key or (name, json.dumps(candidate.params, sort_keys=True))

    while len(questions) < settings["per_scene"] and pools:
        family = rng.choice(sorted({TEMPLATES[name].family for name in pools}))
        name = rng.choice(sorted(n for n in pools if TEMPLATES[n].family == family))
        pool = pools[name]
        by_answer = defaultdict(list)
        for candidate in pool:
            if balancer.allowed(name, candidate.answer, settings):
                by_answer[candidate.answer].append(candidate)
        if not by_answer:
            del pools[name]
            continue
        counts = balancer.counts[name]
        least = min(counts[a] for a in by_answer)
        answer = rng.choice(sorted(a for a in by_answer if counts[a] == least))
        chosen = rng.choice(by_answer[answer])
        key = key_of(name, chosen)
        pools[name] = [c for c in pool if key_of(name, c) != key]
        per_template[name] += 1
        if not pools[name] or per_template[name] >= settings["max_per_template"]:
            del pools[name]
        balancer.add(name, chosen.answer)
        spec = TEMPLATES[name]
        questions.append({
            "id": f"{truth['scene_id']}_q{len(questions):02d}",
            "family": spec.family,
            "family_name": FAMILIES[spec.family],
            "template": name,
            "requires_multiview": spec.requires_multiview,
            "question": rng.choice(chosen.variants),
            "answer": chosen.answer,
            "params": chosen.params,
        })
    return {"schema_version": SCHEMA_VERSION, "scene_id": truth["scene_id"],
            "split": truth["split"], "questions": questions}


# --------------------------------------------------------------------------
# Standalone generation and statistics


def scene_dirs(dataset, split):
    return sorted(p for p in (dataset / split).glob("scene_*") if (p / "ground_truth.json").is_file())


def generate_dataset_questions(dataset, settings, overwrite=False):
    for split in ("train", "val", "test"):
        balancer = AnswerBalancer()
        for path in scene_dirs(dataset, split):
            target = path / "questions.json"
            if target.exists() and not overwrite:
                raise SystemExit(f"{target} existe déjà ; utiliser --overwrite.")
            truth = json.loads((path / "ground_truth.json").read_text())
            # Same seed derivation as generate_dataset.py, so both paths agree.
            result = generate_scene_questions(truth, settings, random.Random(truth["seed"] ^ 0x9E57), balancer)
            target.write_text(json.dumps(result, indent=2) + "\n")


def summarize(dataset):
    """Per-template counts, answer distributions and majority-answer accuracy."""
    for split in ("train", "val", "test"):
        paths = [p / "questions.json" for p in scene_dirs(dataset, split) if (p / "questions.json").is_file()]
        if not paths:
            continue
        answers = defaultdict(Counter)
        for path in paths:
            for q in json.loads(path.read_text())["questions"]:
                answers[(q["family"], q["template"])][q["answer"]] += 1
        total = sum(sum(c.values()) for c in answers.values())
        print(f"\n[{split}] {len(paths)} scenes, {total} questions ({total / len(paths):.1f}/scene)")
        print(f"{'template':<18}{'n':>7}{'answers':>9}{'majority':>10}  distribution")
        for (_, name), counts in sorted(answers.items()):
            n = sum(counts.values())
            top = ", ".join(f"{a}:{c}" for a, c in counts.most_common(6))
            print(f"{name:<18}{n:>7}{len(counts):>9}{max(counts.values()) / n:>10.1%}  {top}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        help="JSON file with a 'questions' section (default: the dataset's generation_config.json)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stats", action="store_true", help="only print statistics of existing questions")
    args = parser.parse_args()
    if not args.stats:
        config_path = args.config or args.dataset / "generation_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        generate_dataset_questions(args.dataset, copy.deepcopy(config.get("questions", {})), args.overwrite)
    summarize(args.dataset)


if __name__ == "__main__":
    main()
