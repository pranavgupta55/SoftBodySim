"""Save / load scenes as versioned JSON, validate loaded states, and the world-level undo history.

File format (FORMAT, VERSION 1), one JSON object:

    {"format": "softbodysim-scene", "version": 1, "name": "...",
     "world": {"width": W, "height": H, "inset": INSET},      # world units; the box the scene was made in
     "env": {"gravity": [gx, gy], "friction": f, "damping": d},
     "view": {"filled": bool, "show_grid": bool, "zoom": z},
     "bodies": [{"type": "soft" | "rigid" | "fixed" | "body", "wall": bool,
                 "points": [[x...], [y...]], "prev": ..., "vels": ..., "inv_mass": [...],
                 "edges": [[i...], [j...]], "winding": +-1.0,
                 "point_col": [r, g, b], "stick_col": ..., "fill_col": ..., "point_radius": r, "stick_width": w,
                 # soft:  "stiffness", "skip_sets", "springs": [[i...], [j...]], "rest_len": [...]
                 # rigid: "stiffness", "rest": [[x...], [y...]]
                 }, ...],                                       # in draw order (containers before contents)
     "hooks": [{"pos": [x, y], "col": [r, g, b], "radius": r}, ...],
     "ropes": [{"ends": [END, END], "rest_length": L, "stiffness": s, "col": [r, g, b], "width": w}, ...]}

    END = {"body": body_index, "point": point_index} | {"hook": hook_index} | {"anchor": [x, y]}

Chains are saved as their pieces (rigid boxes + ropes), which is all the World keeps of them. Floats are written with
Python's shortest round-trip repr, so save -> load gives bit-identical arrays. A scene made in a different world box
(another screen size) is moved to the new box: centred horizontally, its floor kept at the same distance from the
bottom wall, the walls themselves rebuilt on the new box; it's refused if it then doesn't fit.

Every loaded state is validated (state_problems) before it replaces the world: the collision system assumes it starts
from an intersection-free state, so no crossing edges and every point at least EnvConfig.contact_radius from other
bodies' edges (containment is fine).
"""
import json
import math
import os
import re

import numpy as np

from entities import Body, EnvConfig, FixedBody, Hook, RigidBody, Rope, SoftBody

FORMAT = "softbodysim-scene"
VERSION = 1
SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")
HISTORY_LIMIT = 64  # world undo snapshots kept (oldest dropped first)
DEFAULT_ENV = {"gravity": [float(v) for v in EnvConfig.gravity], "friction": float(EnvConfig.friction),
               "damping": float(EnvConfig.damping)}  # applied when a preset (which has no env of its own) is loaded
_KINDS = ((SoftBody, "soft"), (RigidBody, "rigid"), (FixedBody, "fixed"), (Body, "body"))
_CLASSES = {"soft": SoftBody, "rigid": RigidBody, "fixed": FixedBody, "body": Body}


class SceneError(ValueError):
    """A scene that can't be loaded: malformed, a newer version, or a state that intersects / doesn't fit."""


def _kind(body):
    for cls, name in _KINDS:
        if isinstance(body, cls):
            return name
    raise SceneError(f"not a body: {body!r}")


def is_wall(body, box):
    """One of the world box's walls: a fixed 2-point segment lying on the box outline. box = (width, height, inset)."""
    if not body.fixed or body.points.shape[1] != 2:
        return False
    w, h, inset = box
    (x0, x1), (y0, y1) = body.points
    return any(abs(a - v) < 1e-6 and abs(b - v) < 1e-6
               for a, b, v in ((x0, x1, inset), (x0, x1, w - inset), (y0, y1, inset), (y0, y1, h - inset)))


# ---------------- World -> data
def capture(world, box=None, as_json=True):
    """The world's bodies, hooks and ropes as a dict. as_json: plain lists (for json.dump); False keeps numpy copies
    (the cheaper in-memory undo snapshot). box (width, height, inset) marks the walls."""
    arr = (lambda a: np.asarray(a).tolist()) if as_json else (lambda a: np.array(a))
    col = (lambda c: [int(v) for v in c]) if as_json else tuple
    index = {id(b): k for k, b in enumerate(world.bodies)}
    hook_index = {id(hk.pos): k for k, hk in enumerate(world.hooks)}
    bodies = []
    for b in world.bodies:
        d = {"type": _kind(b), "wall": bool(box is not None and is_wall(b, box)),
             "points": arr(b.points), "prev": arr(b.prev), "vels": arr(b.vels), "inv_mass": arr(b.inv_mass),
             "edges": arr(b.edges), "winding": float(b.winding),
             "point_col": col(b.point_col), "stick_col": col(b.stick_col), "fill_col": col(b.fill_col),
             "point_radius": b.point_radius, "stick_width": b.stick_width}
        if isinstance(b, SoftBody):
            d.update(stiffness=float(b.stiffness), skip_sets=int(b.skip_sets), springs=arr(b.springs),
                     rest_len=arr(b.rest_len))
        elif isinstance(b, RigidBody):
            d.update(stiffness=float(b.stiffness), rest=arr(b.rest))
        bodies.append(d)
    ropes = []
    for r in world.ropes:
        ends = []
        for body, ref in r.ends:
            if body is not None:
                if id(body) not in index:
                    raise SceneError("a rope is attached to a body that isn't in the world")
                ends.append({"body": index[id(body)], "point": int(ref)})
            elif id(ref) in hook_index:  # a Hook end shares the hook's position array
                ends.append({"hook": hook_index[id(ref)]})
            else:
                ends.append({"anchor": arr(ref)})
        ropes.append({"ends": ends, "rest_length": float(r.rest_length), "stiffness": float(r.stiffness),
                      "col": col(r.col), "width": r.width})
    hooks = [{"pos": arr(hk.pos), "col": col(hk.col), "radius": hk.radius} for hk in world.hooks]
    return {"bodies": bodies, "hooks": hooks, "ropes": ropes}


def scene_data(world, box, view=None, name=""):
    """A complete, JSON-ready scene file dict for the world in box (width, height, inset)."""
    data = {"format": FORMAT, "version": VERSION, "name": name,
            "world": {"width": float(box[0]), "height": float(box[1]), "inset": float(box[2])}, "env": current_env()}
    if view is not None:
        data["view"] = {"filled": bool(view.filled), "show_grid": bool(view.show_grid), "zoom": float(view.zoom)}
    data.update(capture(world, box))
    return data


def dumps(data):
    """Compact-ish JSON: one body / rope / hook per line, so files diff sensibly."""
    head = {k: v for k, v in data.items() if k not in ("bodies", "hooks", "ropes")}
    out = json.dumps(head, allow_nan=False)[:-1]
    for key in ("bodies", "hooks", "ropes"):
        items = data.get(key, [])
        rows = ",\n  ".join(json.dumps(item, allow_nan=False, separators=(",", ":")) for item in items)
        out += f', "{key}": [\n  {rows}\n]' if items else f', "{key}": []'
    return out + "}\n"


# ---------------- Data -> world
def _float_array(value, shape_rows=None, what="array"):
    a = np.array(value, dtype=float)
    if shape_rows is not None and (a.ndim != 2 or a.shape[0] != shape_rows):
        raise SceneError(f"{what}: expected {shape_rows} rows")
    if not np.isfinite(a).all():
        raise SceneError(f"{what}: not finite")
    return a


def build(data, offset=(0.0, 0.0)):
    """(bodies, ropes, hooks) from capture() / scene_data() output, everything moved by offset (world units; walls
    are handled by the caller). Raises SceneError on malformed data."""
    off = np.asarray(offset, dtype=float).reshape(2, 1)
    moved = bool(off.any())
    bodies = []
    try:
        for k, d in enumerate(data["bodies"]):
            cls = _CLASSES.get(d["type"])
            if cls is None:
                raise SceneError(f"body {k}: unknown type {d['type']!r}")
            pts = _float_array(d["points"], 2, f"body {k} points")
            n = pts.shape[1]
            style = dict(point_col=tuple(d["point_col"]), stick_col=tuple(d["stick_col"]), fill_col=tuple(d["fill_col"]),
                         point_radius=d["point_radius"], stick_width=d["stick_width"])
            if cls is SoftBody:
                b = SoftBody(pts, stiffness=float(d["stiffness"]), skip_sets=int(d["skip_sets"]), **style)
            elif cls is RigidBody:
                b = RigidBody(pts, stiffness=float(d["stiffness"]), **style)
            else:
                b = cls(pts, **style)
            if moved and not d.get("wall"):
                b.points += off
            b.prev[:] = _float_array(d["prev"], 2, f"body {k} prev") + (off if moved and not d.get("wall") else 0.0)
            b.vels[:] = _float_array(d["vels"], 2, f"body {k} vels")
            b.inv_mass[:] = _float_array(d["inv_mass"], None, f"body {k} inv_mass")
            edges = np.array(d["edges"], dtype=int).reshape(2, -1)
            if edges.size and (edges.min() < 0 or edges.max() >= n):
                raise SceneError(f"body {k}: edge index out of range")
            b.edges = b.draw_edges = edges
            b.winding = float(d["winding"])
            if cls is SoftBody:
                springs = np.array(d["springs"], dtype=int).reshape(2, -1)
                if springs.size and (springs.min() < 0 or springs.max() >= n):
                    raise SceneError(f"body {k}: spring index out of range")
                b.springs = b.draw_edges = springs
                b.rest_len = _float_array(d["rest_len"], None, f"body {k} rest_len")
                if b.rest_len.shape != (springs.shape[1],):
                    raise SceneError(f"body {k}: one rest length per spring expected")
                b.spring_count = np.maximum(np.bincount(springs.ravel(), minlength=n), 1)
            elif cls is RigidBody:
                b.rest = _float_array(d["rest"], 2, f"body {k} rest")
                if b.rest.shape != pts.shape:
                    raise SceneError(f"body {k}: rest shape size differs from its points")
            bodies.append(b)
        hooks = [Hook(_float_array(h["pos"], None, "hook pos") + off[:, 0], col=tuple(h["col"]), radius=h["radius"])
                 for h in data["hooks"]]
        ropes = []
        for k, r in enumerate(data["ropes"]):
            ends = []
            for e in r["ends"]:
                if "body" in e:
                    body, idx = bodies[int(e["body"])], int(e["point"])
                    if not 0 <= idx < body.points.shape[1]:
                        raise SceneError(f"rope {k}: point index out of range")
                    ends.append((body, idx))
                elif "hook" in e:
                    ends.append(hooks[int(e["hook"])])
                else:
                    ends.append(tuple(_float_array(e["anchor"], None, f"rope {k} anchor") + off[:, 0]))
            ropes.append(Rope(ends[0], ends[1], rest_length=float(r["rest_length"]), stiffness=float(r["stiffness"]),
                              col=tuple(r["col"]), width=r["width"]))
    except SceneError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise SceneError(f"malformed scene ({type(e).__name__}: {e})") from None
    return bodies, ropes, hooks


def _fit_walls(bodies, data, box):
    """Rebuild the walls saved in data's box on the current box (each endpoint coordinate on a saved wall line moves
    to the matching current one)."""
    sw, sh, si = (float(data["world"][k]) for k in ("width", "height", "inset"))
    w, h, inset = box
    xmap = {si: inset, sw - si: w - inset}
    ymap = {si: inset, sh - si: h - inset}

    def snap(v, table):
        for old, new in table.items():
            if abs(v - old) < 1e-6:
                return new
        return v
    for b, d in zip(bodies, data["bodies"]):
        if d.get("wall"):
            for c in range(b.points.shape[1]):
                b.points[0, c] = snap(b.points[0, c], xmap)
                b.points[1, c] = snap(b.points[1, c], ymap)
            b.prev[:] = b.points


def state_problems(bodies, box, min_gap=None, limit=5):
    """Why this state can't start a simulation (empty list if it can): points outside the world box, edges of
    different bodies (or a self-colliding body's non-neighboring edges) crossing, or a point closer than min_gap
    (default EnvConfig.contact_radius, the placement rule) to another body's edge. Pairs of fixed bodies are exempt,
    like in the solver. Containment is allowed. limit: stop after this many problems."""
    problems = []
    if not bodies:
        return ["the scene has no bodies"]
    w, h, inset = box
    P = np.concatenate([b.points for b in bodies], axis=1)
    if not np.isfinite(P).all():
        return ["a point is not finite"]
    counts = [b.points.shape[1] for b in bodies]
    offsets = np.cumsum([0] + counts)
    point_body = np.repeat(np.arange(len(bodies)), counts)
    edges = np.concatenate([b.edges + o for b, o in zip(bodies, offsets[:-1])], axis=1)
    edge_body = np.repeat(np.arange(len(bodies)), [b.edges.shape[1] for b in bodies])
    fixed = np.array([b.fixed for b in bodies])
    selfc = np.array([b.self_collide for b in bodies])
    eps = 1e-6
    out = (P[0] < inset - eps) | (P[0] > w - inset + eps) | (P[1] < inset - eps) | (P[1] > h - inset + eps)
    if out.any():
        problems.append(f"{int(out.sum())} point(s) outside the world box (body {int(point_body[np.argmax(out)])})")
    r = EnvConfig.contact_radius if min_gap is None else min_gap
    a, b = P[:, edges[0]], P[:, edges[1]]
    e_lo, e_hi = np.minimum(a, b), np.maximum(a, b)
    # points vs edges of other bodies (not both fixed): distance >= contact_radius
    for s in range(0, P.shape[1], 256):
        p = P[:, s:s + 256]
        pb = point_body[s:s + 256]
        near = ((p[:, :, None] >= e_lo[:, None, :] - r) & (p[:, :, None] <= e_hi[:, None, :] + r)).all(axis=0)
        near &= (pb[:, None] != edge_body[None, :]) & ~(fixed[pb][:, None] & fixed[edge_body][None, :])
        i, e = np.nonzero(near)
        if i.size:
            ab = b[:, e] - a[:, e]
            t = np.clip(np.sum((p[:, i] - a[:, e]) * ab, axis=0) / np.maximum(np.sum(ab * ab, axis=0), 1e-12), 0, 1)
            d = np.hypot(*(p[:, i] - (a[:, e] + t * ab)))
            close = d < r - 1e-9
            if close.any():
                k = int(np.argmax(close))
                problems.append(f"a point of body {int(pb[i[k]])} is {d[k]:.2f} from an edge of body "
                                f"{int(edge_body[e[k]])} (closer than {r:g})")
                if len(problems) >= limit:
                    return problems
    # edges vs edges: no proper crossings (different bodies not both fixed, or a self-colliding body's own
    # non-adjacent edges)

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    E = edges.shape[1]
    for s in range(0, E, 256):
        sl = slice(s, s + 256)
        eb = edge_body[sl]
        near = ((e_lo[:, sl, None] <= e_hi[:, None, :]) & (e_hi[:, sl, None] >= e_lo[:, None, :])).all(axis=0)
        same = eb[:, None] == edge_body[None, :]
        shared = ((edges[0, sl, None] == edges[0][None, :]) | (edges[0, sl, None] == edges[1][None, :]) |
                  (edges[1, sl, None] == edges[0][None, :]) | (edges[1, sl, None] == edges[1][None, :]))
        allowed = np.where(same, selfc[eb][:, None] & ~shared, ~(fixed[eb][:, None] & fixed[edge_body][None, :]))
        near &= allowed & (np.arange(s, s + eb.size)[:, None] < np.arange(E)[None, :])
        i, j = np.nonzero(near)
        i = i + s
        if i.size:
            p, q, c, d = a[:, i], b[:, i], a[:, j], b[:, j]
            hit = (cross(q - p, c - p) * cross(q - p, d - p) < 0) & (cross(d - c, p - c) * cross(d - c, q - c) < 0)
            if hit.any():
                k = int(np.argmax(hit))
                problems.append(f"an edge of body {int(edge_body[i[k]])} crosses an edge of body {int(edge_body[j[k]])}")
                if len(problems) >= limit:
                    return problems
    return problems


def load_gap():
    """Smallest point-edge gap a loaded state may have. Not contact_radius (the rule for new placements): bodies
    resting on each other settle slightly inside it (2.7 of 3 measured), so a scene saved from a running simulation
    would be refused. What continuous collision detection needs is a strictly intersection-free start; ccd_gap is the
    gap it keeps itself (halved for rounding slack)."""
    return EnvConfig.ccd_gap / 2


def parse(data, box):
    """Check a scene dict and build it for the world box (width, height, inset): (bodies, ropes, hooks). Raises
    SceneError with the reason if it's malformed, from a newer version, or its state intersects / doesn't fit."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise SceneError("not a scene file")
    if not isinstance(data.get("version"), int) or data["version"] > VERSION:
        raise SceneError(f"scene version {data.get('version')!r} is newer than this game ({VERSION})")
    try:
        sw, sh, si = (float(data["world"][k]) for k in ("width", "height", "inset"))
    except (KeyError, TypeError, ValueError):
        raise SceneError("malformed scene (world box)") from None
    w, h, inset = box
    same = abs(sw - w) < 1e-9 and abs(sh - h) < 1e-9 and abs(si - inset) < 1e-9
    # another box: centre horizontally, keep the distance to the floor
    offset = (0.0, 0.0) if same else ((w - sw) / 2, (h - inset) - (sh - si))
    bodies, ropes, hooks = build(data, offset)
    if not same:
        _fit_walls(bodies, data, box)
    problems = state_problems(bodies, box, load_gap())
    if problems:
        raise SceneError("; ".join(problems))
    return bodies, ropes, hooks


def install(world, bodies, ropes, hooks):
    """Replace the world's contents in place (same World object, so every reference to it stays valid)."""
    world.bodies, world.ropes, world.hooks = list(bodies), list(ropes), list(hooks)
    world.grabbed = None
    world._rebuild()


def apply_env(env):
    EnvConfig.gravity = np.array(env["gravity"], dtype=float)
    EnvConfig.friction = float(env["friction"])
    EnvConfig.damping = float(env["damping"])


def load_data(world, data, box, view=None):
    """Validate a scene dict and, if it's fine, replace the world's contents with it and apply its env and view
    settings (filled / grid). Raises SceneError (the world is untouched then)."""
    parsed = parse(data, box)
    env = data.get("env", DEFAULT_ENV)
    try:
        env = {"gravity": [float(v) for v in env["gravity"]][:2], "friction": float(env["friction"]),
               "damping": float(env["damping"])}
    except (KeyError, TypeError, ValueError):
        raise SceneError("malformed scene (env)") from None
    install(world, *parsed)
    apply_env(env)
    if view is not None and isinstance(data.get("view"), dict):
        v = data["view"]
        view.filled = bool(v.get("filled", view.filled))
        view.show_grid = bool(v.get("show_grid", view.show_grid))


# ---------------- Files
def safe_name(name):
    """A file-system-safe scene name: letters, digits, - and _ (others become _), at most 32 chars."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name.strip())[:32].strip("_") or "scene"


def scene_path(name, folder=None):
    return os.path.join(folder or SCENES_DIR, safe_name(name) + ".json")


def list_scenes(folder=None):
    """Saved scene names (file names without .json), sorted."""
    folder = folder or SCENES_DIR
    if not os.path.isdir(folder):
        return []
    return sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".json") and not f.startswith("."))


def auto_name(folder=None):
    """The first unused scene_N name."""
    taken = set(list_scenes(folder))
    n = 1
    while f"scene_{n}" in taken:
        n += 1
    return f"scene_{n}"


def save_file(world, box, name, view=None, folder=None):
    """Write the world to <folder>/<name>.json (folder created if needed). Returns the path."""
    return write_data(scene_data(world, box, view), name, folder)


def write_data(data, name, folder=None):
    """Write a scene_data() dict to <folder>/<name>.json under the (sanitised) name. Returns the path."""
    path = scene_path(name, folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = dumps(dict(data, name=safe_name(name)))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)  # never leaves a half-written scene behind
    return path


def read_file(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SceneError(f"can't read {os.path.basename(path)} ({type(e).__name__})") from None


def preset_data(name, box):
    """scene.PRESETS[name] built for the box, as a scene dict with the default env."""
    from scene import PRESETS
    w, h, inset = box
    world = PRESETS[name](w, h)
    data = {"format": FORMAT, "version": VERSION, "name": name,
            "world": {"width": float(w), "height": float(h), "inset": float(inset)}, "env": DEFAULT_ENV}
    data.update(capture(world, box))
    return data


# ---------------- World undo / redo
def snapshot_bytes(snap):
    """Approximate memory of an in-memory snapshot (its numpy arrays)."""
    total = 0
    for d in snap["bodies"] + snap["hooks"] + snap["ropes"]:
        for v in d.values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
        for e in d.get("ends", []):
            if "anchor" in e:
                total += e["anchor"].nbytes
    return total


def current_env():
    return {"gravity": [float(v) for v in EnvConfig.gravity], "friction": float(EnvConfig.friction),
            "damping": float(EnvConfig.damping)}


class WorldHistory:
    """Snapshot undo / redo for the whole world. push() before a change; undo() restores the latest snapshot (saving
    the current state for redo); redo() goes forward again. Snapshots are capture(as_json=False) dicts: numpy copies
    of every body's arrays plus the object graph by index, rebuilt into the same World object on restore (exact
    arrays). Gravity and the other EnvConfig settings are only part of a step pushed with env=True (a scene load,
    which sets them); other steps leave them as they are. At most `limit` undo steps (oldest dropped)."""

    def __init__(self, limit=HISTORY_LIMIT):
        self.limit = limit
        self.undo_stack, self.redo_stack = [], []  # (label, snapshot)
        self.world = None  # the World the snapshots belong to (a different one clears them)

    def _bind(self, world):
        if world is not self.world:
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.world = world

    def _snap(self, world, env):
        snap = capture(world, as_json=False)
        if env:
            snap["env"] = current_env()
        return snap

    def push(self, world, label, env=False):
        self._bind(world)
        self.undo_stack.append((label, self._snap(world, env)))
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()

    def can_undo(self, world):
        return world is self.world and bool(self.undo_stack)

    def can_redo(self, world):
        return world is self.world and bool(self.redo_stack)

    def _step(self, world, src, dst):
        self._bind(world)
        if not src:
            return None
        label, snap = src.pop()
        dst.append((label, self._snap(world, "env" in snap)))
        install(world, *build(snap))
        if "env" in snap:
            apply_env(snap["env"])
        return label

    def undo(self, world):
        """Restore the state before the last change; returns its label (None if there's nothing to undo)."""
        return self._step(world, self.undo_stack, self.redo_stack)

    def redo(self, world):
        """Go forward to the state the last undo left; returns its label (None if there's nothing to redo)."""
        return self._step(world, self.redo_stack, self.undo_stack)

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
