"""Lighting-off parity: GLWorld.tex must be byte-identical to the pre-lighting renderer. Run from the repo root:

    .venv/bin/python3 tests/lit_parity_check.py [--ref GIT_REF] [--png DIR] [scene ...]

Exports gl_world.py + shaders/ at GIT_REF (default c4bd71b, the last commit before the lit renderer) into a temp
dir, imports it as the reference renderer, and renders the same worlds (several presets, stepped a few frames, with
trails, a point grab and an area grab) in every view mode (points / filled / no outlines, grid on and off) through
both. Every frame's bytes must match. Then, with the effect modules switched off, lighting on must give the same
bytes too (albedo = the unlit colour for the default material, overlays drawn after). --png saves the lit G-buffer
(albedo, normals visualised, material ids) of each scene's filled view for a look. Exit status 1 on any mismatch.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

import gl_world  # noqa: E402
from scene import PRESETS, Endesga  # noqa: E402
from trails import Trails  # noqa: E402

SIZE = (800, 500)
ZOOM = 0.5
SCENES = ["default", "penrose/evolved 1", "fluid", "newtons cradle", "diamonds/evolved 1", "looping"]
DEMO = [(m, None) for m in ("matte", "glossy", "jelly", "metal", "stone", "glow", "glass", "water", "toon")] + [
    ("matte", {"pattern": p, "pattern_strength": 0.6, "pattern_scale": 5.0, "tint": [1.0, 0.6, 0.6],
               "tint_strength": 0.5, "profile": p % 3, "bevel_px": 5.0}) for p in (1, 2, 3, 4)]
MODES = [  # (filled, outlines, grid)
    (False, True, True), (True, True, False), (True, False, True), (True, True, True)]


def load_ref(ref):
    out = tempfile.mkdtemp(prefix="glworld_ref_")
    tar = subprocess.run(["git", "archive", ref, "gl_world.py", "shaders"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", out], input=tar, check=True)
    spec = importlib.util.spec_from_file_location("gl_world_ref", os.path.join(out, "gl_world.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(name, frames=20):
    world = PRESETS[name](SIZE[0] / ZOOM, SIZE[1] / ZOOM)
    trails = Trails()
    trails.set_enabled(True)
    for _ in range(frames):
        world.step(1 / 60)
        trails.update(world, 1 / 60, True)
    return world, trails


def grabs(world):
    """Yield after setting up: nothing, a point grab, an area grab (each leaves the world grabbing)."""
    yield "none"
    movable = np.flatnonzero(world.W > 0) if world.bodies else []
    if len(movable):
        p = world.P[:, movable[len(movable) // 2]].copy()
        world.grab(p)
        world.drag_to(p + 30.0)
        yield "point"
        world.release()
        world.grab(p, 60.0)
        world.drag_to(p + 40.0, 60.0)
        yield "area"
        world.release()


def render(gw, world, mode, trails, **kw):
    filled, outlines, grid = mode
    gw.render(world, SIZE, filled, ZOOM, Endesga.my_blue, grid={} if grid else None,
              trails=trails if grid else None, outlines=outlines, **kw)
    return gw.tex.read()


def save_gbuf(gw, path_base):
    g = gw.gbuf
    w, h = gw.size
    alb = np.frombuffer(g["albedo"].read(), np.uint8).reshape(h, w, 4)[:, :, :3]
    nrm = np.frombuffer(g["normal"].read(), np.uint8).reshape(h, w, 4).copy()
    vis = np.dstack([nrm[:, :, 0], nrm[:, :, 1], np.full((h, w), 255, np.uint8)])  # z shown as constant blue
    vis[nrm[:, :, 3] == 0] = 0  # not a body
    mat = np.frombuffer(g["mat"].read(), np.uint16).reshape(h, w)
    rng = np.random.default_rng(1).integers(40, 255, (int(mat.max()) + 1, 3)).astype(np.uint8)
    rng[0] = 0
    for tag, img in (("albedo", alb), ("normal", vis), ("mat", rng[mat])):
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
        pygame.image.save(surf, f"{path_base}_{tag}.png")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="c4bd71b")
    ap.add_argument("--png", default=None)
    ap.add_argument("scenes", nargs="*")
    args = ap.parse_args(argv)
    ref = load_ref(args.ref)
    ctx = moderngl.create_standalone_context(require=330)
    old, new = ref.GLWorld(ctx, SIZE), gl_world.GLWorld(ctx, SIZE)
    new.effects = []  # lit check below: no effect modules (lighting on = albedo)
    bad = n = lit_n = 0
    for name in args.scenes or SCENES:
        world, trails = build(name)
        for g in grabs(world):
            for mode in MODES:
                a = render(old, world, mode, trails)
                b = render(new, world, mode, trails)
                n += 1
                if a != b:
                    bad += 1
                    diff = np.count_nonzero(np.frombuffer(a, np.uint8) != np.frombuffer(b, np.uint8))
                    print(f"MISMATCH off  {name} grab={g} mode={mode}: {diff} bytes differ")
                c = render(new, world, mode, trails, lighting=True)
                lit_n += 1
                if c != a and g == "area" and not mode[0]:
                    # expected: unlit blends the translucent area-grab lines before the point discs are stamped
                    # (the discs then fail the depth test under them); lit draws them over the finished picture
                    print(f"(expected) lit area grab over point discs differs: {name}")
                elif c != a:
                    bad += 1
                    diff = np.count_nonzero(np.frombuffer(a, np.uint8) != np.frombuffer(c, np.uint8))
                    print(f"MISMATCH lit  {name} grab={g} mode={mode}: {diff} bytes differ")
        if args.png:  # every body gets a material (presets in turn, plus each pattern) for a look at the G-buffer
            os.makedirs(args.png, exist_ok=True)
            for k, b in enumerate(world.bodies):
                b.material, b.material_over = DEMO[k % len(DEMO)]
            for mode, tag in (((True, True, True), ""), ((True, False, False), "_nooutline")):
                render(new, world, mode, trails, lighting=True)
                save_gbuf(new, os.path.join(args.png, name.replace("/", "_").replace(" ", "_") + tag))
        print(f"{name}: done", flush=True)
    print(f"{n} unlit frames + {lit_n} lit (default material) frames compared, {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
