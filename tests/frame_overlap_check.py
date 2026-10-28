"""Frames in flight: the app no longer calls ctx.finish() every frame (main_opengl.GpuTimer reads a GL query one
frame late), so gl_world's streamed data (orphaned instance buffers, the fill vertex texture ring) is rewritten while
earlier frames may still be drawing from it. This renders N consecutive frames of moving worlds three ways and checks
every frame's bytes match:

    reference   step + render + ctx.finish() each frame (the old loop)
    one late    step + render + GpuTimer, as the app runs now (one frame in flight)
    unbounded   step + render, no sync at all (as many frames queued as the driver takes)

Each frame's GLWorld.tex is copied on the GPU into its own texture (no read-back until the end, which would sync).
CPU physics (repeatable). Every view (points / filled / no outlines), lighting as the world has it plus lit on a plain world. Run from the
repo root:

    .venv/bin/python3 tests/frame_overlap_check.py [frames] [scene ...]

Exit status 1 on any mismatch.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from entities import EnvConfig  # noqa: E402
from gl_world import GLWorld  # noqa: E402
from main_opengl import GpuTimer, grid_colors  # noqa: E402
from scene import PRESETS, Endesga  # noqa: E402
from trails import Trails  # noqa: E402

WORLD = (1800, 1169)
ZOOM = 0.5
SIZE = (int(WORLD[0] * ZOOM), int(WORLD[1] * ZOOM))
SCENES = [("penrose/evolved 1 halloween", None), ("penrose/evolved 1", True), ("fluid", None)]
VIEWS = [(False, True), (True, True), (True, False)]  # (filled, outlines)


def run(ctx, name, lit, view, frames, mode):
    """Snapshots (one texture per frame) of `frames` frames of a fresh world; mode: "finish" / "late" / "none"."""
    world = PRESETS[name](*WORLD)
    if lit is not None:
        world.lighting = lit
    for _ in range(30):  # into motion
        world.step(1 / 60)
    gw = GLWorld(ctx, SIZE)
    trails = Trails()
    trails.set_enabled(True)
    timer = GpuTimer(ctx) if mode == "late" else None
    snaps = []
    filled, outlines = view
    for k in range(frames):
        world.step(1 / 60)
        trails.update(world, 1 / 60, True)
        if k == frames // 2:  # a grab mid-run: the overlay lines and their own pass
            movable = np.flatnonzero(world.W > 0)
            world.grab(world.P[:, movable[len(movable) // 2]].copy(), 60.0)
        if timer is not None:
            timer.begin()
        gw.render(world, SIZE, filled, ZOOM, world.background or Endesga.my_blue, outlines=outlines,
                  grid=grid_colors(world.background) if k % 3 == 0 else None, trails=trails if k % 2 else None)
        tex = ctx.texture(SIZE, 4)
        fbo = ctx.framebuffer([tex])
        ctx.copy_framebuffer(fbo, gw.fbo)
        snaps.append((tex, fbo))
        if timer is not None:
            timer.end()
            timer.collect()
        elif mode == "finish":
            ctx.finish()
    out = [tex.read() for tex, _ in snaps]
    for tex, fbo in snaps:
        fbo.release()
        tex.release()
    gw.release()
    return out


def main(argv):
    frames = int(argv[0]) if argv else 24
    names = argv[1:]
    scenes = [(n, None) for n in names] if names else SCENES
    pygame.init()
    EnvConfig.backend = "cpu"  # bitwise repeatable physics (the MLX solves' float32 atomics aren't run to run)
    ctx = moderngl.create_standalone_context(require=330)
    bad = n = 0
    for name, lit in scenes:
        for view in VIEWS:
            ref = run(ctx, name, lit, view, frames, "finish")
            for mode in ("late", "none"):
                got = run(ctx, name, lit, view, frames, mode)
                for k, (a, b) in enumerate(zip(ref, got)):
                    n += 1
                    if a != b:
                        bad += 1
                        diff = np.count_nonzero(np.frombuffer(a, np.uint8) != np.frombuffer(b, np.uint8))
                        print(f"MISMATCH {name} lit={lit} view={view} {mode} frame {k}: {diff} bytes differ")
            print(f"{name} lit={lit} view={view}: {frames} frames x 2 modes compared", flush=True)
    print(f"{n} frames compared, {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
