"""GLWorld.render cost with lighting off vs on. Run from the repo root:

    .venv/bin/python3 tests/lit_perf_check.py [frames] [scene ...]

Offscreen standalone context, render surface 800x500 at zoom 0.5 (main_opengl's size for a 1600x1000 window), the
filled view with grid. Per scene (default: penrose/evolved 1, fluid) and mode it steps the world one tick per frame
and prints the median ms of the render() call alone (CPU side: GL calls are queued) and of render() + ctx.finish()
(CPU + GPU). Lit runs use no effect modules (G-buffer + albedo copy) and give every body a material in turn, with
patterns, so the material and rest-frame paths are exercised. The "ref" row is the pre-lighting renderer
(lit_parity_check.load_ref at c4bd71b) on the same world.
"""
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

import gl_world  # noqa: E402
from scene import PRESETS, Endesga  # noqa: E402

SIZE, ZOOM = (800, 500), 0.5
DEMO = ["matte", "glossy", "stone", "metal", "toon"]


def measure(gw, ctx, world, frames, lit, sync):
    times = []
    kw = {"lighting": True} if lit else {}
    for _ in range(frames):
        world.step(1 / 60)
        t0 = time.perf_counter()
        gw.render(world, SIZE, True, ZOOM, Endesga.my_blue, grid={}, **kw)
        if sync:
            ctx.finish()
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times[3:]))


def main(argv):
    frames = int(argv[0]) if argv else 60
    names = argv[1:] or ["penrose/evolved 1", "fluid"]
    ctx = moderngl.create_standalone_context(require=330)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lit_parity_check import load_ref
    ref = load_ref("c4bd71b")
    for name in names:
        world = PRESETS[name](SIZE[0] / ZOOM, SIZE[1] / ZOOM)
        gw = gl_world.GLWorld(ctx, SIZE)
        gw.effects = []
        gw.render(world, SIZE, True, ZOOM, Endesga.my_blue)
        ctx.finish()
        out = {}
        old = ref.GLWorld(ctx, SIZE)
        for sync in (False, True):
            out["ref", sync] = measure(old, ctx, world, frames, False, sync)
        for lit in (False, True):
            if lit:
                for k, b in enumerate(world.bodies):
                    b.material = DEMO[k % len(DEMO)]
            for sync in (False, True):
                out[lit, sync] = measure(gw, ctx, world, frames, lit, sync)
        print(f"{name} ({len(world.bodies)} bodies): render() ms, median of {frames}")
        for lit, label in (("ref", "ref         "), (False, "lighting off"), (True, "lighting on ")):
            print(f"  {label}  cpu {out[lit, False]:6.2f}   cpu+gpu {out[lit, True]:6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
