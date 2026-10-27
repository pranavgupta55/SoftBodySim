"""Time scale vs body shape in the fluid world. Run from the repo root:

    .venv/bin/python3 tests/fluid_scale_check.py [backend] [scales]

Longer substeps used to squash the fluid's octagons (bottom-row area / rest area 0.82 at h = 1/240 s, 0.57 at
1/60 s), and high time scales used to lengthen them (physics_budget, max_substeps). With fixed ticks a time scale
only changes how many ticks run, and an over-budget frame drops time instead. Settles the presets_basic fluid world
for 6 s at 1x, then from that state runs each scale (default 1, 4, 10) the way main_opengl does (1/120 s real frames,
physics_budget on) for 4 simulated seconds, and averages each body's area / rest area over the last second, bottom
quarter of the pile (largest y). Every scale must stay within 0.02 of scale 1's bottom-quarter mean (exit status 1
if not). Also prints the delivered speed (simulated / requested time) and the tick length.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

from entities import EnvConfig  # noqa: E402
import presets_basic  # noqa: E402

SIZE = (1620, 1168)  # world units of an 1800 x 1169 window (as the app sizes it)
FRAME = 1 / 120
TOLERANCE = 0.02


def areas(world, movable):
    out = []
    for k in movable:
        x, y = world.bodies[k].points
        out.append(0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)))
    return np.array(out)


def bottom_area(world, movable, rest):
    a = areas(world, movable) / rest
    cy = np.array([world.bodies[k].points[1].mean() for k in movable])
    bot = np.argsort(cy)[-(len(movable) // 4):]
    return a[bot].mean(), a.mean()


def main(argv):
    EnvConfig.backend = argv[0] if argv else "cpu"
    scales = [float(s) for s in argv[1].split(",")] if len(argv) > 1 else [1.0, 4.0, 10.0]
    world = presets_basic.build_fluid(*SIZE)
    movable = [k for k, b in enumerate(world.bodies) if not b.fixed]
    rest = areas(world, movable)
    for _ in range(6 * 60):
        world.step(1 / 60)
    saved = (world.P.copy(), world.PREV.copy(), world.V.copy(), world.ticks)
    results = {}
    for scale in scales:
        world.P[:], world.PREV[:], world.V[:] = saved[:3]
        world._substep_cost, world._tick_acc = None, 0.0
        start, sim, asked, samples = world.ticks, 0.0, 0.0, []
        while sim < 4.0:
            world.step(FRAME, scale)
            sim += world.last_sim_dt
            asked += world.last_requested_dt
            if sim >= 3.0 and world.last_sim_dt > 0:
                samples.append(bottom_area(world, movable, rest))
        bot, allb = np.mean(samples, axis=0)
        results[scale] = bot
        print(f"scale {scale:5g}: bottom-quarter area/rest {bot:.3f}, all bodies {allb:.3f}, delivered "
              f"{sim / asked * scale:.2f}x of {scale:g}x, {world.ticks - start} ticks of 1/{1 / world.tick_length():g} s "
              f"({type(world.backend).__name__})", flush=True)
    ref = results[scales[0]]
    bad = [s for s, v in results.items() if abs(v - ref) > TOLERANCE]
    print("OK" if not bad else f"FAIL: scales {bad} differ from {scales[0]:g}x by more than {TOLERANCE}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
