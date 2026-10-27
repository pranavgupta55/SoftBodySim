"""Robustness checks for GLWorld.render on a standalone context, and the persistent surface-pattern seed. Run from the
repo root:

    .venv/bin/python3 tests/gl_world_guard_check.py

- render with a released framebuffer bound, release() then render again (lit and unlit): no error
- a zero-size view (0 x 0, 0 x h, w x 0), lit and unlit: no error, the frame is the plain background
- pattern_seed / pattern_rest round-trip through scene_io: a saved and reloaded lit scene (also after its soft bodies
  deformed, and rendered by a GLWorld that has seen other bodies first) renders with a 0 px diff
Exit status 1 on the first failed check.
"""
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

pygame.init()
import scene  # noqa: E402
import scene_io  # noqa: E402
from entities import FixedBody, rect_points  # noqa: E402
from gl_world import GLWorld  # noqa: E402

W, H, ZOOM = 900, 584, 0.5


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def img(gw):
    return np.frombuffer(gw.tex.read(), np.uint8).reshape(gw.size[1], gw.size[0], 4)[:, :, :3].astype(np.int64)


def demo(steps=0):
    w = scene.PRESETS["lighting demo"](W / ZOOM, H / ZOOM)
    for _ in range(steps):
        w.step(1 / 60)
    return w


def main():
    ctx = moderngl.create_standalone_context(require=410)
    world = demo()
    bg = tuple(world.background or (20, 22, 30))

    # S6: a released framebuffer bound when render starts, and render after release()
    gw = GLWorld(ctx, (W, H))
    for lit in (False, True):
        fb = ctx.framebuffer([ctx.texture((4, 4), 4)])
        fb.use()
        fb.release()
        gw.render(world, (W, H), True, ZOOM, bg, lighting=lit)
        check(True, f"render with a released framebuffer bound (lit {lit})")
    other = GLWorld(ctx, (W, H))
    other.render(world, (W, H), True, ZOOM, bg, lighting=True)
    other.fbo.use()
    other.release()
    other.fbo.release()
    gw.render(world, (W, H), True, ZOOM, bg, lighting=True)
    check(True, "render after another GLWorld's framebuffer was released")
    gw.release()
    for _ in range(2):  # (the first lit frame after a reset differs: the effects' temporal state warms up)
        gw.render(world, (W, H), True, ZOOM, bg, lighting=True)
    a = img(gw)
    check(a.std() > 1, "release() then render again (lit): a full frame")
    gw.render(world, (W, H), True, ZOOM, bg)
    check(img(gw).std() > 1, "... and unlit")
    gw.render(world, (W, H), True, ZOOM, bg, lighting=True)

    # A11: zero-size views
    for lit in (False, True):
        for vs in ((0, 0), (0, 100), (100, 0), (-5, 10)):
            gw.render(world, vs, True, ZOOM, bg, lighting=lit)
            ctx.finish()
            b = img(gw)
            check((b == np.array(bg[:3])).all(), f"view {vs} lit {lit}: no error, plain background")
    gw.render(world, (W, H), True, ZOOM, bg, lighting=True)
    check(np.array_equal(img(gw), a), "a full view afterwards renders as before")

    # S4: pattern_seed round trip (the s6b scenario: same GLWorld, original then reloaded world)
    box = (W / ZOOM, H / ZOOM, scene.INSET)
    for steps in (0, 40):
        gw2 = GLWorld(ctx, (W, H))
        if steps:
            gw2.render(demo(), (W, H), True, ZOOM, bg, lighting=True)  # this GLWorld has seen other bodies first
        w1 = demo()
        gw2.render(w1, (W, H), True, ZOOM, bg, lighting=True)  # first sight: seeds and rest frames assigned
        for _ in range(steps):
            w1.step(1 / 60)
        # (saving settles the world's open CCD span first, which may move it: render after)
        data = json.loads(scene_io.dumps(scene_io.scene_data(w1, box, view=scene.ViewConfig)))
        for _ in range(2):
            gw2.render(w1, (W, H), True, ZOOM, bg, lighting=True)
        a = img(gw2)
        check(all("pattern_seed" in d for d in data["bodies"]), f"steps {steps}: every body saves its pattern_seed")
        w2 = scene.PRESETS["blank"](W / ZOOM, H / ZOOM)
        scene_io.load_data(w2, data, box)
        check([b.pattern_seed for b in w2.bodies] == [b.pattern_seed for b in w1.bodies],
              f"steps {steps}: pattern seeds round-trip")
        for _ in range(2):
            gw2.render(w2, (W, H), True, ZOOM, bg, lighting=True)
        b = img(gw2)
        diff = int((np.abs(a - b).sum(axis=2) > 0).sum())
        check(diff == 0, f"steps {steps}: reloaded scene renders identically ({diff} px differ)")
        gw2.release()

    # old files without the keys: seeds by index; new bodies get fresh seeds past them
    for d in data["bodies"]:
        d.pop("pattern_seed", None)
        d.pop("pattern_rest", None)
    w3 = scene.PRESETS["blank"](W / ZOOM, H / ZOOM)
    w3.pattern_seq = 500  # a counter left over from the world's previous scene
    scene_io.load_data(w3, data, box)
    gw.render(w3, (W, H), True, ZOOM, bg, lighting=True)
    check([b.pattern_seed for b in w3.bodies] == list(range(len(w3.bodies))), "old file: seeds default to the index")
    again = json.loads(scene_io.dumps(scene_io.scene_data(w3, box, view=scene.ViewConfig)))
    w4 = scene.PRESETS["blank"](W / ZOOM, H / ZOOM)
    scene_io.load_data(w4, again, box)
    w4.add_many([FixedBody(rect_points(300, 300, 40, 40))])
    gw.render(w4, (W, H), True, ZOOM, bg, lighting=True)
    check(w4.bodies[-1].pattern_seed == len(w3.bodies), "a new body gets the next free seed")
    gw.release()
    ctx.release()
    print("all passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL", e)
        sys.exit(1)
