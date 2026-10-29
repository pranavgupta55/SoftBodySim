"""The Look card's overrides (visual_tune.py) on the halloween map. Run from the repo root:

    .venv/bin/python3 tests/visual_tune_check.py [--png DIR]

1. attach() on a freshly built map reads the map's own values (its lighting block) as the defaults.
2. Every slider changes the lit image (each set to a value away from the map's, rendered offscreen, compared with
   the untouched render), in every view mode (points / filled / no outlines), and leaves the bodies' positions alone.
3. The values survive a palette switch (palettes.apply_palette) and come back on switching back.
4. Save -> a fresh build of the map -> attach: the saved values are applied, and its render matches the tuned world's.
   Reset all + save removes the map's entry. A corrupt or missing file gives the map's own values, no error.
5. The Look card draws, a slider drag and a label click (reset) work.
Uses a temp overrides file (never local/visual_overrides.json). Exit status 1 on any failure.
"""
import os
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

pygame.init()

import palettes  # noqa: E402
import visual_tune as VT  # noqa: E402
from gl_world import GLWorld  # noqa: E402
from scene import PRESETS  # noqa: E402
from ui_style import HUD  # noqa: E402

MAP = "penrose/evolved 1 halloween"
SIZE = (800, 520)
ZOOM = 0.5
EXPECT = {"gi_gain": 6.4, "ambient": 0.08, "floor_lightness": 0.4, "floor_saturation": 0.3, "shape_glow": 1.0,
          "shape_albedo": palettes.GLOW_ALBEDO, "tile_clear": 0.75, "tile_glow": 0.06, "tile_reflect": 0.04,
          "tile_absorb": 0.0, "exposure": 1.0, "tile_rim": 0.0, "tile_ior": 1.5, "tone_knee": 0.8}
TUNED = {"gi_gain": 2.0, "ambient": 0.3, "floor_lightness": 0.7, "floor_saturation": 0.9, "shape_glow": 2.5,
         "shape_albedo": 0.8, "tile_clear": 0.3, "tile_glow": 0.3, "tile_reflect": 0.6, "tile_absorb": 0.1,
         "exposure": 1.6, "tile_rim": 1.0, "tile_ior": 2.2, "tone_knee": 0.3}
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def build():
    world = PRESETS[MAP](SIZE[0] / ZOOM, SIZE[1] / ZOOM)
    for _ in range(5):
        world.step(1 / 60)
    return world


def positions(world):
    return np.concatenate([np.array(b.points, float).ravel() for b in world.bodies])


class Renderer:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=410)
        self.gw = GLWorld(self.ctx, SIZE)

    def __call__(self, world, filled=True, outlines=True):
        for _ in range(4):  # (GI settles over a few frames)
            self.gw.render(world, SIZE, filled, ZOOM, world.background, outlines=outlines, lighting=True)
        self.ctx.finish()
        return np.frombuffer(self.gw.tex.read(), np.uint8).reshape(SIZE[1], SIZE[0], 4)[..., :3].astype(np.int16)


def diff(a, b):
    return float(np.abs(a - b).mean())


def main(argv):
    png = argv[argv.index("--png") + 1] if "--png" in argv else None
    tmp = tempfile.mkdtemp(prefix="visual_tune_")
    path = os.path.join(tmp, "visual_overrides.json")
    render = Renderer()

    # 1. defaults
    world = build()
    check(getattr(world, "map_key", None) == MAP, f"map_key {getattr(world, 'map_key', None)!r}")
    VT.attach(world, path)
    d = world.look_defaults
    check(all(abs(d.get(k, -1) - v) < 1e-6 for k, v in EXPECT.items()), f"defaults from the map: {d}")

    # 2. every slider changes the image, in every mode, physics untouched
    pos0 = positions(world)
    for filled, outlines, mode in ((True, True, "filled"), (True, False, "borderless"), (False, True, "hollow")):
        base = render(world, filled, outlines)
        moved = []
        for key, value in TUNED.items():
            VT.set_value(world, key, value)
            img = render(world, filled, outlines)
            VT.set_value(world, key, d[key])
            moved.append((key, diff(img, base)))
        back = diff(render(world, filled, outlines), base)
        print(f"     {mode}: " + ", ".join(f"{k} {v:.2f}" for k, v in moved) + f"; restored {back:.3f}")
        # (the hollow view draws only outlines and points: the floor, ambient and GI still show; tiles' and shapes'
        # material fields need filled bodies)
        needed = [k for k in TUNED if filled or k in ("gi_gain", "ambient", "floor_lightness", "floor_saturation")]
        check(all(v > 0.05 for k, v in moved if k in needed), f"{mode}: every slider changes the image")
        check(back < 0.05, f"{mode}: back to the map's values gives the same image")
    check(np.array_equal(positions(world), pos0), "bodies' positions untouched")

    # 3. palette switch keeps the values
    VT.apply(world, TUNED)
    tuned_img = render(world)
    if png:
        os.makedirs(png, exist_ok=True)
        pygame.image.save(pygame.surfarray.make_surface(render(world).astype(np.uint8).swapaxes(0, 1)),
                          os.path.join(png, "tuned.png"))
    name = world.palette
    other = next(n for n in palettes.PALETTES if n != name)
    palettes.apply_palette(world, other)
    v = VT.values(world)
    check(all(abs(v[k] - TUNED[k]) < 1e-6 for k in TUNED), f"values kept on a switch to {other!r}: {v}")
    palettes.apply_palette(world, name)
    check(diff(render(world), tuned_img) < 0.05, f"same image after switching back to {name!r}")

    # 4. save / reload
    VT.save(world.map_key, VT.changed(world), path)
    fresh = build()
    VT.attach(fresh, path)
    v = VT.values(fresh)
    check(all(abs(v[k] - TUNED[k]) < 1e-6 for k in TUNED), f"saved values applied on reload: {v}")
    check(all(abs(fresh.look_defaults[k] - EXPECT[k]) < 1e-6 for k in EXPECT), "reload keeps the map's defaults")
    check(diff(render(fresh), tuned_img) < 0.05, "reloaded render matches the tuned one")
    VT.LookCard(lambda: type("E", (), {"world": fresh})(), path).reset()
    VT.save(fresh.map_key, VT.changed(fresh), path)
    check(MAP not in VT.load_all(path), "reset all + save removes the map's entry")
    with open(path, "w") as f:
        f.write("{not json")
    w2 = build()
    VT.attach(w2, path)
    check(all(abs(VT.values(w2)[k] - EXPECT[k]) < 1e-6 for k in EXPECT), "corrupt file: the map's values")
    with open(path, "w") as f:
        f.write('{"%s": {"gi_gain": "x", "tile_clear": 9, "bogus": 1}}' % MAP)
    w3 = build()
    VT.attach(w3, path)
    check(VT.get(w3, "gi_gain") == 6.4 and VT.get(w3, "tile_clear") == 1.0, "bad entries skipped, out of range clamped")
    w4 = build()
    VT.attach(w4, os.path.join(tmp, "missing", "none.json"))
    check(VT.get(w4, "gi_gain") == 6.4, "missing file: the map's values")

    # 5. the card
    ed = type("E", (), {"world": w4})()
    card = VT.LookCard(lambda: ed, path)
    card.place((0, 0, HUD.px(150), card.card_height()))
    surf = pygame.Surface((card.rect.w, card.rect.h), pygame.SRCALPHA)
    card.draw_card(surf, (0, 0), (0, 0))
    slider = next((r, s) for r, s in card.widgets() if s[0] == "slider" and s[1][0] == "tile_clear")
    track = card._track(slider[0])
    card.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(track.x + track.w // 4, track.centery), button=1))
    card.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(track.x + track.w // 4, track.centery), button=1))
    check(abs(VT.get(w4, "tile_clear") - 0.25) < 0.02, f"slider drag sets tile clear ({VT.get(w4, 'tile_clear')})")
    check(card.status() == "edited", "status: edited")
    card.draw_card(surf, (0, 0), (0, 0))
    card.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(slider[0].x + 2, slider[0].centery), button=1))
    check(abs(VT.get(w4, "tile_clear") - 0.75) < 1e-6, "label click resets it")
    if png:
        pygame.image.save(surf, os.path.join(png, "card.png"))
    print("all visual_tune checks passed" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
