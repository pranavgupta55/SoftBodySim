"""Sidebar checks: collapsible cards, a fixed legible scale, scrolling with momentum and rubber-band ends. Run from
the repo root:

    .venv/bin/python3 tests/sidebar_check.py [--shots DIR]

1. Scroll physics (hud_layer.HudLayer with stand-in cards, driven with explicit times): one wheel notch coasts and
   decays to rest; the coast is the same at 30 and 144 fps; past either end the overscroll stays within its cap and
   springs back to exactly the end; the layer redraws every frame while moving and never once at rest.
2. App (offscreen main_opengl.main, 1512x982 window, penrose/evolved 1): the defaults (Controls and Worlds open;
   Compute, Gravity, Palette and Lights collapsed; the Material card opens when it appears), header clicks open and
   close a card, the scale stays at the legible width scale with every card open (the sidebar scrolls instead), the
   wheel over the sidebar scrolls it (and leaves the grab circle alone), the wheel over the world still resizes the
   grab circle, and clicks on a card's controls hit them during and after a scroll. With --shots, PNGs of the sidebar
   (default, all open at the top, all open at the bottom) go to DIR.
Exit status 1 on any failure.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import pygame  # noqa: E402

pygame.init()

import hud_layer  # noqa: E402
import ui_style as ui  # noqa: E402
from entities import EnvConfig  # noqa: E402
from scene import ViewConfig  # noqa: E402

FAILS = []
SIZE = (1512, 982)


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


class Stub:
    """A stand-in card: a fixed height, a flat fill."""

    def __init__(self, title, h):
        self.card_title, self.h, self.rect = title, h, None

    def card_height(self, compact=False):
        return self.h

    def place(self, rect):
        self.rect = rect

    def draw_card(self, surface, origin, mouse_px, compact=False):
        ui.HUD.draw_frame(surface, self.rect.move(-origin[0], -origin[1]))


# ---------------- 1. scroll physics
def make_layer():
    hud = hud_layer.HudLayer(SIZE, 2, 0.1, [Stub("Controls", 700), Stub("Worlds", 900)])
    hud.draw((0, 0), key="k", now=0.0)
    return hud


def run(hud, t, until, fps=60, key="k"):
    """Draw frames from time t to `until`; returns (t, frames redrawn, frames at rest that redrew)."""
    redrawn = rest_redraws = 0
    while t < until - 1e-9:
        t += 1 / fps
        was_moving = hud.moving
        hud.draw((0, 0), key=key, now=t)
        redrawn += hud.dirty is not None
        rest_redraws += hud.dirty is not None and not was_moving
    return t, redrawn, rest_redraws


def check_physics():
    hud = make_layer()
    hi = hud.max_scroll()
    check(hi > 0, f"stand-in cards overflow the sidebar (max scroll {hi})")
    hud.wheel(-1)  # one notch down
    t, redrawn, _ = run(hud, 0.0, 0.25)
    check(redrawn == 15 and hud.moving, f"redrawn every frame while moving ({redrawn}/15)")
    t, _, _ = run(hud, t, 2.0)
    step = hud_layer.WHEEL_STEP * ui.HUD.scale
    check(not hud.moving and hud.velocity == 0.0, "momentum decays to rest")
    check(abs(hud.scroll - step) < 2.0, f"one notch coasts ~{step:.0f} px ({hud.scroll:.1f})")
    _, redrawn, _ = run(hud, t, t + 1.0)
    check(redrawn == 0, f"no redraws once at rest ({redrawn} in 1 s)")

    rests = []
    for fps in (30, 144):  # frame-rate independent
        h = make_layer()
        h.wheel(-2)
        run(h, 0.0, 2.0, fps)
        rests.append(h.scroll)
    check(abs(rests[0] - rests[1]) < 1.0, f"same coast at 30 and 144 fps ({rests[0]:.2f} vs {rests[1]:.2f})")

    cap = hud_layer.OVERSCROLL_FRAC * hud.rect.h
    for name, notches, end in (("bottom", -40, hi), ("top", 40, 0.0)):
        h = make_layer()
        h.scroll = float(hi) if end else 0.0
        h.draw((0, 0), key="k", now=0.0)
        t, peak = 0.0, 0.0
        for k in range(10):  # a hard flick past the end
            h.wheel(notches / 10)
            t, _, _ = run(h, t, t + 1 / 60)
            peak = max(peak, abs(h.scroll - end))
        for _ in range(60):
            t, _, _ = run(h, t, t + 1 / 60)
            peak = max(peak, abs(h.scroll - end))
        check(0 < peak <= cap + 1e-6, f"{name}: overscroll within the cap ({peak:.1f} of {cap:.1f} px)")
        t, _, _ = run(h, t, t + 1.5)
        check(h.scroll == end and h.velocity == 0.0 and not h.moving,
              f"{name}: springs back to exactly the end ({h.scroll!r})")
        _, redrawn, rest = run(h, t, t + 1.0)
        check(redrawn == 0, f"{name}: no redraws at rest ({redrawn})")

    # a spring-back from overscroll settles in about a quarter second
    h = make_layer()
    h.scroll = float(hi) + cap
    t, settle = 0.0, None
    for k in range(120):
        t, _, _ = run(h, t, t + 1 / 120, fps=120)
        if settle is None and abs(h.scroll - hi) < 1.0:
            settle = t
    check(settle is not None and 0.1 < settle < 0.4, f"spring-back takes ~0.25 s ({settle})")


# ---------------- 2. app
def check_app(shots):
    import moderngl
    import main_opengl
    ctx = moderngl.create_standalone_context(require=330)
    fbo = ctx.simple_framebuffer(SIZE)
    fbo.use()
    display = main_opengl.Display(ctx, SIZE, SIZE, lambda: None)
    seen = {"script": []}
    real_get_pos = pygame.mouse.get_pos
    mouse = {"pos": (300, 300)}
    pygame.mouse.get_pos = lambda: mouse["pos"]
    state = {"phase": "start", "wait": 0}

    def post(kind, **kw):
        pygame.event.post(pygame.event.Event(kind, **kw))

    def click(pos):
        post(pygame.MOUSEBUTTONDOWN, pos=pos, button=1)
        post(pygame.MOUSEBUTTONUP, pos=pos, button=1)

    def key(k):
        post(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0)
        post(pygame.KEYUP, key=k, mod=0, unicode="", scancode=0)

    def wheel(y):
        post(pygame.MOUSEWHEEL, x=0, y=int(y), precise_x=0.0, precise_y=float(y), flipped=False, which=0)

    def shot(hud, name):
        if shots:
            path = os.path.join(shots, f"sidebar_{name}.png")
            pygame.image.save(hud.surface, path)
            seen["script"].append(path)

    def header(hud, card):
        r, o = next((r, o) for c, r, o in hud.placed if c is card)
        return hud.header_rect(r, o).center

    def opened(hud):
        return {hud_layer.card_title(c): o for c, r, o in hud.placed}

    def on_frame(i, loc):
        hud, editor, world = loc["hud"], loc["editor"], loc["world"]
        gravity = loc["gravity_dial"]
        lights_card, material_card = loc["lights_card"], loc["material_card"]
        legible = min(max(hud.rect.w / (ui.CARD_WIDTH_BASE + 8), ui.HUD_MIN_SCALE), ui.HUD_MAX_SCALE)
        ph = state["phase"]
        if state["wait"] > 0:
            state["wait"] -= 1
            return
        if ph == "start" and i == 0:
            seen["full"] = loc["full"]
            if loc["full"]:
                key(pygame.K_h)  # a fullscreen world: its floating panel holds the sidebar
            state.update(wait=2)
        elif ph == "start":
            check(loc["sidebar_shown"], f"the sidebar is shown (fullscreen world: {seen['full']}, H panel)")
            seen["defaults"] = opened(hud)
            seen["scale0"] = ui.HUD.scale
            check(abs(ui.HUD.scale - legible) < 1e-9 and ui.HUD.scale >= ui.HUD_MIN_SCALE,
                  f"legible scale {ui.HUD.scale:.3f} (width scale {legible:.3f})")
            shot(hud, "default")
            if not ViewConfig.lighting:
                key(pygame.K_l)  # lighting on: the Lights card
            state.update(phase="lit", wait=2)
        elif ph == "lit":
            d = opened(hud)
            check(d.get("Lights") is False, f"Lights card shown, collapsed ({d.get('Lights')})")
            click(header(hud, gravity))
            state.update(phase="grav_open", wait=1)
        elif ph == "grav_open":
            check(opened(hud).get("Gravity") is True, "a header click opens Gravity")
            g = gravity.rect
            check(hud.rect.contains(g) and g.h == gravity.card_height(), "the open card is placed in the sidebar")
            click(header(hud, gravity))
            state.update(phase="grav_closed", wait=1)
        elif ph == "grav_closed":
            check(opened(hud).get("Gravity") is False, "a second click closes it")
            check(gravity.rect.x < -1000, "a collapsed card's widgets are off-screen (no clicks reach them)")
            # a click on the collapsed header's body area must not reach the card below
            key(pygame.K_SPACE)  # pause, then pick bodies: the Material card
            state.update(phase="pick", wait=1)
        elif ph == "pick":
            bodies = [b for b in world.bodies if not editor._is_wall(b) and not b.fixed][:2]
            editor.picked = list(bodies)
            state.update(phase="material", wait=2)
        elif ph == "material":
            d = opened(hud)
            check(d.get("Material") is True, f"the Material card opens when it appears ({d.get('Material')})")
            for c in hud.cards:
                hud.set_open(c, True)
            state.update(phase="all_open", wait=2)
        elif ph == "all_open":
            check(all(opened(hud).values()), "every card open")
            check(ui.HUD.scale == seen["scale0"], f"scale unchanged with every card open ({ui.HUD.scale:.3f})")
            check(hud.content_h > hud.rect.h, f"the cards overflow ({hud.content_h} > {hud.rect.h}): scrolling")
            shot(hud, "all_open_top")
            seen["radius"] = ViewConfig.grab_area_radius
            ViewConfig.grab_area = True
            mouse["pos"] = (hud.rect.centerx, hud.rect.centery)
            for _ in range(3):
                wheel(-1)
            state.update(phase="moving", wait=1)
        elif ph == "moving":
            check(hud.moving and hud.scroll > 0, f"the wheel over the sidebar scrolls it ({hud.scroll:.1f})")
            check(ViewConfig.grab_area_radius == seen["radius"], "... and leaves the grab circle alone")
            # click a gravity direction while the sidebar moves: the current offset
            from gravity_dial import direction_index
            target = next(k for k in (2, 6, 0, 4) if k != direction_index())
            c = gravity.button_center(target)
            if hud.rect.collidepoint(c):
                seen["g_before"] = EnvConfig.gravity.copy()
                seen["g_target"] = target
                click(c)
            state.update(phase="moving_click")
        elif ph == "moving_click":
            from gravity_dial import direction_index
            if "g_target" in seen:
                check(direction_index() == seen["g_target"], f"a click during the motion hits its button "
                                                            f"(gravity {direction_index()} / {seen['g_target']})")
            else:
                check(False, "gravity button on screen while moving")
            for _ in range(60):
                wheel(-1)  # to the bottom, and past it
            state.update(phase="bottom")
        elif ph == "bottom":
            seen["over"] = max(seen.get("over", 0), hud.scroll - hud.max_scroll())
            if not hud.moving:
                check(hud.scroll == hud.max_scroll(), f"at rest exactly at the bottom ({hud.scroll} / {hud.max_scroll()})")
                check(seen["over"] > 0, f"the bottom rubber-bands ({seen['over']:.1f} px past it)")
                shot(hud, "all_open_bottom")
                # a click after the motion: the Lights card's +point button (near the bottom)
                rect = next(r for r, spec in lights_card.widgets() if spec[0] == "button" and spec[1] == "+point")
                seen["n_lights"] = len(getattr(world, "lights", None) or [])
                if hud.rect.contains(rect):
                    click(rect.center)
                else:  # the Palette card's first button instead
                    seen["n_lights"] = None
                state.update(phase="after_click", wait=1)
        elif ph == "after_click":
            if seen["n_lights"] is not None:
                check(len(getattr(world, "lights", None) or []) == seen["n_lights"] + 1, "a click after scrolling hits +point")
            else:
                check(False, "+point on screen at the bottom")
            mouse["pos"] = (200, 200)  # over the world: the wheel resizes the grab circle again
            wheel(1)
            state.update(phase="world_wheel", wait=1)
        elif ph == "world_wheel":
            check(ViewConfig.grab_area_radius > seen["radius"], "the wheel over the world still resizes the grab circle")
            ViewConfig.grab_area = False
            ViewConfig.grab_area_radius = seen["radius"]
            state.update(phase="done")
            loc_done["done"] = True

    loc_done = {"done": False}
    try:
        main_opengl.main(["--scene", "penrose/evolved 1"], display=display, max_frames=600, on_frame=on_frame)
    finally:
        pygame.mouse.get_pos = real_get_pos
    d = seen.get("defaults", {})
    want = {"Controls": True, "Worlds": True, "Compute": False, "Gravity": False, "Palette": False}
    check(all(d.get(k) == v for k, v in want.items()), f"defaults {d}")
    check(loc_done["done"], f"the app script ran to the end (stopped at {state['phase']})")
    for p in seen["script"]:
        print("shot", p)


def main(argv):
    shots = None
    if "--shots" in argv:
        shots = argv[argv.index("--shots") + 1]
        os.makedirs(shots, exist_ok=True)
    for f in (check_physics, lambda: check_app(shots)):
        try:
            f()
        except Exception as e:  # report and go on
            import traceback
            traceback.print_exc()
            check(False, f"raised {type(e).__name__}: {e}")
    print(f"{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
