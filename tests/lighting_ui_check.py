"""Lighting data and UI checks (research/lighting_contract.md, workstream U). Run from the repo root:

    .venv/bin/python3 tests/lighting_ui_check.py [frames]

1. Save / load: the lighting demo's materials, overrides, lights, ambient and lighting switch survive scene_data ->
   JSON -> load_data exactly; the shipped scenes (older files, no lighting data) load and re-save without any.
2. Material card: presets, slider drags, colours and reset on picked bodies, each one undo step (undo / redo
   restore them); copy / duplicate keep materials; a spawned body gets the card's last material; palettes' role
   materials go only to bodies without their own.
3. Lights card: add, select, slider drag, gizmo drag and aim, delete, ambient, each undoable.
4. The "lighting demo" preset is intersection-free and steps 120 ticks headless (several world sizes).
5. App smoke (offscreen main_opengl.main, `frames` frames, default 160): the demo starts lit, L toggles lighting off
   and on, H opens the panel, both cards appear, a click on "+point" adds a light, a posted mouse drag moves it,
   a Material preset button sets the picked bodies' material.
Exit status 1 on any failure.
"""
import copy
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import pygame  # noqa: E402

pygame.init()

import lights as L  # noqa: E402
import materials as M  # noqa: E402
import palettes  # noqa: E402
import scene_io  # noqa: E402
from editor import Editor  # noqa: E402
from entities import polygon_points  # noqa: E402
from lighting_ui import LightsCard, MaterialCard  # noqa: E402
from presets_basic import validate_layout  # noqa: E402
from scene import INSET, PRESETS, Endesga, ViewConfig  # noqa: E402
from solver import World  # noqa: E402
from ui_style import HUD  # noqa: E402

BOX = (1512.0, 982.0, float(INSET))
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def materials_of(world):
    return [(getattr(b, "material", None), dict(getattr(b, "material_over", None) or {})) for b in world.bodies]


def lights_of(world):
    return [light.to_dict() for light in getattr(world, "lights", None) or []]


def norm(v):
    """Tuples and lists alike (JSON turns tuples into lists)."""
    return json.loads(json.dumps(v))


# ---------------- 1. save / load
def check_round_trip():
    world = PRESETS["lighting demo"](*BOX[:2])
    data = json.loads(scene_io.dumps(scene_io.scene_data(world, BOX, ViewConfig, "demo")))
    loaded = World()
    scene_io.load_data(loaded, data, BOX)
    check(norm(materials_of(loaded)) == norm(materials_of(world)), "materials and overrides round-trip")
    check(lights_of(loaded) == lights_of(world) and len(lights_of(world)) == 4, "lights round-trip")
    check(tuple(loaded.ambient) == tuple(world.ambient), "ambient round-trips")
    check(getattr(loaded, "lighting", None) is True, "lighting switch round-trips")
    again = json.loads(scene_io.dumps(scene_io.scene_data(loaded, BOX, ViewConfig, "demo")))
    check(again == data, "save -> load -> save is stable")
    check(all(isinstance(light, L.Light) for light in loaded.lights), "loaded lights are Light objects")
    # another box: lights move with the scene
    other = World()
    scene_io.load_data(other, data, (BOX[0] + 200, BOX[1], BOX[2]))
    check(abs(other.lights[0].pos[0] - world.lights[0].pos[0] - 100) < 1e-9, "lights move with a re-centred scene")
    # older files: no lighting data in, none out, and loading one clears the previous scene's lights
    for name in scene_io.list_scenes():
        path = scene_io.scene_path(name)
        old = scene_io.read_file(path)
        w = World()
        w.lights, w.ambient = [L.Light()], (1.0, 0.0, 0.0)
        box = tuple(float(old["world"][k]) for k in ("width", "height", "inset"))
        scene_io.load_data(w, old, box)
        out = scene_io.capture(w, box)
        flat = json.dumps(out)
        check("material" not in flat and "lights" not in flat and "ambient" not in flat and w.lights == []
              and w.ambient is None and all(b.material is None and b.material_over == {} for b in w.bodies),
              f"old scene {name} loads without lighting data")
    # bad lights are refused
    bad = copy.deepcopy(data)
    bad["state"]["lights"][0]["pos"] = [1.0]
    try:
        scene_io.load_data(World(), bad, BOX)
        check(False, "a malformed light is refused")
    except scene_io.SceneError:
        check(True, "a malformed light is refused")


# ---------------- 2 / 3. cards, undo
def make_editor(world):
    HUD.set_scale(1.0)
    ed = Editor(world, Endesga, (756, 491), (BOX[0], BOX[1]), 2, INSET, ViewConfig.zoom, view=ViewConfig,
                screen_size=(1512, 982))
    ed.paused = True
    return ed


def mouse(kind, pos, button=1):
    if kind == pygame.MOUSEMOTION:
        return pygame.event.Event(kind, pos=pos, rel=(0, 0), buttons=(1, 0, 0))
    return pygame.event.Event(kind, pos=pos, button=button)


def slider_row(card, field):
    return next(rect for rect, spec in card.widgets() if spec[0] == "slider" and spec[1][0] == field)


def drag_slider(card, field, frac):
    row = slider_row(card, field)
    track = card._track(row)
    y = row.centery
    card.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (track.x + 1, y)))
    card.handle_event(mouse(pygame.MOUSEMOTION, (track.x + int(track.w * frac / 2), y)))
    card.handle_event(mouse(pygame.MOUSEMOTION, (track.x + int(track.w * frac), y)))
    card.handle_event(mouse(pygame.MOUSEBUTTONUP, (track.x + int(track.w * frac), y)))


def click_button(card, label):
    rect = next(rect for rect, spec in card.widgets() if spec[0] in ("button", "chip") and spec[1] == label)
    return card.handle_event(mouse(pygame.MOUSEBUTTONDOWN, rect.center))


def check_material_card():
    world = PRESETS["default"](*BOX[:2])
    ed = make_editor(world)
    card = MaterialCard(lambda: ed)
    card.place(pygame.Rect(1300, 10, 160, card.card_height()))
    movable = [b for b in world.bodies if not ed._is_wall(b) and not b.fixed]
    picked = movable[:2]
    ed.picked = list(picked)
    check(card.bodies() == [b for b in world.bodies if b in picked], "card acts on the picked bodies")
    before = materials_of(world)
    steps = len(ed.history.undo_stack)
    click_button(card, "glass")
    check(all(b.material == "glass" and b.material_over == {} for b in picked), "preset button sets the material")
    drag_slider(card, "roughness", 0.5)
    check(all(abs(b.material_over.get("roughness", -1) - 0.5) < 0.051 for b in picked), "slider drag overrides a field")
    swatch = next(rect for rect, spec in card.widgets() if spec[0] == "swatch")
    card.handle_event(mouse(pygame.MOUSEBUTTONDOWN, swatch.center))
    check(all("tint" in b.material_over and M.of_body(b).tint_strength > 0 for b in picked), "swatch sets the tint")
    click_button(card, "glow colour")
    card.handle_event(mouse(pygame.MOUSEBUTTONDOWN, swatch.center))
    check(all(M.of_body(b).emissive_strength > 0 for b in picked), "glow target sets the emissive colour")
    check(len(ed.history.undo_stack) - steps == 4, "preset, drag, two colours: four undo steps "
                                                    f"({len(ed.history.undo_stack) - steps})")
    after = materials_of(world)
    for _ in range(4):
        ed.world_undo()
    check(materials_of(world) == before, "undo restores the materials")
    for _ in range(4):
        ed.world_redo()
    check(norm(materials_of(world)) == norm(after), "redo brings them back")
    picked = [world.bodies[k] for k, (m, _) in enumerate(materials_of(world)) if m == "glass"]
    ed.picked = list(picked)
    click_button(card, "reset")
    check(all(b.material == "glass" and b.material_over == {} for b in picked), "reset clears the overrides only")
    # copy / duplicate keep materials
    picked[0].material_over = {"specular": 0.3}
    ed.picked = [picked[0]]
    ed.copy_selection()
    clip_bodies = ed.clipboard[0]
    check(clip_bodies[0].material == "glass" and clip_bodies[0].material_over == {"specular": 0.3}
          and clip_bodies[0].material_over is not picked[0].material_over, "copy keeps the material (own dict)")
    n = len(world.bodies)
    ed.duplicate_selection()
    if ed.paste is not None:  # no room nearby: place the ghost somewhere free
        ed._place_ghost(ed._ghost_new(np.array([BOX[0] * 0.5, BOX[1] * 0.15]))[0])
    new = world.bodies[n:] if len(world.bodies) > n else []
    check(len(new) >= 1 and all(b.material == "glass" and b.material_over == {"specular": 0.3} for b in new),
          "duplicate keeps the material")
    # spawning: the card's last material
    ed.picked = [picked[0]]
    click_button(card, "metal")
    ed.picked = []
    ed.kind, ed.mode, ed.relation = "soft", "regular", "free"
    ed.points = [np.array(p) for p in polygon_points(BOX[0] * 0.2, BOX[1] * 0.12, 20, 6).T]
    n = len(world.bodies)
    ed._spawn()
    check(len(world.bodies) == n + 1 and world.bodies[-1].material == "metal", "a spawned body gets the material")
    ed.kind = ed.mode = None
    ed.points = []
    # palettes: role materials only for bodies without their own
    palettes.PALETTES["_test"] = dict(palettes.PALETTES["forest"], materials={"hexagon": "jelly"})
    try:
        dw = PRESETS["penrose/evolved 1"](*BOX[:2]) if "penrose/evolved 1" in PRESETS else None
        if dw is not None:
            shapes = [b for b in dw.bodies if getattr(b, "palette_role", None) == "hexagon"]
            shapes[0].material = "metal"
            palettes.apply_palette(dw, "_test")
            check(shapes[0].material == "metal" and all(b.material == "jelly" for b in shapes[1:]),
                  "palette materials skip bodies with their own")
            palettes.apply_palette(dw, "forest")
            check(shapes[0].material == "metal" and all(b.material is None for b in shapes[1:]),
                  "switching palettes takes its materials back")
    finally:
        del palettes.PALETTES["_test"]


def check_lights_card():
    world = PRESETS["default"](*BOX[:2])
    world.lighting = True
    ed = make_editor(world)
    card = LightsCard(lambda: ed, ViewConfig)
    card.place(pygame.Rect(1300, 10, 160, card.card_height()))
    steps = len(ed.history.undo_stack)
    for label in ("+point", "+spot", "+sun"):
        click_button(card, label)
    check([light.kind for light in world.lights] == [L.POINT, L.SPOT, L.DIRECTIONAL] and card.selected == 2,
          "add buttons add point / spot / sun, the new one selected")
    click_button(card, "<")
    check(card.selected == 1, "< selects the previous light")
    drag_slider(card, "cone_outer", 0.5)
    check(abs(world.lights[1].cone_outer - 45) < 1.5, f"cone slider ({world.lights[1].cone_outer})")
    click_button(card, "shadows")
    check(world.lights[1].casts_shadows is False, "shadows switch")
    swatch = next(rect for rect, spec in card.widgets() if spec[0] == "swatch" and spec[1] == tuple(Endesga.cyan))
    card.handle_event(mouse(pygame.MOUSEBUTTONDOWN, swatch.center))
    check(max(world.lights[1].color) == 1.0 and world.lights[1].color[0] < 0.5, "swatch sets the light colour")
    drag_slider(card, "ambient", 0.5)
    check(abs(max(world.ambient) - 0.5) < 0.03, "ambient strength slider")
    # gizmo drag (screen px = world units x zoom x scale) and aim
    z, s = ed.zoom, ed.scale
    light = world.lights[0]
    p0 = (round(light.pos[0] * z * s), round(light.pos[1] * z * s))
    used = card.handle_world_event(mouse(pygame.MOUSEBUTTONDOWN, p0))
    card.handle_world_event(mouse(pygame.MOUSEMOTION, (p0[0] - 60, p0[1] + 40)))
    card.handle_world_event(mouse(pygame.MOUSEBUTTONUP, (p0[0] - 60, p0[1] + 40)))
    check(used and card.selected == 0 and abs(world.lights[0].pos[0] - (light.pos[0])) < 1e-9
          and abs(world.lights[0].pos[0] - (p0[0] - 60) / (z * s)) < 1.5, "gizmo drag moves the light")
    card.select(1)
    aim = card._aim_point(world.lights[1])
    a0 = (round(aim[0] * z * s), round(aim[1] * z * s))
    spot = world.lights[1]
    card.handle_world_event(mouse(pygame.MOUSEBUTTONDOWN, a0))
    target = (round(spot.pos[0] * z * s) + 100, round(spot.pos[1] * z * s))
    card.handle_world_event(mouse(pygame.MOUSEMOTION, target))
    card.handle_world_event(mouse(pygame.MOUSEBUTTONUP, target))
    check(abs(world.lights[1].dir[0] - 1.0) < 1e-6, f"aim handle turns the spot ({world.lights[1].dir})")
    miss = card.handle_world_event(mouse(pygame.MOUSEBUTTONDOWN, (5, 5)))
    check(miss is False, "a press away from the gizmos isn't used")
    card.handle_world_event(mouse(pygame.MOUSEBUTTONUP, (5, 5)))
    del_key = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE, mod=0, unicode="", scancode=0)
    card.handle_world_event(del_key)
    check(len(world.lights) == 2 and card.selected is None, "Delete removes the selected light")
    n_steps = len(ed.history.undo_stack) - steps
    check(n_steps == 10, f"ten undo steps (3 adds, slider, switch, colour, ambient, move, aim, delete): {n_steps}")
    for _ in range(n_steps):
        ed.world_undo()
    check(getattr(world, "lights", []) == [] and getattr(world, "ambient", None) is None, "undo removes them all")
    check(getattr(world, "lighting", None) is True, "undo leaves the lighting switch alone")
    for _ in range(n_steps):
        ed.world_redo()
    check(len(world.lights) == 2 and abs(max(world.ambient) - 0.5) < 0.03, "redo brings them back")
    surf = pygame.Surface((756, 491), pygame.SRCALPHA)
    card.select(0)
    card.draw_world(surf)
    card.draw_card(pygame.Surface((1512, 982), pygame.SRCALPHA), (0, 0), (0, 0))
    check(surf.get_bounding_rect().w > 0, "gizmos draw")


# ---------------- 4. the demo preset
def check_demo():
    for size in ((1512, 982), (1280, 760), (2400, 1500)):
        w = PRESETS["lighting demo"](*size)
        try:
            validate_layout(w)
            ok = True
        except ValueError as e:
            ok = str(e)
        movable = sum(b.points.shape[1] for b in w.bodies if not b.fixed)
        check(ok is True, f"demo {size} intersection-free ({ok}), {movable} movable points")
        presets = {getattr(b, "material", None) for b in w.bodies}
        check(set(M.PRESETS) <= presets, f"demo {size} shows every preset")
        y0 = np.mean([b.points[1].mean() for b in w.bodies if not b.fixed])
        for _ in range(120):
            w.step(1 / 240)
        y1 = np.mean([b.points[1].mean() for b in w.bodies if not b.fixed])
        check(w.ticks == 120 and np.isfinite(w.P).all() and y1 > y0, f"demo {size} steps 120 ticks and falls")
        check(w.lighting is True and len(w.lights) == 4 and {lt.kind for lt in w.lights} == {L.POINT, L.SPOT, L.DIRECTIONAL},
              f"demo {size} lights")


# ---------------- 5. app smoke
def check_app(frames):
    import moderngl
    import main_opengl
    size = (1512, 982)
    ctx = moderngl.create_standalone_context(require=330)
    fbo = ctx.simple_framebuffer(size)
    fbo.use()
    display = main_opengl.Display(ctx, size, size, lambda: None)
    seen = {}

    def key(k):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0))
        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=k, mod=0, unicode="", scancode=0))

    def post(kind, pos):
        pygame.event.post(mouse(kind, pos))

    def on_frame(i, loc):
        world, editor, hud = loc["world"], loc["editor"], loc["hud"]
        lights_card, material_card = loc["lights_card"], loc["material_card"]
        if i == 0:
            seen["lit_at_start"] = ViewConfig.lighting and world.lighting
            key(pygame.K_l)  # off
        elif i == 3:
            seen["off"] = not ViewConfig.lighting and lights_card not in hud.cards
            key(pygame.K_l)  # on again
            key(pygame.K_h)  # the panel
            hud.set_open(lights_card)  # (the sidebar starts with it collapsed, hud_layer.DEFAULT_OPEN)
        elif i == 6:
            seen["cards_on"] = lights_card in hud.cards and ViewConfig.lighting
            rect = next(r for r, spec in lights_card.widgets() if spec[0] == "button" and spec[1] == "+point")
            seen["n_lights"] = len(world.lights)
            post(pygame.MOUSEBUTTONDOWN, rect.center)
            post(pygame.MOUSEBUTTONUP, rect.center)
        elif i == 8:
            seen["added"] = len(world.lights) == seen["n_lights"] + 1
            light = world.lights[-1]
            seen["pos0"] = light.pos
            z, s = editor.zoom, editor.scale
            p = (round(light.pos[0] * z * s), round(light.pos[1] * z * s))
            post(pygame.MOUSEBUTTONDOWN, p)
            post(pygame.MOUSEMOTION, (p[0] - 100, p[1] - 50))
            post(pygame.MOUSEBUTTONUP, (p[0] - 100, p[1] - 50))
        elif i == 10:
            seen["dragged"] = abs(world.lights[-1].pos[0] - seen["pos0"][0] + 100 / (editor.zoom * editor.scale)) < 1.5
            key(pygame.K_SPACE)  # pause, then pick two bodies
        elif i == 12:
            bodies = [b for b in world.bodies if not editor._is_wall(b) and not b.fixed][:2]
            editor.picked = list(bodies)
            seen["picked"] = bodies
        elif i == 14:
            seen["material_card"] = material_card in hud.cards
            if seen["material_card"]:
                rect = next(r for r, spec in material_card.widgets() if spec[0] == "button" and spec[1] == "toon")
                post(pygame.MOUSEBUTTONDOWN, rect.center)
                post(pygame.MOUSEBUTTONUP, rect.center)
        elif i == 16:
            seen["toon"] = all(b.material == "toon" for b in seen["picked"])
            key(pygame.K_SPACE)  # run again
            key(pygame.K_l)  # lighting off for the rest
        elif i == frames - 1:
            seen["ticks"] = world.ticks

    main_opengl.main(["--scene", "lighting demo"], display=display, max_frames=frames, on_frame=on_frame)
    check(seen.get("lit_at_start"), "the demo starts lit")
    check(seen.get("off"), "L turns lighting off (Lights card gone)")
    check(seen.get("cards_on"), "L turns it back on, Lights card shown")
    check(seen.get("added"), "+point adds a light")
    check(seen.get("dragged"), "a mouse drag moves the light")
    check(seen.get("material_card"), "the Material card shows for picked bodies")
    check(seen.get("toon"), "its preset button sets the material")
    check(seen.get("ticks", 0) > 0, f"the world ran ({seen.get('ticks')} ticks)")


def main(argv):
    frames = int(argv[0]) if argv else 160
    for f in (check_round_trip, check_material_card, check_lights_card, check_demo, lambda: check_app(frames)):
        try:
            f()
        except Exception as e:  # report and go on
            import traceback
            traceback.print_exc()
            check(False, f"{getattr(f, '__name__', 'app')} raised {type(e).__name__}: {e}")
    print(f"{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
