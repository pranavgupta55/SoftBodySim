"""The game: scene, physics, editor and HUD, presented through moderngl.

Ported from Framework/mainScreenOpenGL.py: the game still draws into low-res pygame surfaces, which are uploaded to
GPU textures every frame and drawn with a fullscreen quad. On top of that port, the three layers (world, translucent
editor ghost, UI) are composited and nearest-neighbour upscaled in one fragment shader (shaders/composite.frag)
instead of three pygame.transform.scale calls and blits on the CPU. The world fills the area left of a fixed right-hand
sidebar (ViewConfig.sidebar_frac of the screen width; the world box, grid and editor all use that area). The sidebar
holds every card (Controls, Compute, Gravity) at native resolution on its own layer (hud_layer.py), uploaded to its own
texture and drawn as a blended quad over the composite (shaders/hud.frag), in both modes. While the editor places a
hook, its peg grid overlay (peg_layer.py, native resolution) is drawn the same way between the two (shaders/pegs.frag);
it's uploaded only when it changes.

    python3 main_opengl.py                      fullscreen, GPU compositing
    python3 main_opengl.py --window 1280x800    windowed
    python3 main_opengl.py --scene diamonds     start in a preset world (scene.PRESETS; W switches worlds),
    python3 main_opengl.py --scene "newtons cradle"    e.g. this one (quote names with spaces)
    python3 main_opengl.py --basic              the Framework pipeline as-is: layers composited on the CPU into one
                                                surface, uploaded as one texture, drawn by shaders/basic.frag

Needs moderngl (pip install moderngl). Uses an OpenGL 4.1 core profile context (the newest macOS offers) with
GLSL 330 shaders. Hotkeys (all listed on the sidebar's Controls card): F compute card full / summary, G grid,
V filled/points view, T motion trails, P pause, numpad 1-9 gravity (direction = numpad layout, 5 = off/on), W worlds, ESC quit,
1-4 debug overlays (debug_overlay.py), [ / ] physics speed 0.25x-2x and N one physics frame while paused (timectl.py).

Worlds: `session` (world_selector.WorldSession) holds the Custom world and the preset worlds; `world` and `editor`
are always its active world and that world's editor (rebound whenever the selector switches or resets a world, and
swapped in hud.cards). The session is also the sidebar's Worlds card.

Timing: the compute card's "present" section is split into upload / render / gpu wait / flip. "gpu wait" is a
GPU-tagged section around ctx.finish(): the time the CPU blocks until the GPU has finished this frame's texture
uploads and draw. It is a lower bound on GPU work (anything the GPU finished while the CPU was still issuing
commands is not counted). Set MEASURE_GPU = False to skip the sync.
"""
import argparse
import array
import math
import os
import sys
import time
from collections import namedtuple

import moderngl
import pygame

from scene import Endesga, ViewConfig, INSET, PRESETS
from editor import Editor, PEG_ALPHA, PEG_COL
from profiler import Profiler
from perf_hud import PerfHUD
from hud_layer import HudLayer
from world_selector import WorldSession
from trails import Trails
from gravity_dial import GravityDial
from debug_overlay import DebugOverlays
from timectl import TimeControl

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
FPS = 60
SCALE_DOWN = 2  # real screen px per render surface px
GHOST_ALPHA = 100  # opacity of the translucent ghost layer (screenT)
OVERLAY_LEGEND_PX = 6  # the debug overlay legend's top-left corner, UI px in from the top-left of the screen
MEASURE_GPU = True  # ctx.finish() before the flip so the compute card can show GPU time

# ---------------- moderngl plumbing
Display = namedtuple("Display", "ctx window_size fb_size flip")


def load_shader(ctx, vert_name, frag_name):
    with open(os.path.join(SHADER_DIR, vert_name)) as f:
        vert_src = f.read()
    with open(os.path.join(SHADER_DIR, frag_name)) as f:
        frag_src = f.read()
    return ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)


def surface_swizzle(surface):
    """Texture swizzle mapping a 32-bit pygame surface's raw bytes to RGBA (e.g. 'BGRA' for ARGB8888 on little endian).
    The Framework file uploaded tobytes(..., 'RGBA') with swizzle 'BGRA', which swaps red and blue; this derives
    the swizzle from the surface's channel shifts instead, so the raw buffer can be uploaded without conversion."""
    shifts = surface.get_shifts()
    byte = [s // 8 if sys.byteorder == "little" else 3 - s // 8 for s in shifts]
    return "".join("RGBA"[b] for b in byte)


def open_display(window=None):
    """Create the pygame OpenGL window (fullscreen unless window=(w, h)) and a moderngl context on it."""
    # macOS only offers OpenGL 4.1 as a core, forward-compatible profile; without these it gives a legacy 2.1 context
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    if window is None:
        pygame.display.set_mode((0, 0), flags | pygame.FULLSCREEN)
    else:
        pygame.display.set_mode(window, flags)
    ctx = moderngl.create_context(require=330)
    window_size = pygame.display.get_window_size()
    fb_size = tuple(ctx.screen.viewport[2:])  # drawable size: larger than the window on a high-DPI drawable
    return Display(ctx, window_size, fb_size, pygame.display.flip)


class GLPresenter:
    """Uploads the low-res layers to textures and draws them to the bound framebuffer, upscaled by `scale`."""

    def __init__(self, ctx, layers, scale, fb_size, background, basic=False, hud=None, hud_scale=1.0):
        """layers: the (game, ghost, ui) pygame surfaces; scale: framebuffer px per surface px; hud: the sidebar's
        HudLayer, drawn on top at native resolution; hud_scale: framebuffer px per HUD layer px (window px; > 1 on
        high-DPI)."""
        self.ctx, self.layers, self.scale, self.fb_size, self.basic = ctx, layers, scale, fb_size, basic
        self.hud, self.hud_scale = hud, hud_scale
        self.background = tuple(c / 255 for c in background)
        w, h = layers[0].get_size()
        swizzle = surface_swizzle(layers[0])
        self.textures = []
        for _ in range(1 if basic else 3):
            tex = ctx.texture((w, h), 4)
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)  # crisp pixel art when upscaled
            tex.swizzle = swizzle
            self.textures.append(tex)
        # Fullscreen quad (x, y, u, v) as in the Framework file; the UVs put pygame's top row at the top
        self.quad = ctx.buffer(data=array.array("f", [
            -1.0, 1.0, 0.0, 0.0,  # top left
            -1.0, -1.0, 0.0, 1.0,  # bottom left
            1.0, 1.0, 1.0, 0.0,  # top right
            1.0, -1.0, 1.0, 1.0,  # bottom right
        ]))
        if basic:
            self.prog = load_shader(ctx, "basic.vert", "basic.frag")
            self.vao = ctx.vertex_array(self.prog, [(self.quad, "2f 2f", "in_vert", "in_text")])
        else:
            self.prog = load_shader(ctx, "basic.vert", "composite.frag")
            # composite.frag places pixels from gl_FragCoord, so the UVs are unused (and optimised out): skip them
            self.vao = ctx.vertex_array(self.prog, [(self.quad, "2f 8x", "in_vert")])
            for unit, name in enumerate(("u_game", "u_ghost", "u_ui")):
                self.prog[name].value = unit
            self.prog["u_background"].value = self.background
            self.prog["u_ghost_alpha"].value = GHOST_ALPHA / 255
        if hud is not None:
            # The sidebar layer: a texture of just the sidebar strip, drawn as its own blended quad on top (its top
            # hud.used rows: all of them, the sidebar is opaque).
            self.hud_tex = ctx.texture(hud.surface.get_size(), 4)
            self.hud_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.hud_tex.swizzle = surface_swizzle(hud.surface)
            self.hud_prog = load_shader(ctx, "basic.vert", "hud.frag")
            self.hud_vao = ctx.vertex_array(self.hud_prog, [(self.quad, "2f 8x", "in_vert")])
            self.hud_prog["u_hud"].value = 0
            self.hud_prog["u_scale"].value = hud_scale
            self.hud_prog["u_fb_height"].value = fb_size[1]
            self.hud_prog["u_origin"].value = (hud.rect.x * hud_scale, hud.rect.y * hud_scale)
            self.hud_rows = 0  # rows uploaded this frame
        self.peg_tex, self.peg_version = None, None  # the peg overlay's texture (made on first use) and its version
        self.peg_prog = load_shader(ctx, "basic.vert", "pegs.frag")
        self.peg_vao = ctx.vertex_array(self.peg_prog, [(self.quad, "2f 8x", "in_vert")])
        self.peg_prog["u_pegs"].value = 0
        self.peg_prog["u_scale"].value = hud_scale
        self.peg_prog["u_ui_scale"].value = scale
        self.peg_prog["u_fb_height"].value = fb_size[1]
        self.peg_prog["u_ui_mask"].value = not basic  # basic mode has no separate UI texture to mask with
        self.peg_prog["u_color"].value = tuple(c / 255 for c in PEG_COL) + (PEG_ALPHA / 255,)
        if not basic:
            self.peg_prog["u_ui"].value = 1

    @staticmethod
    def _write(tex, surface):
        w, h = surface.get_size()
        if surface.get_pitch() == 4 * w:
            buf = surface.get_buffer()  # raw pixels, no copy or channel shuffle (the swizzle handles the order)
            tex.write(buf)
            del buf  # unlocks the surface
        else:
            tex.write(pygame.image.tobytes(surface, "RGBA"))  # padded rows: rare; tobytes repacks them

    def upload(self):
        if self.basic:
            game, ghost, ui = self.layers
            game.blit(ghost, (0, 0))  # the Framework pipeline composites on the CPU (ghost keeps its surface alpha)
            game.blit(ui, (0, 0))
            self._write(self.textures[0], game)
        else:
            for tex, surface in zip(self.textures, self.layers):
                self._write(tex, surface)

    def render(self, shake=(0, 0)):
        ctx, (fb_w, fb_h) = self.ctx, self.fb_size
        w, h = self.layers[0].get_size()
        ctx.clear(*self.background)
        if self.basic:
            # The quad fills the viewport, so put the viewport over the scaled surface (top-left aligned, like a blit)
            vw, vh = round(w * self.scale), round(h * self.scale)
            ctx.viewport = (0, fb_h - vh, vw, vh)
            self.textures[0].use(location=0)
            self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            ctx.viewport = (0, 0, fb_w, fb_h)
            return
        ctx.viewport = (0, 0, fb_w, fb_h)
        for unit, tex in enumerate(self.textures):
            tex.use(location=unit)
        prog = self.prog
        prog["u_fb_height"].value = fb_h
        prog["u_origin"].value = (0.0, 0.0)
        prog["u_scale"].value = self.scale
        prog["u_shake"].value = (shake[0] * self.scale / SCALE_DOWN, shake[1] * self.scale / SCALE_DOWN)
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)

    def render_pegs(self, layer):
        """Blend the editor's peg overlay (a peg_layer.PegLayer, or None when not placing a hook) over the frame. Its
        mask is uploaded to an R8 texture only when the layer was rebuilt (layer.version changed)."""
        if layer is None:
            return
        if self.peg_tex is None or self.peg_tex.size != (layer.w, layer.h):
            self.peg_tex = self.ctx.texture((layer.w, layer.h), 1, dtype="f1")
            self.peg_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.peg_version = None
        if layer.version != self.peg_version:
            self.peg_tex.write(layer.mask.tobytes())  # row 0 = the top row, like the other layers
            self.peg_version = layer.version
        ctx = self.ctx
        ctx.viewport = (0, 0, *self.fb_size)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.peg_tex.use(location=0)
        if not self.basic:
            self.textures[2].use(location=1)  # the UI layer
        self.peg_vao.render(mode=moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

    def upload_hud(self):
        """Upload the HUD layer's used rows (the top `hud.used` rows; texture row 0 is the surface's top row)."""
        hud = self.hud
        self.hud_rows = rows = hud.used if hud is not None else 0
        if not rows:
            return
        surface = hud.surface
        w = surface.get_width()
        if surface.get_pitch() == 4 * w:
            buf = surface.get_buffer()
            self.hud_tex.write(memoryview(buf)[:4 * w * rows], viewport=(0, 0, w, rows))
            del buf  # unlocks the surface
        else:
            self.hud_tex.write(pygame.image.tobytes(surface.subsurface((0, 0, w, rows)), "RGBA"), viewport=(0, 0, w, rows))

    def render_hud(self):
        """Blend the uploaded HUD rows over the frame: viewport = the layer's used rect, so no other pixel is shaded."""
        if not self.hud_rows:
            return
        ctx, (fb_w, fb_h), k, rect = self.ctx, self.fb_size, self.hud_scale, self.hud.rect
        x0, y0 = round(rect.x * k), round(rect.y * k)
        x1, y1 = round(rect.right * k), round((rect.y + self.hud_rows) * k)
        ctx.viewport = (x0, fb_h - y1, x1 - x0, y1 - y0)  # GL viewports are bottom-up
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA  # pygame's "over" blit
        self.hud_tex.use(location=0)
        self.hud_vao.render(mode=moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        ctx.viewport = (0, 0, fb_w, fb_h)


# ---------------- The game loop, presented through GLPresenter
def draw_sidebar(hud, editor, perf_hud, mouse_px, draw_top):
    """The sidebar for this frame: every card at native resolution (hud.cards, in order), then the cursor on top.
    mouse_px: the mouse in window px; draw_top(surface, origin): the main loop's message + cursor drawer."""
    perf_hud.collapsed = not editor.show_compute  # F / [f]: the Compute card as its short summary
    hud.draw(mouse_px)
    hud.overlay(draw_top)


def main(argv=None, display=None, max_frames=None, on_frame=None):
    """argv: command line (see the module docstring). display / max_frames / on_frame are test hooks: a Display to
    render into instead of opening a window, a frame limit, and a callback(frame_index, locals) after each frame."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--window", help="windowed mode at WxH screen px instead of fullscreen, e.g. 1280x800")
    parser.add_argument("--basic", action="store_true", help="the Framework pipeline: CPU composite, one texture")
    parser.add_argument("--scene", default="default", choices=sorted(PRESETS), help="preset scene to start with")
    args = parser.parse_args(argv)

    pygame.init()
    if display is None:
        display = open_display(tuple(int(v) for v in args.window.split("x")) if args.window else None)
    ctx = display.ctx
    clock = pygame.time.Clock()
    fps = FPS
    scaleDownFactor = SCALE_DOWN
    win_w, win_h = display.window_size
    screen_width = int(win_w / scaleDownFactor)
    screen_height = int(win_h / scaleDownFactor)
    # Per-pixel alpha surfaces (SRCALPHA rather than convert_alpha: same blending, no display surface needed)
    screen2 = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    screenT = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)  # its alpha (100) is applied on the GPU
    screenUI = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    if args.basic:
        screenT.set_alpha(GHOST_ALPHA)  # the CPU composite blits it, so it needs the surface alpha
    timer = 0
    shake = [0, 0]
    oscillating_random_thing = 0
    click = False

    # framebuffer px per window px (the window may be backed by a larger, high-DPI drawable) and per surface px
    dpi_scale = display.fb_size[0] / win_w
    fb_scale = scaleDownFactor * dpi_scale
    hud = HudLayer(display.window_size, scaleDownFactor, ViewConfig.sidebar_frac)  # the sidebar, native (window px) res
    presenter = GLPresenter(ctx, (screen2, screenT, screenUI), fb_scale, display.fb_size, Endesga.my_blue,
                            basic=args.basic, hud=hud, hud_scale=dpi_scale)

    # The world is measured in its own units and fills the area left of the sidebar: that area's size on the render
    # surfaces (UI px) / the zoom. Every scene builder takes world_width / world_height, so all get the smaller world.
    world_px = (hud.rect.x // scaleDownFactor, screen_height)  # the world area, render-surface px
    world_width, world_height = world_px[0] / ViewConfig.zoom, world_px[1] / ViewConfig.zoom
    world_view = screen2.subsurface((0, 0) + world_px)  # the grid, trails and bodies are drawn in the world area only
    profiler = Profiler()

    def make_editor(w):
        return Editor(w, Endesga, (screen_width, screen_height), (world_width, world_height), scaleDownFactor, INSET,
                      ViewConfig.zoom, view=ViewConfig, screen_size=display.window_size, sidebar=hud.rect)

    # Custom (the default scene at launch, kept for the session) + the presets; only the active world steps, and it
    # alone has the profiler attached (the session moves it)
    session = WorldSession(PRESETS, (world_width, world_height), make_editor, profiler, start=args.scene)
    world, editor = session.world, session.editor
    perf_hud = PerfHUD(profiler)
    trails = Trails()  # T: fading paths behind moving bodies, drawn under them
    gravity_dial = GravityDial()  # the gravity selector card (arrow keys, 0 and numpad too)
    # the sidebar's cards, top to bottom (hud_layer.py describes a card provider; append new cards here)
    # (the Controls card is the active world's editor: the session swaps it in this list when it switches worlds)
    hud.cards = [editor, session, perf_hud, gravity_dial]
    session.bind_cards(hud.cards)
    hud.layout()

    overlays = DebugOverlays()  # keys 1-4
    time_ctl = TimeControl()  # [ / ] physics speed, N frame step
    editor.overlays, editor.time = overlays, time_ctl

    def draw_top(surface, origin=(0, 0)):
        """What goes over the sidebar, at UI-surface px minus origin: the editor message and the cursor."""
        editor.draw_message(surface, origin)
        cx, cy = mx - origin[0], my - origin[1]
        pygame.draw.circle(surface, Endesga.black, (cx + 1, cy + 1), 2, 1)
        pygame.draw.circle(surface, Endesga.white, (cx, cy), 2, 1)

    # ---------------- Main Game Loop
    last_time = time.time()
    running = True
    frame_index = 0
    while running:
        profiler.frame()
        world, editor = session.world, session.editor  # the active world (anything may have switched / replaced it)
        profiler.begin("draw")

        # ---------------- Reset Variables and Clear screens
        mx, my = pygame.mouse.get_pos()
        mx, my = mx / scaleDownFactor, my / scaleDownFactor  # render-surface pixels (cursor, UI)
        wx, wy = mx / ViewConfig.zoom, my / ViewConfig.zoom  # world units (physics)
        screen2.fill(Endesga.my_blue)
        screenT.fill((0, 0, 0, 0))
        screenUI.fill((0, 0, 0, 0))
        dt = time.time() - last_time
        dt *= fps
        last_time = time.time()
        timer -= 1 * dt
        shake = [0, 0]
        oscillating_random_thing += math.pi / fps * dt

        profiler.switch("events")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if session.handle_event(event):  # W, the Worlds card and the world selector panel (may switch worlds)
                world, editor = session.world, session.editor
                continue
            if gravity_dial.handle_event(event):  # arrow / 0 / numpad keys, and clicks on the gravity card
                continue
            if editor.handle_event(event):  # the Controls card, the rest of the sidebar, editor panels and placement
                continue
            if event.type == pygame.MOUSEBUTTONDOWN:
                click = True
                world.grab((wx, wy))
            if event.type == pygame.MOUSEBUTTONUP:
                click = False
                world.release()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_g:
                    ViewConfig.show_grid = not ViewConfig.show_grid
                if event.key == pygame.K_v:
                    ViewConfig.filled = not ViewConfig.filled
                if event.key == pygame.K_t:
                    ViewConfig.trails = not ViewConfig.trails
                if pygame.K_1 <= event.key <= pygame.K_4:
                    overlays.toggle(event.key - pygame.K_1)
                if event.key == pygame.K_LEFTBRACKET:
                    time_ctl.slower()
                if event.key == pygame.K_RIGHTBRACKET:
                    time_ctl.faster()
                if event.key == pygame.K_n and editor.can_frame_step():
                    time_ctl.request_step()

        # ---------------- Updating Screen
        profiler.switch("physics")
        world.drag_to((wx, wy))
        overlays.attach(world)  # World.debug: a step recorder while overlay 2 or 4 is on, else None
        # dt is in frames here, World wants seconds; the time scale applies to the physics only (timectl.py)
        stepped = not editor.paused
        if stepped:
            world.step(*time_ctl.step_args(dt / fps))
            overlays.after_step(world)
        for _ in range(time_ctl.take_steps() if editor.paused else 0):  # N: one nominal frame per press
            world.step(*time_ctl.step_args(1 / fps))
            overlays.after_step(world)
            stepped = True
        time_ctl.pending = 0  # a step requested just before resuming is dropped
        profiler.switch("draw")
        if ViewConfig.show_grid:
            world.draw_grid(world_view, zoom=ViewConfig.zoom)
        with profiler.section("trails"):
            trails.set_enabled(ViewConfig.trails)
            trails.update(world, dt / fps, stepped)
            trails.draw(world_view, zoom=ViewConfig.zoom)  # under the bodies
        world.draw(world_view, filled=ViewConfig.filled, zoom=ViewConfig.zoom)
        if overlays.any():
            with profiler.section("overlays"):
                overlays.draw(world_view, world, ViewConfig.zoom)
        editor.draw_world(screen2, screenT, (mx, my))
        if overlays.any():  # the legend: top left of the world area
            overlays.draw_legend(screenUI, world, (OVERLAY_LEGEND_PX, OVERLAY_LEGEND_PX))
        editor.draw_ui(screenUI, (mx, my))  # panels, buttons and hints (in the world area)
        session.draw(screenUI, (mx, my))  # the world selector panel (W), over the editor's panels

        profiler.switch("ui")
        pygame.mouse.set_visible(False)
        draw_top(screenUI)  # message + cursor
        draw_sidebar(hud, editor, perf_hud, pygame.mouse.get_pos(), draw_top)  # native res, over the composite

        # ---------------- Present: textures -> one shader pass (+ the HUD quad) -> swap
        profiler.switch("present")
        with profiler.section("upload"):
            presenter.upload()
            with profiler.section("hud"):
                presenter.upload_hud()
        with profiler.section("render"):
            presenter.render(shake)
            presenter.render_pegs(editor.peg_layer())  # native-res peg grid while placing a hook
            with profiler.section("hud"):
                presenter.render_hud()
        if MEASURE_GPU:
            with profiler.section("gpu wait", device="gpu"):
                ctx.finish()
        with profiler.section("flip"):
            display.flip()
        if on_frame is not None:
            on_frame(frame_index, locals())
        frame_index += 1
        if max_frames is not None and frame_index >= max_frames:
            running = False
        profiler.switch("idle", device="idle")  # the frame cap's sleep
        clock.tick(fps)
    profiler.frame()
    return profiler


if __name__ == "__main__":
    main()
    pygame.quit()
