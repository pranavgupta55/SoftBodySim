"""The game: scene, physics, editor and HUD, presented through moderngl.

Ported from Framework/mainScreenOpenGL.py. The world itself (background, G grid, T trails, every body, rope and
hook) is drawn on the GPU by gl_world.py into a low-res framebuffer the size of the render surface, with shaders that
reproduce pygame's own rasterisation pixel for pixel. Everything else still draws into low-res pygame surfaces:
screen2 (now transparent: debug overlays and the editor's world-space previews over the GPU world), screenT (the
translucent editor ghost) and screenUI (panels, hints, cursor). Only their rows that hold anything (now or last
frame) are uploaded. The GPU world and the three layers are composited and nearest-neighbour upscaled in one fragment
shader (shaders/composite.frag) instead of pygame.transform.scale calls and blits on the CPU. The world fills the area left of a fixed right-hand
sidebar (ViewConfig.sidebar_frac of the screen width; the world box, grid and editor all use that area). The sidebar
holds every card (Controls, Compute, Gravity) at native resolution on its own layer (hud_layer.py), uploaded to its own
texture and drawn as a blended quad over the composite (shaders/hud.frag), in both modes. The sidebar is redrawn
and re-uploaded only when what it shows changes (sidebar_key; the Compute card's live numbers at 10 Hz). While the editor places a
hook, its peg grid overlay (peg_layer.py, native resolution) is drawn the same way between the two (shaders/pegs.frag);
it's uploaded only when it changes.

    python3 main_opengl.py                      fullscreen, GPU compositing
    python3 main_opengl.py --window 1280x800    windowed
    python3 main_opengl.py --scene diamonds     start in a preset world (scene.PRESETS; W switches worlds),
    python3 main_opengl.py --scene "newtons cradle"    e.g. this one (quote names with spaces)
    python3 main_opengl.py --basic              the Framework pipeline as-is: everything drawn by pygame, layers
                                                composited on the CPU into one surface, uploaded as one texture,
                                                drawn by shaders/basic.frag
    python3 main_opengl.py --steps 8x1          small steps: 8 substeps x 1 solver iteration per frame (default 4x4)

Needs moderngl (pip install moderngl). Uses an OpenGL 4.1 core profile context (the newest macOS offers) with
GLSL 330 shaders. Hotkeys (all listed on the sidebar's Controls card): F compute card full / summary, G grid,
V view (points / filled / filled without outlines), T motion trails, L lighting (the lit renderer; its Lights and
Material cards, lighting_ui.py), SPACE pause, right click spawn menu, arrow keys gravity (two for a diagonal; 0 = off/on; numpad 1-9 too), W worlds, H panel (fullscreen worlds), B point / circle grab, O circle radius (4 presets; the wheel too), ESC quit,
1-4 debug overlays (debug_overlay.py), [ / ] or the Controls card's - / + physics speed 1%-10000% and N one physics frame while paused (timectl.py).
Sidebar: click a card's title row to open / close it; the wheel over the sidebar scrolls it (hud_layer.py).

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
import threading
import time
from collections import namedtuple

import moderngl
import numpy as np
import pygame

from entities import EnvConfig
from scene import Endesga, ViewConfig, INSET, PRESETS, FULLSCREEN_PRESETS
from editor import Editor, PEG_ALPHA, PEG_COL
from profiler import Profiler
from perf_hud import PerfHUD
from hud_layer import HudLayer
from world_selector import WorldSession
from trails import Trails
from gravity_dial import GravityDial
from palettes import PaletteCard
from lighting_ui import LightsCard, MaterialCard
from debug_overlay import DebugOverlays
from timectl import TimeControl
from gl_world import GLWorld

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
FPS = 160  # frame cap, purely visual: physics runs fixed 1/240 s ticks at any frame rate (World.step)
SCALE_DOWN = 2  # real screen px per render surface px
GHOST_ALPHA = 100  # opacity of the translucent ghost layer (screenT)
GRAB_AREA_MIN, GRAB_AREA_MAX = 10.0, 400.0  # the area grab's circle radius range (world units)
GRAB_AREA_WHEEL = 1.15  # each wheel notch scales the circle by this
GRAB_AREA_PRESETS = (20.0, 40.0, 80.0, 160.0)  # O cycles the circle's radius through these (world units)
OVERLAY_LEGEND_PX = 6  # the debug overlay legend's top-left corner, UI px in from the top-left of the screen
MEASURE_GPU = True  # ctx.finish() before the flip so the compute card can show GPU time
INPUT_EVENTS = {pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL}

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


def used_rows(surface):
    """(top, bottom) of the rows of a 32-bit surface holding any non-zero pixel (a cleared transparent layer is all
    zeros), or None. A numpy scan of the raw buffer: ~40x faster than Surface.get_bounding_rect."""
    w, h = surface.get_size()
    if surface.get_pitch() != 4 * w:
        box = surface.get_bounding_rect()
        return (box.top, box.bottom) if box.h else None
    buf = surface.get_buffer()
    rows = np.flatnonzero(np.frombuffer(buf, np.uint32).reshape(h, w).any(axis=1))
    del buf  # unlocks the surface
    return (int(rows[0]), int(rows[-1]) + 1) if rows.size else None


class GLPresenter:
    """Uploads the low-res layers to textures and draws them to the bound framebuffer, upscaled by `scale`."""

    def __init__(self, ctx, layers, scale, fb_size, background, basic=False, hud=None, hud_scale=1.0, world_tex=None):
        """layers: the (game, ghost, ui) pygame surfaces; scale: framebuffer px per surface px; hud: the sidebar's
        HudLayer, drawn on top at native resolution; hud_scale: framebuffer px per HUD layer px (window px; > 1 on
        high-DPI); world_tex: the GPU world layer's texture (gl_world.GLWorld.tex, the layers' size), composited
        under the game layer, or None (the game layer holds the whole world)."""
        self.ctx, self.layers, self.scale, self.fb_size, self.basic = ctx, layers, scale, fb_size, basic
        self.world_tex = world_tex
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
            for unit, name in enumerate(("u_game", "u_ghost", "u_ui", "u_world")):
                self.prog[name].value = unit
            self.prog["u_world_on"].value = world_tex is not None
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
        # per layer: the rows (top, bottom) that held anything at the last upload (None: all transparent); the
        # textures start out uploaded in full
        self.layer_rows = [(0, h)] * len(self.textures)
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

    @staticmethod
    def _write_rows(tex, surface, top, bottom):
        """Upload rows [top, bottom) of `surface` into the same rows of `tex`."""
        w = surface.get_width()
        if surface.get_pitch() == 4 * w:
            buf = surface.get_buffer()
            tex.write(memoryview(buf)[4 * w * top:4 * w * bottom], viewport=(0, top, w, bottom - top))
            del buf  # unlocks the surface
        else:
            part = surface.subsurface((0, top, w, bottom - top))
            tex.write(pygame.image.tobytes(part, "RGBA"), viewport=(0, top, w, bottom - top))

    def upload(self):
        if self.basic:
            game, ghost, ui = self.layers
            game.blit(ghost, (0, 0))  # the Framework pipeline composites on the CPU (ghost keeps its surface alpha)
            game.blit(ui, (0, 0))
            self._write(self.textures[0], game)
            return
        # Only the rows that hold something now or held something last frame: with the world on the GPU these
        # layers are mostly transparent (the cursor, a panel, an editor preview), so most frames upload a few rows
        for k, (tex, surface) in enumerate(zip(self.textures, self.layers)):
            rows = used_rows(surface)
            last, self.layer_rows[k] = self.layer_rows[k], rows
            span = [r for r in (rows, last) if r is not None]
            if span:
                self._write_rows(tex, surface, min(r[0] for r in span), max(r[1] for r in span))

    def set_background(self, rgb):
        """The color behind the game layer (the active world's background)."""
        background = tuple(c / 255 for c in rgb)
        if background != self.background:
            self.background = background
            if not self.basic:
                self.prog["u_background"].value = background

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
        if self.world_tex is not None:
            self.world_tex.use(location=3)
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
        if not rows or hud.dirty is None:  # (nothing redrawn: the texture still holds it)
            return
        top, bottom = max(hud.dirty[0], 0), min(hud.dirty[1], rows)
        if bottom > top:
            self._write_rows(self.hud_tex, hud.surface, top, bottom)

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


def preload_gpu_backend():
    """Import gpu_backend (and with it MLX, tens of ms) on a background thread, so the first world that runs on the
    GPU (backend "auto" picks it for big scenes such as diamonds and fluid) doesn't pay for it when it's opened.
    Only the import: kernels are still built on first use, on the main thread. Without MLX this does nothing (World
    then imports it, and fails, exactly as before). Skipped when every world is forced onto the CPU."""
    if EnvConfig.backend == "cpu" or "gpu_backend" in sys.modules:
        return

    def load():
        try:
            import gpu_backend  # noqa: F401  (the import system's lock makes a concurrent import wait for this one)
        except Exception:
            pass

    threading.Thread(target=load, name="gpu-backend-preload", daemon=True).start()


# ---------------- The game loop, presented through GLPresenter
def draw_sidebar(hud, editor, perf_hud, mouse_px, draw_top, key=None):
    """The sidebar for this frame: every card at native resolution (hud.cards, in order), then the cursor on top.
    mouse_px: the mouse in window px; draw_top(surface, origin): the main loop's message + cursor drawer; key: the
    state the cards show (HudLayer.draw: redrawn only when it changes), None to redraw regardless."""
    perf_hud.collapsed = not editor.show_compute  # F / [f]: the Compute card as its short summary
    perf_hud.world = editor.world  # its backend label
    hud.draw(mouse_px, key)
    hud.overlay(draw_top)


SIDEBAR_SAFETY_HZ = 2  # the sidebar is redrawn at least this often, whatever its key says (HudLayer.draw)


def sidebar_key(inputs, mouse_px, hud, session, editor, panel_open, now, look=()):
    """HudLayer.draw's key for this frame: what the cards show that can change without an input event changing
    `inputs` (keys / buttons / wheel seen so far), plus the mouse while it is over the sidebar (hover). look: the
    lighting cards' versions (a slider or gizmo drag changes them without input events); the sidebar's open cards and
    scroll offset (the scroll keeps moving after the wheel's events, HudLayer.step)."""
    return (inputs, mouse_px if hud.rect.collidepoint(mouse_px) else None, session.name, session.open, id(editor),
            editor.paused, editor.active, editor.stage, editor.show_compute, ViewConfig.show_grid, ViewConfig.filled,
            ViewConfig.outlines, ViewConfig.lighting,
            ViewConfig.trails, ViewConfig.grab_area, ViewConfig.grab_area_radius, panel_open, int(now * SIDEBAR_SAFETY_HZ),
            tuple(look), hud.state_key())


def grid_colors(background):
    """draw_grid colors for a world's own background: its default ones (made for Endesga.my_blue) are lifted toward
    white by about as much as they are from my_blue. None (the default background) keeps the defaults."""
    if background is None:
        return {}
    lift = lambda t: tuple(int(c + (255 - c) * t) for c in background)
    return {"line_col": lift(0.10), "edge_cell_col": lift(0.025), "point_cell_col": lift(0.06)}


def main(argv=None, display=None, max_frames=None, on_frame=None):
    """argv: command line (see the module docstring). display / max_frames / on_frame are test hooks: a Display to
    render into instead of opening a window, a frame limit, and a callback(frame_index, locals) after each frame."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--window", help="windowed mode at WxH screen px instead of fullscreen, e.g. 1280x800")
    parser.add_argument("--basic", action="store_true", help="the Framework pipeline: CPU composite, one texture")
    folders = sorted({name.split("/", 1)[0] for name in PRESETS if "/" in name})
    parser.add_argument("--scene", default="penrose/evolved 1", choices=sorted(PRESETS) + folders,
                        help="preset scene to start with (a folder name: its first world)")
    parser.add_argument("--steps", metavar="SxI",
                        help="physics schedule for every world: substeps x solver iterations per frame, e.g. 8x1 "
                             "(small steps, entities.SMALL_STEPS); default 4x4. A world's own env still wins")
    args = parser.parse_args(argv)
    if args.steps:
        try:
            EnvConfig.substeps, EnvConfig.iterations = (max(1, int(v)) for v in args.steps.lower().split("x"))
        except ValueError:
            parser.error(f"--steps wants SxI, e.g. 8x1 (got {args.steps!r})")

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
    # The world (background, grid, trails, bodies) is drawn on the GPU into its own layer, under screen2, which then
    # only holds what pygame still draws in world space (debug overlays, editor previews). --basic: all pygame
    gpu_world = None if args.basic else GLWorld(ctx, (screen_width, screen_height))
    presenter = GLPresenter(ctx, (screen2, screenT, screenUI), fb_scale, display.fb_size, Endesga.my_blue,
                            basic=args.basic, hud=hud, hud_scale=dpi_scale,
                            world_tex=gpu_world.tex if gpu_world is not None else None)

    # The world is measured in its own units and fills the area left of the sidebar: that area's size on the render
    # surfaces (UI px) / the zoom. Every scene builder takes world_width / world_height, so all get the smaller world.
    world_px = (hud.rect.x // scaleDownFactor, screen_height)  # the world area, render-surface px
    world_width, world_height = world_px[0] / ViewConfig.zoom, world_px[1] / ViewConfig.zoom
    world_view = screen2.subsurface((0, 0) + world_px)  # the grid, trails and bodies are drawn in the world area only
    profiler = Profiler()

    full_width = screen_width / ViewConfig.zoom  # a fullscreen world's width: the whole window
    no_sidebar = pygame.Rect(win_w, 0, 0, win_h)  # a fullscreen editor's sidebar while the panel is hidden

    def make_editor(w):
        full = getattr(w, "fullscreen", False)
        return Editor(w, Endesga, (screen_width, screen_height), (full_width if full else world_width, world_height),
                      scaleDownFactor, INSET, ViewConfig.zoom, view=ViewConfig, screen_size=display.window_size,
                      sidebar=no_sidebar if full else hud.rect)

    def build_fullscreen(builder):
        def build(_w, h):
            w = builder(full_width, h)
            w.fullscreen = True
            return w
        return build

    presets = {name: build_fullscreen(b) if name in FULLSCREEN_PRESETS else b for name, b in PRESETS.items()}
    panel_open = False  # the sidebar floating over a fullscreen world (H)
    shown_name = None  # the active world's name last frame (a fullscreen world is unpaused when it's switched to)
    grid_world = None  # the world whose G grid state ViewConfig.show_grid holds

    def toggle_panel():
        """H / the Palette card's button, in a fullscreen world: show / hide the sidebar as a panel floating over
        the right side of the world (the world keeps its size and shape)."""
        nonlocal panel_open
        if getattr(session.world, "fullscreen", False):
            panel_open = not panel_open
            session.editor.sidebar = hud.rect if panel_open else no_sidebar

    # Custom (the default scene at launch, kept for the session) + the presets; only the active world steps, and it
    # alone has the profiler attached (the session moves it)
    start = args.scene if args.scene in presets else next(n for n in presets if n.startswith(args.scene + "/"))
    session = WorldSession(presets, (world_width, world_height), make_editor, profiler, start=start)
    world, editor = session.world, session.editor
    perf_hud = PerfHUD(profiler)
    trails = Trails()  # T: fading paths behind moving bodies, drawn under them
    gravity_dial = GravityDial()  # the gravity selector card (arrow keys, 0 and numpad too)
    # the sidebar's cards, top to bottom (hud_layer.py describes a card provider; append new cards here)
    # (the Controls card is the active world's editor: the session swaps it in this list when it switches worlds)
    hud.cards = [editor, session, perf_hud, gravity_dial]
    palette_card = PaletteCard(lambda: session.world, toggle_panel)  # last, while the active world has palettes

    def toggle_lighting():
        """L / the Lights card's button: the lit renderer for the active world (world.lighting; ViewConfig.lighting
        follows the active world's, see the main loop)."""
        ViewConfig.lighting = session.world.lighting = not ViewConfig.lighting

    # while lighting is on: the Lights card, and the Material card while bodies are picked (lighting_ui.py)
    lights_card = LightsCard(lambda: session.editor, ViewConfig, toggle_lighting)
    material_card = MaterialCard(lambda: session.editor)
    look_cards = (material_card, lights_card)
    lit_seen = None  # (world, its lighting switch) last frame
    session.bind_cards(hud.cards)
    hud.layout()

    overlays = DebugOverlays()  # keys 1-4
    time_ctl = TimeControl()  # [ / ] physics speed, N frame step
    editor.overlays, editor.time = overlays, time_ctl

    def draw_top(surface, origin=(0, 0)):
        """What goes over the sidebar, at UI-surface px minus origin: the editor message and the cursor."""
        editor.draw_message(surface, origin)
        if full and not panel_open and not session.open:
            return  # a fullscreen world with its panel hidden (H): no cursor, just the world (W's panel still has one)
        cx, cy = mx - origin[0], my - origin[1]
        pygame.draw.circle(surface, Endesga.black, (cx + 1, cy + 1), 2, 1)
        pygame.draw.circle(surface, Endesga.white, (cx, cy), 2, 1)

    # ---------------- Main Game Loop
    last_time = time.time()
    inputs = 0  # key / button / wheel events so far (part of the sidebar's redraw key)
    running = True
    frame_index = 0
    while running:
        profiler.frame()
        world, editor = session.world, session.editor  # the active world (anything may have switched / replaced it)
        full = getattr(world, "fullscreen", False)
        if world is not grid_world:  # G is per world: keep the old one's grid state, take the new one's (default on)
            if grid_world is not None:
                grid_world.show_grid = ViewConfig.show_grid
            ViewConfig.show_grid = getattr(world, "show_grid", True)
            grid_world = world
        if full and session.name != shown_name:  # a fullscreen world runs however it was opened (H, W panel, ...)
            editor.paused, panel_open = False, False
        shown_name = session.name
        sidebar_shown = not full or panel_open
        if full:
            editor.sidebar = hud.rect if panel_open else no_sidebar
        lit = bool(getattr(world, "lighting", False))  # L is per world (a demo world starts lit)
        if lit_seen is None:  # the first world: lit if it is or ViewConfig.lighting was set before main()
            lit = ViewConfig.lighting = bool(ViewConfig.lighting or lit)
            if lit:
                world.lighting = True
        elif (world, lit) != lit_seen:  # another world, or this one's switch changed (a scene load)
            ViewConfig.lighting = lit
        lit_seen = (world, lit)
        for card in look_cards:  # the lighting cards come and go too (before the Palette card)
            wanted = card.wanted(ViewConfig)
            if wanted != (card in hud.cards):
                if wanted:
                    hud.cards.insert(hud.cards.index(palette_card) if palette_card in hud.cards else len(hud.cards), card)
                else:
                    hud.cards.remove(card)
        if (world.palette is not None) != (palette_card in hud.cards):  # the Palette card comes and goes with it
            if palette_card in hud.cards:
                hud.cards.remove(palette_card)
            else:
                hud.cards.append(palette_card)
        profiler.begin("draw")

        # ---------------- Reset Variables and Clear screens
        mx, my = pygame.mouse.get_pos()
        mx, my = mx / scaleDownFactor, my / scaleDownFactor  # render-surface pixels (cursor, UI)
        wx, wy = mx / ViewConfig.zoom, my / ViewConfig.zoom  # world units (physics)
        background = world.background or Endesga.my_blue  # presets may have their own (World.background)
        screen2.fill(background if gpu_world is None else (0, 0, 0, 0))
        presenter.set_background(background)
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
            if event.type in INPUT_EVENTS:
                inputs += 1  # (the sidebar is redrawn after any of these)
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and editor.scenes.state is None and not editor.active:
                if event.key == pygame.K_h and full:  # H: the floating panel over a fullscreen world
                    toggle_panel()
                    sidebar_shown = panel_open
                    continue
            if sidebar_shown and editor.scenes.state is None and hud.handle_event(event, pygame.mouse.get_pos()):
                continue  # the sidebar's card headers (open / close), its scrolling, clicks between its cards
            if not sidebar_shown and event.type != pygame.KEYDOWN and event.type != pygame.KEYUP:
                if session.open and event.type == pygame.MOUSEBUTTONDOWN:  # the Worlds panel (W) still takes clicks
                    if hud.rect.collidepoint(event.pos) and not session.on_panel(event.pos):
                        session.toggle()  # on the hidden cards: a click outside the panel
                        continue
                    if session.handle_event(event):
                        world, editor = session.world, session.editor
                        continue
                # no sidebar: its cards take no other clicks (the editor still gets them, below)
            elif editor.scenes.state is None and session.handle_event(event):  # W, the Worlds card and panel
                world, editor = session.world, session.editor  # (may switch worlds)
                continue
            # arrow / 0 / numpad keys, and clicks on the gravity card (not while the modal save / open dialog is up)
            elif editor.scenes.state is None and palette_card in hud.cards and palette_card.handle_event(event):
                continue  # clicks on the Palette card
            elif editor.scenes.state is None and any(c in hud.cards and c.handle_event(event) for c in look_cards):
                continue  # clicks and slider drags on the lighting cards
            elif editor.scenes.state is None and gravity_dial.handle_event(event):
                continue
            if editor.scenes.state is None and not session.open and lights_card.handle_world_event(event):
                continue  # light gizmos: pick, drag, aim; Delete removes the selected light
            if editor.handle_event(event):  # the Controls card, the rest of the sidebar, editor panels and placement
                continue
            if event.type == pygame.MOUSEBUTTONDOWN:
                click = True
                world.grab((wx, wy), ViewConfig.grab_area_radius if ViewConfig.grab_area else None)
            if event.type == pygame.MOUSEBUTTONUP:
                click = False
                world.release()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_g:
                    ViewConfig.show_grid = not ViewConfig.show_grid
                if event.key == pygame.K_v:
                    if not ViewConfig.filled:  # points -> filled -> filled without outlines -> points
                        ViewConfig.filled, ViewConfig.outlines = True, True
                    elif ViewConfig.outlines:
                        ViewConfig.outlines = False
                    else:
                        ViewConfig.filled, ViewConfig.outlines = False, True
                if event.key == pygame.K_t:
                    ViewConfig.trails = not ViewConfig.trails
                if event.key == pygame.K_l:
                    toggle_lighting()
                if event.key == pygame.K_b:  # B: grab the nearest point, or every point in a circle
                    ViewConfig.grab_area = not ViewConfig.grab_area
                if event.key == pygame.K_o and ViewConfig.grab_area:  # O: next preset radius (after the current one)
                    bigger = [r for r in GRAB_AREA_PRESETS if r > ViewConfig.grab_area_radius + 1e-6]
                    ViewConfig.grab_area_radius = bigger[0] if bigger else GRAB_AREA_PRESETS[0]
                if pygame.K_1 <= event.key <= pygame.K_4:
                    overlays.toggle(event.key - pygame.K_1)
                if event.key == pygame.K_LEFTBRACKET:
                    time_ctl.slower()
                if event.key == pygame.K_RIGHTBRACKET:
                    time_ctl.faster()
                if event.key == pygame.K_n and editor.can_frame_step():
                    time_ctl.request_step()
            if event.type == pygame.MOUSEWHEEL and ViewConfig.grab_area:  # the wheel resizes the grab circle
                r = ViewConfig.grab_area_radius * GRAB_AREA_WHEEL ** event.y
                ViewConfig.grab_area_radius = min(GRAB_AREA_MAX, max(GRAB_AREA_MIN, r))

        # ---------------- Updating Screen
        profiler.switch("physics")
        world.drag_to((wx, wy), ViewConfig.grab_area_radius if ViewConfig.grab_area else None)
        overlays.attach(world)  # World.debug: a step recorder while overlay 2 or 4 is on, else None
        # dt is in frames here, World wants seconds; the time scale applies to the physics only (timectl.py). The
        # world runs whole fixed ticks and carries the rest, so the frame rate never changes the physics
        stepped = not editor.paused
        if stepped:
            world.step(*time_ctl.step_args(dt / fps))
            time_ctl.note(dt / fps, getattr(world, "last_sim_dt", 0.0))  # the delivered speed (Controls card)
            overlays.after_step(world)
        else:
            time_ctl.reset_actual()
        for _ in range(time_ctl.take_steps() if editor.paused else 0):  # N: one 1/60 s frame (x scale) per press
            world.step_frame(time_ctl.scale)
            overlays.after_step(world)
            stepped = True
        time_ctl.pending = 0  # a step requested just before resuming is dropped
        profiler.switch("draw")
        view = screen2 if full else world_view  # a fullscreen world fills the whole render surface
        if gpu_world is not None:
            with profiler.section("trails"):
                trails.set_enabled(ViewConfig.trails)
                trails.update(world, dt / fps, stepped)
            with profiler.section("world gl"):  # grid, trails and bodies into the GPU world layer
                gpu_world.render(world, view.get_size(), ViewConfig.filled, ViewConfig.zoom, background,
                                 outlines=ViewConfig.outlines, lighting=ViewConfig.lighting,
                                 grid=grid_colors(world.background) if ViewConfig.show_grid else None,
                                 trails=trails if ViewConfig.trails else None)
        else:
            if ViewConfig.show_grid:
                world.draw_grid(view, zoom=ViewConfig.zoom, **grid_colors(world.background))
            with profiler.section("trails"):
                trails.set_enabled(ViewConfig.trails)
                trails.update(world, dt / fps, stepped)
                trails.draw(view, zoom=ViewConfig.zoom)  # under the bodies
            world.draw(view, filled=ViewConfig.filled, zoom=ViewConfig.zoom, outlines=ViewConfig.outlines)
        if overlays.any():
            with profiler.section("overlays"):
                overlays.draw(view, world, ViewConfig.zoom)
        if ViewConfig.grab_area and not editor.active:  # the area grab's circle around the cursor
            pygame.draw.circle(screen2, (255, 255, 255), (mx, my), max(2, int(ViewConfig.grab_area_radius
                                                                                * ViewConfig.zoom)), 1)
        editor.draw_world(screen2, screenT, (mx, my))
        lights_card.draw_world(screen2)  # light gizmos, while lighting is on
        if overlays.any():  # the legend: top left of the world area
            overlays.draw_legend(screenUI, world, (OVERLAY_LEGEND_PX, OVERLAY_LEGEND_PX))
        editor.draw_ui(screenUI, (mx, my))  # panels, buttons and hints (in the world area)
        session.draw(screenUI, (mx, my))  # the world selector panel (W), over the editor's panels

        profiler.switch("ui")
        pygame.mouse.set_visible(False)
        draw_top(screenUI)  # message + cursor
        if sidebar_shown:
            mouse_px = pygame.mouse.get_pos()
            draw_sidebar(hud, editor, perf_hud, mouse_px, draw_top,  # native res, over the composite
                         sidebar_key(inputs, mouse_px, hud, session, editor, panel_open, time.perf_counter(),
                                     [c.version for c in look_cards]))

        # ---------------- Present: textures -> one shader pass (+ the HUD quad) -> swap
        profiler.switch("present")
        with profiler.section("upload"):
            presenter.upload()
            if sidebar_shown:
                with profiler.section("hud"):
                    presenter.upload_hud()
        with profiler.section("render"):
            presenter.render(shake)
            presenter.render_pegs(editor.peg_layer())  # native-res peg grid while placing a hook
            if sidebar_shown:
                with profiler.section("hud"):
                    presenter.render_hud()
        if MEASURE_GPU:
            with profiler.section("gpu wait", device="gpu"):
                ctx.finish()
        with profiler.section("flip"):
            display.flip()
        if frame_index == 0:  # the first frame is up: load the GPU backend while the next ones idle in clock.tick
            preload_gpu_backend()
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
