"""OpenGL version of main_screen.py: same scene, physics, editor and HUD, presented through moderngl.

Ported from Framework/mainScreenOpenGL.py: the game still draws into low-res pygame surfaces, which are uploaded to
GPU textures every frame and drawn with a fullscreen quad. On top of that port, the three layers (world, translucent
editor ghost, UI) are composited and nearest-neighbour upscaled in one fragment shader (shaders/composite.frag)
instead of three pygame.transform.scale calls and blits on the CPU.

    python3 main_opengl.py                      fullscreen, GPU compositing
    python3 main_opengl.py --window 1280x800    windowed
    python3 main_opengl.py --basic              the Framework pipeline as-is: layers composited on the CPU into one
                                                surface, uploaded as one texture, drawn by shaders/basic.frag

Needs moderngl (pip install moderngl). Uses an OpenGL 4.1 core profile context (the newest macOS offers) with
GLSL 330 shaders. Hotkeys match main_screen.py: SPACE HUD, G grid, V filled/points view, P pause, ESC quit.

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

from scene import Endesga, ViewConfig, INSET, build_scene
from editor import Editor
from profiler import Profiler
from perf_hud import PerfHUD
import ui_style

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
FPS = 60
SCALE_DOWN = 2  # real screen px per render surface px (as in main_screen.py)
GHOST_ALPHA = 100  # screenT.set_alpha(100) in main_screen.py
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

    def __init__(self, ctx, layers, scale, fb_size, background, basic=False):
        """layers: the (game, ghost, ui) pygame surfaces; scale: framebuffer px per surface px."""
        self.ctx, self.layers, self.scale, self.fb_size, self.basic = ctx, layers, scale, fb_size, basic
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


# ---------------- The game (main_screen.py's loop, presented through GLPresenter)
def main(argv=None, display=None, max_frames=None, on_frame=None):
    """argv: command line (see the module docstring). display / max_frames / on_frame are test hooks: a Display to
    render into instead of opening a window, a frame limit, and a callback(frame_index, locals) after each frame."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--window", help="windowed mode at WxH screen px instead of fullscreen, e.g. 1280x800")
    parser.add_argument("--basic", action="store_true", help="the Framework pipeline: CPU composite, one texture")
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
        screenT.set_alpha(GHOST_ALPHA)  # the CPU composite blits it, so it needs main_screen.py's surface alpha
    timer = 0
    shake = [0, 0]
    oscillating_random_thing = 0
    toggle = True
    click = False

    # framebuffer px per surface px (the window may be backed by a larger, high-DPI drawable)
    fb_scale = scaleDownFactor * display.fb_size[0] / win_w
    presenter = GLPresenter(ctx, (screen2, screenT, screenUI), fb_scale, display.fb_size, Endesga.my_blue, basic=args.basic)

    # The world is measured in its own units; the render surfaces above are its size times the zoom
    world_width, world_height = screen_width / ViewConfig.zoom, screen_height / ViewConfig.zoom
    world = build_scene(world_width, world_height)
    editor = Editor(world, Endesga, (screen_width, screen_height), (world_width, world_height), scaleDownFactor, INSET,
                    ViewConfig.zoom, view=ViewConfig)
    profiler = Profiler()
    world.profiler = profiler
    perf_hud = PerfHUD(profiler)

    # ---------------- Main Game Loop
    last_time = time.time()
    running = True
    frame_index = 0
    while running:
        profiler.frame()
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
        editor.show_card = toggle  # the info card (and its buttons) only exists while the HUD is shown
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if editor.handle_event(event):  # info card buttons, editor panels and placement come first
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
                if event.key == pygame.K_SPACE:
                    toggle = not toggle
                    editor.show_card = toggle
                if event.key == pygame.K_g:
                    ViewConfig.show_grid = not ViewConfig.show_grid
                if event.key == pygame.K_v:
                    ViewConfig.filled = not ViewConfig.filled

        # ---------------- Updating Screen
        profiler.switch("physics")
        world.drag_to((wx, wy))
        if not editor.paused:
            world.step(dt / fps)  # dt is in frames here, World wants seconds
        profiler.switch("draw")
        if ViewConfig.show_grid:
            world.draw_grid(screen2, zoom=ViewConfig.zoom)
        world.draw(screen2, filled=ViewConfig.filled, zoom=ViewConfig.zoom)
        editor.draw_world(screen2, screenT, (mx, my))
        editor.draw_ui(screenUI, (mx, my))  # panels, buttons and hints, under the HUD cards

        profiler.switch("ui")
        if toggle:
            info_card = editor.draw_info_card(screenUI, (mx, my))  # first card of the top-right column
            perf_hud.draw(screenUI, info_card.bottom + ui_style.CARD_GAP)  # compute card underneath
        editor.draw_message(screenUI)  # over the cards
        if toggle:
            pygame.mouse.set_visible(False)
            pygame.draw.circle(screenUI, Endesga.black, (mx + 1, my + 1), 2, 1)
            pygame.draw.circle(screenUI, Endesga.white, (mx, my), 2, 1)

        # ---------------- Present: textures -> one shader pass -> swap
        profiler.switch("present")
        with profiler.section("upload"):
            presenter.upload()
        with profiler.section("render"):
            presenter.render(shake)
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
