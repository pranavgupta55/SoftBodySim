"""Watch one chunk stuckness test (chunk_test.py) in real time: the very same test world (ChunkTest, same seed,
same step loop), drawn with pygame in the map's palette.

    python3 chunk_viewer.py ["evolved 1 halloween"] [--chunk "grid r0c1"] [--seed 1] [--fill 0.5] [--seconds 5]
                            [--layout FILE.json]

--layout: show another layout's tiles (a map file's "tiles" keys, e.g. an evolved map, or a path) in the map's
settings (spawn, tiling, palette); its keys must be tiles of the map's tiling.

Keys:
    Left / Right    previous / next chunk (row by row)       Up / Down   chunk one row up / down
    G               next grid (grid -> shift_x -> shift_y -> shift_xy)
    S / A           next / previous seed (restarts)          R           restart (same seed)
    Space           pause / resume                           N           one frame (1/60 s) while paused
    1 / 2 / 4       speed 1x / 2x / 4x                       T           outline the shapes still in the chunk
                                                             (red: slower than SLOW_SPEED over the last SLOW_WINDOW)
    Esc / Q         quit
At 1x one sim second takes one real second (frames of 1/60 s, each World.step(1/60) = 4 ticks of 1/240 s, run as
real time accrues). The test stops at its end time and holds the final state and score; R / S run it again.
--frames N --screenshot PATH: run N frames headless-fast, save the window and quit (a check that it draws).
"""
import argparse
import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import chunk_test as ct  # noqa: E402

WINDOW_W = 1280  # px: the chunk world is scaled to this width
HEADER = 54  # px of text above the world
RECT_COL = (255, 220, 90)
STUCK_COL = (255, 40, 60)
MOVING_COL = (255, 255, 255)
MAX_FRAMES_PER_DRAW = 16  # a slow machine drops sim time instead of spiralling


def colour(test, info):
    """The map's palette on the test world's bodies (tiles: "diamond", shapes: "hexagon"), as the app shows it."""
    import scene
    from palettes import DEFAULT, apply_palette
    scene.apply_map_lighting(test.world, info.lighting)
    apply_palette(test.world, info.palette or DEFAULT)


def draw(screen, font, test, info, zoom, state):
    world = test.world
    bg = getattr(world, "background", None) or (30, 30, 40)
    screen.fill(bg)
    surf = screen.subsurface((0, HEADER, screen.get_width(), screen.get_height() - HEADER))
    sp = test.speeds() if state["tint"] else {}
    for b in world.bodies:
        if b.points.shape[1] < 3:
            continue
        b.draw_filled(surf, zoom=zoom)
    if state["tint"]:
        for b in test.remaining:
            loop = np.round(b.points * zoom).astype(int).T.tolist()
            # red: under SLOW_SPEED over the last SLOW_WINDOW (settled: stuck); white: still in the chunk, moving
            col = STUCK_COL if test.time >= ct.SLOW_WINDOW and sp.get(id(b), 0.0) < ct.SLOW_SPEED else MOVING_COL
            pygame.draw.polygon(surf, col, loop, 2)
    x0, y0, w, h = test.rect
    pygame.draw.rect(surf, RECT_COL, pygame.Rect(round(x0 * zoom), round(y0 * zoom), round(w * zoom),
                                                 round(h * zoom)), 2)
    sc = test.score()
    slow = sum(v < ct.SLOW_SPEED for v in test.speeds().values())
    name = info.name + (f" + layout {state['layout_name']}" if state.get("layout_name") else "")
    line1 = (f"{name}  |  chunk {test.chunk['id']}  |  seed {test.seed}  |  t {test.time:5.2f} / "
             f"{test.seconds:g} s  |  speed {state['speed']}x{'  |  PAUSED' if state['paused'] else ''}"
             f"{'  |  DONE' if test.done else ''}")
    line2 = (f"shapes {len(test.shapes)} (fill {test.fill_achieved:.2f})  cleared {len(test.cleared)}  in chunk "
             f"{len(test.remaining)} (slow {slow})  |  score {sc['score']:.3f}  area {sc['area_score']:.3f}  "
             f"flow {sc['flow']:.3f}  |  tiles {len(test.tile_bodies)}")
    for i, text in enumerate((line1, line2)):
        screen.blit(font.render(text, True, (235, 235, 235)), (10, 6 + 24 * i))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("map", nargs="?", default="evolved 1 halloween")
    p.add_argument("--chunk", default="grid r0c0", help='chunk id, e.g. "grid r1c2" or "shift_xy r0c3"')
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fill", type=float, default=ct.FILL)
    p.add_argument("--seconds", type=float, default=ct.SECONDS)
    p.add_argument("--vel", type=float, default=ct.VEL)
    p.add_argument("--frames", type=int, default=None, help="run this many frames as fast as possible, then quit")
    p.add_argument("--screenshot", default=None, help="with --frames: save the last frame here")
    p.add_argument("--layout", default=None,
                   help="a layout file (or map name) whose tiles to show instead of the map's own (same tiling)")
    args = p.parse_args(argv)
    if args.frames is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    info = ct.MapInfo(args.map)
    keys, layout_name = None, None
    if args.layout:
        from diamonds_map import load_layout
        path = ct.map_path(args.layout)
        keys = frozenset(ct.norm_key(k) for k in load_layout(path)[0])
        layout_name = os.path.splitext(os.path.basename(path))[0]
        info.layout_tiles(keys)  # fails early on keys that aren't tiles of the map's tiling
    chunks = ct.chunk_rects(info.size)
    grids = list(ct.GRIDS)
    idx = next(i for i, c in enumerate(chunks) if c["id"] == args.chunk)
    state = {"seed": args.seed, "speed": 1, "paused": False, "tint": True, "layout_name": layout_name}

    def build():
        test = ct.ChunkTest(info, chunks[idx], state["seed"], fill=args.fill, seconds=args.seconds, vel=args.vel,
                            keys=keys)
        colour(test, info)
        return test

    ct.run_chunk(info, chunks[idx], state["seed"], seconds=0.05)  # warm up (numba) on a throwaway copy
    test = build()
    zoom = WINDOW_W / test.world_size[0]
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, HEADER + int(round(test.world_size[1] * zoom))))
    pygame.display.set_caption("chunk stuckness test")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    acc = 0.0
    n = ct.CHUNKS
    frames_left = args.frames
    running = True
    while running:
        dt = clock.tick(120) / 1000.0
        rebuild = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                g, cell = divmod(idx, n * n)
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_RIGHT:
                    idx, rebuild = g * n * n + (cell + 1) % (n * n), True
                elif k == pygame.K_LEFT:
                    idx, rebuild = g * n * n + (cell - 1) % (n * n), True
                elif k == pygame.K_DOWN:
                    idx, rebuild = g * n * n + (cell + n) % (n * n), True
                elif k == pygame.K_UP:
                    idx, rebuild = g * n * n + (cell - n) % (n * n), True
                elif k == pygame.K_g:
                    idx, rebuild = ((g + 1) % len(grids)) * n * n + cell, True
                elif k == pygame.K_s:
                    state["seed"] += 1
                    rebuild = True
                elif k == pygame.K_a:
                    state["seed"] = max(0, state["seed"] - 1)
                    rebuild = True
                elif k == pygame.K_r:
                    rebuild = True
                elif k == pygame.K_SPACE:
                    state["paused"] = not state["paused"]
                elif k == pygame.K_n and state["paused"] and not test.done:
                    test.step()
                elif k in (pygame.K_1, pygame.K_2, pygame.K_4):
                    state["speed"] = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_4: 4}[k]
                elif k == pygame.K_t:
                    state["tint"] = not state["tint"]
        if rebuild:
            test, acc = build(), 0.0
            clock.tick()  # the build's wall time isn't sim time
        if frames_left is not None:  # headless check: fixed frames, no real-time pacing
            for _ in range(min(frames_left, test.frames - test.frame)):
                test.step()
            frames_left = 0
            running = False
        elif not state["paused"] and not test.done:
            acc += dt * state["speed"]
            steps = 0
            while acc >= 1.0 / ct.FPS and not test.done and steps < MAX_FRAMES_PER_DRAW * state["speed"]:
                test.step()
                acc -= 1.0 / ct.FPS
                steps += 1
            if steps >= MAX_FRAMES_PER_DRAW * state["speed"]:
                acc = 0.0  # behind: drop the backlog rather than spiral
        else:
            acc = 0.0
        draw(screen, font, test, info, zoom, state)
        pygame.display.flip()
    if args.screenshot:
        pygame.image.save(screen, args.screenshot)
        res = test.result()
        print({k: v for k, v in res.items() if k != "stuck"})
    pygame.quit()


if __name__ == "__main__":
    main()
