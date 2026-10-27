"""Headless smoke run of main_opengl.main's frame loop. Run from the repo root:

    .venv/bin/python3 tests/app_smoke_check.py [frames] [scene ...]

Renders into an offscreen moderngl framebuffer (main's display hook; no window) for `frames` frames (default 120) per
scene (default: penrose/evolved 1, fluid), and along the way turns on trails and debug overlays 2 and 4, holds a
circle grab, switches the time scale to 25% and then 10000%, and steps a paused world with N
(World.step_frame). Prints the ticks run, the delivered speed and the frame rate per phase. Exit status 1 on an
exception or if no ticks ran.
"""
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import moderngl  # noqa: E402

import main_opengl  # noqa: E402
from scene import ViewConfig  # noqa: E402

SIZE = (1600, 1000)


def run(name, frames):
    ctx = moderngl.create_standalone_context(require=330)
    fbo = ctx.simple_framebuffer(SIZE)
    fbo.use()
    display = main_opengl.Display(ctx, SIZE, SIZE, lambda: None)
    phase = {"name": "1x", "t0": time.perf_counter(), "ticks0": 0, "frames": 0}
    log = []

    def end_phase(world, new_name):
        wall = time.perf_counter() - phase["t0"]
        ticks = world.ticks - phase["ticks0"]
        log.append(f"  {phase['name']:7s} {phase['frames']:4d} frames {ticks:6d} ticks  {phase['frames'] / wall:6.1f} fps")
        phase.update(name=new_name, t0=time.perf_counter(), ticks0=world.ticks, frames=0)

    def on_frame(i, loc):
        world, time_ctl, editor, overlays = loc["world"], loc["time_ctl"], loc["editor"], loc["overlays"]
        phase["frames"] += 1
        if i == 0:
            editor.paused = False
            ViewConfig.trails = True
            overlays.toggle(1)
            overlays.toggle(3)
            ViewConfig.grab_area = True
            world.grab((300.0, 300.0), 80.0)  # the mouse sits at (0, 0): next frames drag the circle there
        if i == frames // 4:
            world.release()
            end_phase(world, "25%")
            time_ctl.index = time_ctl.scales.index(0.25)
        if i == frames // 2:
            end_phase(world, "10000%")
            time_ctl.index = time_ctl.scales.index(100.0)
        if i == 3 * frames // 4:
            end_phase(world, "paused+N")
            time_ctl.index = time_ctl.scales.index(1.0)
            editor.paused = True
        if i > 3 * frames // 4 and i % 5 == 0:
            time_ctl.request_step()  # N
        if i == frames - 1:
            end_phase(world, "")
            log.append(f"  total ticks {world.ticks}, actual speed {time_ctl.actual}, "
                       f"last_sim_dt {world.last_sim_dt:.5f}, backend {type(world.backend).__name__}")
            phase["world"] = world

    main_opengl.main(["--scene", name], display=display, max_frames=frames, on_frame=on_frame)
    ViewConfig.grab_area = False
    print(name)
    print("\n".join(log), flush=True)
    return phase["world"].ticks > 0


def main(argv):
    frames = int(argv[0]) if argv else 120
    names = argv[1:] or ["penrose/evolved 1", "fluid"]
    ok = all([run(name, frames) for name in names])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
