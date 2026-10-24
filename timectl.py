"""Slow motion and frame stepping for main_opengl.py (keys [ / ] and N).

The time scale multiplies the physics dt only; rendering, the editor and the profiler keep running at the display
rate. To keep slow motion smooth and the physics the same, the substep length h stays what it is at 1x: a frame at
scale s runs round(EnvConfig.substeps * s) substeps (at least 1) over s times the frame's dt, and the hitch clamp
(EnvConfig.max_dt) scales the same way. So with the default 4 substeps per frame:

    0.25x   1 substep  of 1/240 s per frame   (the 1x trajectory, a quarter as far per frame; 1/4 the physics cost)
    0.5x    2 substeps of 1/240 s per frame
    1x      4 substeps of 1/240 s per frame   (exactly World.step's defaults)
    2x      8 substeps of 1/240 s per frame   (twice the physics cost per frame)

Frame stepping (N, while paused) advances one display frame's worth of physics at the current scale: 1/60 s x scale,
with the same substeps as a running frame, so 0.25x + N steps a single 1/240 s substep at a time.
"""
from entities import EnvConfig

SCALES = (0.25, 0.5, 1.0, 2.0)


class TimeControl:
    def __init__(self, scales=SCALES, scale=1.0):
        self.scales = tuple(scales)
        self.index = self.scales.index(scale)
        self.pending = 0  # frame steps requested (N) and not yet run

    @property
    def scale(self):
        return self.scales[self.index]

    def slower(self):
        self.index = max(self.index - 1, 0)

    def faster(self):
        self.index = min(self.index + 1, len(self.scales) - 1)

    def label(self):
        """The scale as shown on the info card, e.g. "0.25x", "1x"."""
        return f"{self.scale:g}x"

    def request_step(self):
        self.pending += 1

    def take_steps(self):
        """Frame steps to run now (clears the request)."""
        n, self.pending = self.pending, 0
        return n

    def step_args(self, dt):
        """World.step arguments (dt, substeps, max_dt) for `dt` seconds of display time at the current scale."""
        s = self.scale
        return dt * s, max(1, round(EnvConfig.substeps * s)), EnvConfig.max_dt * s
