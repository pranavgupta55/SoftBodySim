"""Slow motion / fast forward and frame stepping for main_opengl.py (keys [ / ], the Controls card's - / + buttons, N).

The time scale multiplies the physics dt only; rendering, the editor and the profiler keep running at the display
rate. To keep slow motion smooth and the physics the same, the substep length h stays what it is at 1x: World.step(dt,
scale) runs round(substeps * s) substeps (at least 1) over s times the frame's dt, and the hitch clamp (max_dt) scales
the same way (substeps / max_dt are the world's own: EnvConfig or its env overrides). With the default 4 substeps:

    0.25x   1 substep  of 1/240 s per frame   (the 1x trajectory, a quarter as far per frame; 1/4 the physics cost)
    0.5x    2 substeps of 1/240 s per frame
    1x      4 substeps of 1/240 s per frame   (exactly World.step's defaults)
    2x      8 substeps of 1/240 s per frame   (twice the physics cost per frame)
    100x    400 substeps of 1/240 s per frame (100 times the cost: the frame rate drops if the machine can't keep up)

Below 1/substeps (0.25x at 4 substeps) the single substep shrinks instead: 0.01x runs 1 substep of 1/6000 s per frame.

Frame stepping (N, while paused) advances one display frame's worth of physics at the current scale: 1/60 s x scale,
with the same substeps as a running frame, so 0.25x + N steps a single 1/240 s substep at a time.
"""
from entities import EnvConfig

# 1% .. 10,000% in 1-2-5 steps (plus 25%, the old slowest)
SCALES = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


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

    def can_slower(self):
        return self.index > 0

    def can_faster(self):
        return self.index < len(self.scales) - 1

    def label(self):
        """The scale as a percentage, as shown on the Controls card, e.g. "1%", "25%", "100%", "10000%"."""
        return f"{self.scale * 100:g}%"

    def request_step(self):
        self.pending += 1

    def take_steps(self):
        """Frame steps to run now (clears the request)."""
        n, self.pending = self.pending, 0
        return n

    def step_args(self, dt):
        """World.step arguments (dt, scale) for `dt` seconds of display time at the current scale."""
        return dt, self.scale

    def substeps(self, base=None):
        """Substeps per frame at the current scale, for `base` substeps at 1x (default EnvConfig.substeps)."""
        return max(1, round((EnvConfig.substeps if base is None else base) * self.scale))
