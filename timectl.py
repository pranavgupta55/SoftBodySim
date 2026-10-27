"""Slow motion / fast forward and frame stepping for main_opengl.py (keys [ / ], the Controls card's - / + buttons, N).

The time scale multiplies the physics time only; rendering, the editor and the profiler keep running at the display
rate. The physics runs in fixed ticks of one substep, 1 / (60 * substeps) s (1/240 s at the default 4 substeps; the
world's own substeps: EnvConfig or its env overrides), and World.step(dt, scale) runs as many whole ticks as
dt * scale holds, carrying the rest to the next frame. So a time scale changes how many ticks run per real second,
never the ticks themselves: every speed follows the same trajectory, tick for tick. With the default 4 substeps:

    0.25x   60 ticks of 1/240 s per real second   (about one every 4th frame at 160 fps)
    1x      240 ticks per second                   (1.5 per frame at 160 fps, 4 per frame at 60 fps)
    2x      480 ticks per second                   (twice the physics cost)
    100x    24000 ticks per second                 (100 times the cost: past EnvConfig.physics_budget a frame runs
                                                    fewer ticks than asked and the rest is dropped, so the delivered
                                                    speed, `actual` below, falls short of the request)

Frame stepping (N, while paused) runs one 1/60 s frame at the current scale: round(substeps x scale) ticks, at least
one, so 1x + N steps 4 ticks and 0.25x + N a single 1/240 s tick at a time (World.step_frame). Collisions are
checked (CCD) once per span of `substeps` ticks counted from tick 0, so a frame that ends mid-span (0.25x + N, or
1.5 ticks per frame at 160 fps) can show the substeps' unchecked proposal, e.g. a point slightly inside an edge,
which the span's end then resolves.
"""
from entities import EnvConfig

# 1% .. 10,000%: 1-2-5 steps below 100%, finer ones just above it (where speeding up is still cheap)
SCALES = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0)


class TimeControl:
    def __init__(self, scales=SCALES, scale=1.0):
        self.scales = tuple(scales)
        self.index = self.scales.index(scale)
        self.pending = 0  # frame steps requested (N) and not yet run
        self.actual = None  # delivered speed (simulated / wall seconds) over the last WINDOW s of running frames
        self._window = []  # [(wall s, simulated s)] of recent running frames

    WINDOW = 1.0  # s of running frames the delivered speed is averaged over

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

    def note(self, wall, sim):
        """A running frame took `wall` seconds and simulated `sim` (World.last_sim_dt), for the delivered speed."""
        self._window.append((wall, sim))
        total = sum(w for w, _ in self._window)
        while len(self._window) > 1 and total - self._window[0][0] >= self.WINDOW:
            total -= self._window.pop(0)[0]
        self.actual = sum(t for _, t in self._window) / total if total > 0 else None

    def reset_actual(self):
        """Forget the delivered speed (paused: nothing is being delivered)."""
        self._window.clear()
        self.actual = None

    def readout(self, short=False):
        """The Controls card's speed text: "speed 500%", or "speed 500% (actual 380%)" while the sim delivers under
        95% of the request (a slow machine, the max_dt clamp or EnvConfig.physics_budget). short: "500%" /
        "500% (380%)", for a narrow card."""
        lagging = self.actual is not None and self.actual < 0.95 * self.scale
        extra = (f" ({self.actual * 100:.0f}%)" if short else f" (actual {self.actual * 100:.0f}%)") if lagging else ""
        return (self.label() if short else f"speed {self.label()}") + extra

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
        """Ticks one N press runs at the current scale (World.step_frame), for `base` substeps (default EnvConfig's)."""
        return max(1, round((EnvConfig.substeps if base is None else base) * self.scale))
