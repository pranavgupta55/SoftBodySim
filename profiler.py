"""Tiny hierarchical frame profiler (pure Python, no pygame) feeding the compute card in perf_hud.py.

Sections nest by call order and are keyed by their path, e.g. ("physics", "ccd", "broad"). Time from sections with
the same path adds up within a frame. Every section is tagged with a device:

    "cpu"   (default) wall time spent on the CPU
    "gpu"   time the GPU spent working on our behalf (see below)
    "idle"  waiting with nothing to do (the frame cap's sleep in clock.tick)

Finished frames from the last `window` seconds (3 s) are kept in `frames`, plus running per-path sums (`totals`).

Usage
-----
    prof = Profiler()
    while running:
        prof.frame()                        # once per frame: files the previous frame, starts a new one
        with prof.section("physics"):       # context manager ...
            world.step(dt)
        prof.begin("draw"); ...; prof.end() # ... or begin / end
        prof.switch("present")              # end the open section and begin a sibling (no re-indenting loops)
        prof.switch("idle", device="idle")
        clock.tick(fps)
        prof.end()

Overhead is well under 1 us per section (two perf_counter calls and a dict update).

Reporting GPU time (for a GPU compute backend, e.g. MLX or PyTorch MPS)
----------------------------------------------------------------------
GPU work is asynchronous: a dispatch returns immediately and the time shows up later, wherever the CPU first waits
for a result. To report it honestly, measure across a device sync. Two ways:

1. A gpu-tagged section that ends with a sync, so its wall time is the GPU's work (the CPU is blocked meanwhile):

       with prof.section("ccd broad", device="gpu"):
           out = kernel(x)
           mx.eval(out)                     # MLX; or torch.mps.synchronize() for MPS

2. Time measured some other way (GPU timestamps, or around a sync you do anyway), added under the open section:

       prof.add("constraints", seconds, device="gpu")

A backend finds the profiler as `world.profiler` (None when nothing is profiling; then skip timing entirely, e.g.
`with prof.section(...) if prof else contextlib.nullcontext():`). Nested sections inside a gpu section count once.

How a frame is split: GPU time is assumed to have blocked the CPU (true for both recipes above), so it is taken out
of the enclosing CPU sections. Frame wall time = cpu + gpu + idle, where cpu also holds untracked loop time; if a
backend adds GPU time that overlapped CPU work, cpu is clamped at 0. Until any GPU time is reported, `gpu_seen` is
False and the card says "not in use".
"""
from collections import deque, namedtuple
from time import perf_counter

DEVICES = ("cpu", "gpu", "idle")

# One finished frame. sections: {path: seconds}; gpu_by_top: {top-level section name: gpu seconds inside it}
Frame = namedtuple("Frame", "end wall sections cpu gpu idle gpu_by_top")


class Profiler:
    def __init__(self, window=3.0):
        self.window = window  # seconds of history kept
        self.frames = deque()  # Frame tuples, oldest first
        self.totals = {}  # path -> seconds summed over the window
        self.counts = {}  # path -> number of frames in the window that recorded it (path dropped at 0)
        self.devices = {}  # path -> device tag (from the first time the path was seen)
        self.wall = 0.0  # summed frame wall time in the window
        self.gpu_seen = False  # has anything ever reported GPU time?
        self.last = None  # the newest Frame, or None
        self._stack = []  # open sections: (path, device, start)
        self._path = ()
        self._pending = None
        self._cur = {}
        self._frame_start = None
        self._gpu_open = 0  # open gpu sections (only the outermost one counts towards the frame's GPU total)
        self._gpu = self._idle = 0.0
        self._gpu_by_top = {}

    # ---------------- recording
    def begin(self, name, device="cpu"):
        path = self._path + (name,)
        self._path = path
        if device == "gpu":
            self._gpu_open += 1
        self._stack.append((path, device, perf_counter()))

    def end(self):
        path, device, start = self._stack.pop()
        dt = perf_counter() - start
        self._path = path[:-1]
        if device == "gpu":
            self._gpu_open -= 1
        self._record(path, dt, device)

    def switch(self, name, device="cpu"):
        """End the open section and begin `name` as its sibling."""
        self.end()
        self.begin(name, device)

    def section(self, name, device="cpu"):
        """`with prof.section(name):` times the block. Returns the profiler itself (no allocation)."""
        self._pending = (name, device)
        return self

    def __enter__(self):
        self.begin(*self._pending)
        return self

    def __exit__(self, *exc):
        self.end()

    def add(self, name, seconds, device="gpu"):
        """Record `seconds` as a child `name` of the open section, measured elsewhere (e.g. across a GPU sync)."""
        self._record(self._path + (name,), seconds, device)

    def _record(self, path, dt, device):
        cur = self._cur
        cur[path] = cur.get(path, 0.0) + dt
        if path not in self.devices:
            if device not in DEVICES:
                raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
            self.devices[path] = device
        if device == "gpu":
            self.gpu_seen = True
            if not self._gpu_open:  # outermost gpu time only (nested gpu sections are already inside it)
                self._gpu += dt
                self._gpu_by_top[path[0]] = self._gpu_by_top.get(path[0], 0.0) + dt
        elif device == "idle" and len(path) == 1:
            self._idle += dt

    def frame(self):
        """Mark a frame boundary: close any open sections, file the finished frame, drop frames older than window."""
        now = perf_counter()
        while self._stack:
            self.end()
        if self._frame_start is not None:
            wall = now - self._frame_start
            cur, gpu, idle = self._cur, self._gpu, self._idle
            f = Frame(now, wall, cur, max(wall - gpu - idle, 0.0), gpu, idle, self._gpu_by_top)
            self.frames.append(f)
            self.last = f
            self.wall += wall
            totals, counts = self.totals, self.counts
            for path, dt in cur.items():
                totals[path] = totals.get(path, 0.0) + dt
                counts[path] = counts.get(path, 0) + 1
            while self.frames and now - self.frames[0].end > self.window:
                old = self.frames.popleft()
                self.wall -= old.wall
                for path, dt in old.sections.items():
                    n = counts[path] - 1
                    if n:
                        counts[path] = n
                        totals[path] -= dt
                    else:  # gone from the window (e.g. physics while paused): drop it
                        del counts[path], totals[path]
        self._cur = {}
        self._gpu = self._idle = 0.0
        self._gpu_by_top = {}
        self._frame_start = now

    # ---------------- reading
    def stats(self):
        """Window means: fps, frame time, max frame time, cpu / gpu / idle per frame (seconds), frame count n."""
        n = len(self.frames)
        if not n:
            return dict(fps=0.0, frame=0.0, frame_max=0.0, cpu=0.0, gpu=0.0, idle=0.0, n=0)
        cpu = gpu = idle = frame_max = 0.0
        for f in self.frames:
            cpu += f.cpu
            gpu += f.gpu
            idle += f.idle
            if f.wall > frame_max:
                frame_max = f.wall
        return dict(fps=n / self.wall if self.wall > 0 else 0.0, frame=self.wall / n, frame_max=frame_max,
                    cpu=cpu / n, gpu=gpu / n, idle=idle / n, n=n)
