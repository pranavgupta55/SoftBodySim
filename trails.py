"""Motion trails (T): a fading path behind every moving body, drawn under the bodies on the world surface.

Each body leaves a trail of its centroid (mean of its points), not of its outline or every point: on the low-res
pixel surface a single 2px stroke per body reads as a clean streak in the body's colour, while per-point trails of a
12-point dodecagon turn into a tangle of spirals as it rolls, and stamped outline echoes smear into blobs. Bodies
are drawn on top, so a body at rest hides its own trail and only real motion shows.

The history is a ring of centroid samples (one per stepped frame, in world units, so zoom just scales them), and a
frame's trails are rasterised in one numpy pass: every segment is sampled at 1px steps and blended straight into the
surface's 32-bit pixels. Its cost follows the trail pixels on screen: ~0.1 ms for the 13-body scene, ~1.6-2.1 ms
for 200 bodies all falling at once with full-length trails, and nothing for bodies at rest or with T off. A fading
accumulation layer would cost a constant ~0.55 ms (fade + blit of the whole surface, measured), but can't drop one
body's trail, rescale with zoom or keep the banded look; on the GPU it would need the bodies on their own layer to
stay under them. Since both bodies and trails go into screen2, the GPU and --basic paths need no changes and
show the same pixels. The fade is in wall-clock time and
banded into a few opacity steps (FADE_BANDS) so it stays crisp like the rest of the pixel art; the newer half of a
trail is 2px wide, the older half 1px.

Breaks and clearing:
    paused        no new samples; what is there fades out, and the first sample after resuming starts a new trail
                  (so edits made while paused never draw a line from the old position)
    teleport      a centroid moving more than JUMP world units in one frame (screen looping, a moved body) breaks
                  its trail there instead of drawing a line across the screen
    bodies change spawn / delete / undo / load: a removed body's trail goes with it, a new body starts empty, the
                  others keep theirs (bodies are matched by identity)
    T off         everything is cleared
Fixed bodies never get a trail.
"""
import numpy as np
import pygame

FADE_SECONDS = 0.75  # how long a point of a trail takes to fade out
MAX_ALPHA = 0.6  # opacity of the newest part of a trail
FADE_BANDS = 4  # opacity steps between MAX_ALPHA and 0 (pixel-art banding instead of a smooth gradient)
JUMP = 120.0  # world units per frame; a bigger centroid move is a teleport and breaks the trail
MIN_SEGMENT_PX = 0.35  # segments shorter than this (surface px) are skipped: resting bodies leave nothing
HISTORY = 96  # ring size in samples: FADE_SECONDS at up to ~120 fps (at 60 fps 45 samples are in use)


class Trails:
    def __init__(self, fade_seconds=FADE_SECONDS):
        self.fade_seconds = fade_seconds
        self.enabled = False
        self.clear()

    def clear(self):
        """Drop every trail (T off, or a caller that knows all history is meaningless now)."""
        self.bodies = []
        self.pos = np.zeros((HISTORY, 0, 2))  # centroid samples, oldest first, world units
        self.gap = np.ones((HISTORY, 0), dtype=bool)  # True: no segment from the previous sample to this one
        self.stamp = np.full(HISTORY, -np.inf)  # time of each sample row (seconds of self.now)
        self.cols = np.zeros((0, 3))
        self.moving = np.zeros(0, dtype=bool)  # per body: not fixed
        self.now = 0.0
        self.resumed = True  # the next sample starts new trails (after a pause, or at the start)

    def set_enabled(self, on):
        """Follow the T toggle (ViewConfig.trails); switching either way starts from nothing."""
        if bool(on) != self.enabled:
            self.enabled = bool(on)
            self.clear()

    # ---------------- History
    def _sync(self, world):
        """Match the history columns to world.bodies by identity (spawns, deletes, undo and load change the list)."""
        bodies = world.bodies
        if len(bodies) == len(self.bodies) and all(a is b for a, b in zip(bodies, self.bodies)):
            return
        pos, gap = np.zeros((HISTORY, len(bodies), 2)), np.ones((HISTORY, len(bodies)), dtype=bool)
        old = {id(b): k for k, b in enumerate(self.bodies)}  # self.bodies keeps these alive, so ids are unique
        for j, b in enumerate(bodies):
            k = old.get(id(b))
            if k is not None:
                pos[:, j], gap[:, j] = self.pos[:, k], self.gap[:, k]
        self.pos, self.gap, self.bodies = pos, gap, list(bodies)
        self.cols = np.array([b.fill_col for b in bodies], dtype=float).reshape(-1, 3)
        self.moving = np.array([not getattr(b, "fixed", False) for b in bodies], dtype=bool)

    def update(self, world, dt, stepped):
        """Once per frame after physics. dt: frame time in seconds; stepped: did the world step this frame (False
        while paused, so nothing is sampled and the trails fade out)."""
        if not self.enabled:
            return
        self.now += dt
        if not stepped:
            self.resumed = True
            return
        self._sync(world)
        if not self.bodies:
            return
        counts = world.offsets[1:] - world.offsets[:-1]
        centroid = (np.add.reduceat(world.P, world.offsets[:-1], axis=1) / counts).T  # (bodies, 2)
        jump = np.hypot(*(centroid - self.pos[-1]).T) > JUMP
        self.pos = np.roll(self.pos, -1, axis=0)
        self.gap = np.roll(self.gap, -1, axis=0)
        self.stamp = np.roll(self.stamp, -1)
        self.pos[-1], self.stamp[-1] = centroid, self.now
        self.gap[-1] = jump | self.resumed
        self.resumed = False

    # ---------------- Drawing
    def draw(self, surface, zoom=1.0):
        """Blend the trails into `surface` (call before the bodies are drawn, so they sit under them)."""
        if not self.enabled or not self.bodies:
            return
        life = 1.0 - (self.now - self.stamp) / self.fade_seconds  # per sample row: 1 new .. 0 faded
        rows = np.flatnonzero(life[:-1] > 0) + 1  # segments (row - 1 -> row) whose start hasn't faded yet
        if rows.size == 0:
            return
        a, b = self.pos[rows - 1], self.pos[rows]  # (segments, bodies, 2) world units
        length = np.abs(b - a).max(axis=2) * zoom  # surface px
        ok = ~self.gap[rows] & self.moving[None, :] & (length >= MIN_SEGMENT_PX)
        sel = np.flatnonzero(ok)  # row-major: older segments first, so newer pixels are written last
        if sel.size == 0:
            return
        seg_k, seg_b = np.divmod(sel, ok.shape[1])
        # per segment: start, step, body, banded opacity (from the life at its middle) and stroke width
        f4 = np.float32  # per-pixel math in float32 / int32: half the memory traffic of float64
        a = (a.reshape(-1, 2)[sel] * zoom).astype(f4)
        d = (b.reshape(-1, 2)[sel] * zoom).astype(f4) - a
        length = length.reshape(-1)[sel]
        fresh = 0.5 * (np.maximum(life[rows - 1], 0.0) + life[rows])[seg_k]
        alpha = (np.ceil(fresh * FADE_BANDS) * (256 * MAX_ALPHA / FADE_BANDS)).astype(np.int32)  # 1/256ths
        # 2px wide for the newer half of a trail, 1px for the older half: the second pixel is beside the sample
        # across the segment's major axis (below for mostly horizontal segments, right for mostly vertical ones)
        wide = fresh > 0.5
        steep = (np.abs(d[:, 1]) > np.abs(d[:, 0])).astype(np.int32)
        # every segment sampled at 1px steps (its end point is the next segment's start, so it's left out)
        n = np.ceil(length).astype(np.int32)
        owner = np.repeat(np.arange(n.size, dtype=np.int32), n)
        t = (np.arange(owner.size, dtype=np.int32) - np.repeat(np.cumsum(n) - n, n)) / n[owner].astype(f4)
        xi = np.floor(a[owner, 0] + t * d[owner, 0]).astype(np.int32)
        yi = np.floor(a[owner, 1] + t * d[owner, 1]).astype(np.int32)
        extra = np.flatnonzero(wide[owner])
        side = steep[owner[extra]]
        xs = np.concatenate([xi, xi[extra] + side])
        ys = np.concatenate([yi, yi[extra] + 1 - side])
        seg = np.concatenate([owner, owner[extra]])
        w, h = surface.get_size()
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        self._blend(surface, xs[inside], ys[inside], seg[inside], alpha, seg_b)

    def _blend(self, surface, xs, ys, seg, alpha, body):
        """pixel = (pixel * (256 - a) + colour * a) / 256 on the RGB channels of pixels (xs, ys), where pixel k
        belongs to segment seg[k] of opacity alpha[seg] and body body[seg]. One gather and one scatter on the 32-bit
        pixels (a pixels3d gather of the 3 channels is several times slower)."""
        px = pygame.surfarray.pixels2d(surface)  # (w, h) view of the uint32 pixels
        flat = px.T.reshape(-1) if px.T.flags.c_contiguous else None  # rows unpadded: index the buffer linearly
        idx = ys * px.shape[0] + xs if flat is not None else None
        under = flat[idx] if flat is not None else px[xs, ys]  # uint32
        shifts = surface.get_shifts()[:3]
        out = under & np.uint32(0xFFFFFFFF ^ sum(0xFF << s for s in shifts))  # alpha (and unused) bits kept
        keep = (256 - alpha)[seg]
        tint = self.cols.astype(np.int32)[body] * alpha[:, None]  # per segment: colour * opacity
        for c, shift in enumerate(shifts):
            u = ((under >> np.uint32(shift)) & np.uint32(0xFF)).astype(np.int32)
            out |= ((u * keep + tint[seg, c]) >> 8).astype(np.uint32) << np.uint32(shift)
        if flat is not None:
            flat[idx] = out
        else:
            px[xs, ys] = out
        del px, flat  # unlocks the surface
