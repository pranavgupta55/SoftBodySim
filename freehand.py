"""Freehand strokes -> polygons, and the snapping rules, for the editor (world units).

A stroke is the list of mouse positions recorded while the button is held. stroke_polygon closes it: jitter is
simplified away (Ramer-Douglas-Peucker), self-crossings are cut out (the larger loop is kept), and the outline is
resampled to evenly spaced points. mirror_stroke_polygon treats the stroke as one half of a shape drawn from a vertical
axis through its first point and mirrors it into one closed polygon. Points are (2,) float arrays; the results are
lists of them, in outline order, ready for Editor._valid.

Snapping: snap_grid rounds to the peg grid (multiples of the spacing, aligned to the world origin); snap_polar keeps a
point's direction from an origin on 15 degree steps and its distance on multiples of the spacing.
"""
import math

import numpy as np

SIMPLIFY_EPS = 1.0  # world units: stroke wiggles smaller than this are jitter


def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _dedupe(pts, tol=1e-6):
    """(k, 2) array without consecutive repeats."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    keep = np.concatenate([[True], np.hypot(*np.diff(pts, axis=0).T) > tol])
    return pts[keep]


def simplify(pts, eps=SIMPLIFY_EPS):
    """Ramer-Douglas-Peucker on an open polyline (k, 2): drop points within eps of the line through their kept
    neighbours. The end points are always kept."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, ab, seg = pts[i], pts[j] - pts[i], pts[i + 1:j]
        length = math.hypot(*ab)
        d = np.hypot(*(seg - a).T) if length < 1e-12 else np.abs(_cross(ab, seg - a)) / length
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return pts[keep]


def area(pts):
    """Signed shoelace area of the closed polygon (k, 2)."""
    x, y = np.asarray(pts, dtype=float).T
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def crossing(pts, closed):
    """The first pair (i, j), i < j, of non-adjacent edges that properly cross, or None. Edge i runs from point i to
    point i + 1 (and the last one back to point 0 when closed)."""
    pts = np.asarray(pts, dtype=float)
    k = len(pts)
    if k < 4:
        return None
    a = pts if closed else pts[:-1]
    b = np.roll(pts, -1, axis=0) if closed else pts[1:]
    m = len(a)
    p, q = a[:, None, :], b[:, None, :]
    c, d = a[None, :, :], b[None, :, :]
    hit = (_cross(q - p, c - p) * _cross(q - p, d - p) < 0) & (_cross(d - c, p - c) * _cross(d - c, q - c) < 0)
    i, j = np.triu_indices(m, 2)
    ok = hit[i, j]
    if closed:
        ok &= ~((i == 0) & (j == m - 1))  # the first and last edges share point 0
    idx = np.flatnonzero(ok)
    return None if idx.size == 0 else (int(i[idx[0]]), int(j[idx[0]]))


def _intersection(p, q, c, d):
    t = _cross(c - p, d - c) / _cross(q - p, d - c)
    return p + t * (q - p)


def untangle_closed(pts, max_cuts=200):
    """Cut a closed outline at its crossings until it's simple, keeping the larger loop at each cut (a stroke that
    overshoots its start, or a figure eight, keeps its main loop)."""
    pts = np.asarray(pts, dtype=float)
    for _ in range(max_cuts):
        pair = crossing(pts, True)
        if pair is None:
            break
        i, j = pair
        k = len(pts)
        x = _intersection(pts[i], pts[i + 1], pts[j], pts[(j + 1) % k])
        inner = np.vstack([x, pts[i + 1:j + 1]])
        outer = np.vstack([x, pts[j + 1:], pts[:i + 1]])
        pts = inner if abs(area(inner)) > abs(area(outer)) else outer
        pts = _dedupe(pts)
        if len(pts) < 3:
            break
    return pts


def untangle_open(pts, max_cuts=200):
    """Cut the loops out of an open polyline at its crossings (its two ends stay)."""
    pts = np.asarray(pts, dtype=float)
    for _ in range(max_cuts):
        pair = crossing(pts, False)
        if pair is None:
            break
        i, j = pair
        x = _intersection(pts[i], pts[i + 1], pts[j], pts[j + 1])
        pts = _dedupe(np.vstack([pts[:i + 1], x, pts[j + 1:]]))
    return pts


def resample(pts, closed, spacing, max_n):
    """Evenly spaced points along the polyline (k, 2), about `spacing` apart (further when that would be more than
    max_n points). Closed: n >= 3 points around the loop, starting at point 0. Open: both ends kept."""
    pts = np.asarray(pts, dtype=float)
    path = np.vstack([pts, pts[:1]]) if closed else pts
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    total = s[-1]
    if closed:
        n = int(min(max(round(total / spacing), 3), max_n))
        t = np.arange(n) * total / n
    else:
        n = int(min(max(round(total / spacing), 1), max_n))
        t = np.linspace(0.0, total, n + 1)
    return np.column_stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])])


def perimeter(pts, closed=True):
    pts = np.asarray(pts, dtype=float)
    path = np.vstack([pts, pts[:1]]) if closed else pts
    return float(np.hypot(*np.diff(path, axis=0).T).sum())


def stroke_polygon(stroke, spacing, max_n):
    """The closed polygon a freehand stroke makes, as a list of points, or None if it's too small (a perimeter under
    3 spacings). Not validated against the world (the editor does that); usually free of self-crossings, but a
    narrow neck can still come out too tight, which validation then rejects."""
    raw = _dedupe(stroke)
    if len(raw) < 3:
        return None
    pts = untangle_closed(simplify(raw))
    if len(pts) < 3 or perimeter(pts) < 3 * spacing:
        return None
    for _ in range(3):  # resampling a simple outline rarely makes a crossing; cut it out and resample again if so
        out = resample(pts, True, spacing, max_n)
        if crossing(out, True) is None:
            break
        pts = untangle_closed(out)
        if len(pts) < 3:
            return None
    return list(out)


def mirror_point(p, axis_x):
    return np.array([2.0 * axis_x - p[0], p[1]])


def mirror_stroke_polygon(stroke, spacing, max_n, axis_x=None):
    """One half drawn from the axis (vertical, through the stroke's first point unless axis_x is given) mirrored into
    a closed symmetric polygon, or None if it's too small. The half is resampled (both ends kept), its first point put
    on the axis, and its last one too when it ends within half a spacing of it; the outline then runs down the half
    and back up the mirrored half, so its order is a plain loop. A half that crosses the axis gives a self-crossing
    outline, which validation rejects."""
    raw = _dedupe(stroke)
    if len(raw) < 2:
        return None
    ax = float(raw[0, 0]) if axis_x is None else float(axis_x)
    half = untangle_open(simplify(raw))
    if perimeter(half, closed=False) < 1.5 * spacing:
        return None
    half = resample(half, False, spacing, max(1, max_n // 2))
    half[0, 0] = ax
    on_axis = abs(half[-1, 0] - ax) < spacing / 2
    if on_axis:
        half[-1, 0] = ax
    back = half[-2:0:-1] if on_axis else half[-1:0:-1]
    out = [p for p in half] + [mirror_point(p, ax) for p in back]
    return out if len(out) >= 3 else None


# ---------------- Snapping
def snap_grid(p, spacing):
    """The nearest grid point (multiples of spacing, aligned to the world origin)."""
    return np.round(np.asarray(p, dtype=float) / spacing) * spacing


def snap_polar(origin, p, spacing, step_deg=15.0):
    """p moved so its direction from origin is a multiple of step_deg and its distance a multiple of spacing (at
    least one spacing)."""
    origin, p = np.asarray(origin, dtype=float), np.asarray(p, dtype=float)
    d = p - origin
    step = math.radians(step_deg)
    angle = round(math.atan2(d[1], d[0]) / step) * step
    length = max(1, round(math.hypot(*d) / spacing)) * spacing
    return origin + length * np.array([math.cos(angle), math.sin(angle)])
