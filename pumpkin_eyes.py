"""The pumpkins' eyes: every pumpkin (pumpkins.py: a body with shape_kind "pumpkin") gets two small black eyes, in
any palette; no other shape gets any.

Where: each eye is a fixed mix (mean value coordinates) of its body's own points, so the eyes move, turn and squash
with the body. The mix is worked out from the pumpkin template (pumpkins.TEMPLATE, eyes at pumpkins.EYES: on the
face, under the stem) and is the same for every pumpkin, as mean value coordinates don't change when a shape is moved,
turned or scaled (kept on the body as `eye_rig`: (weights (2, n), half width)). Radius pumpkins.EYE_RADIUS x its
half width at rest (pumpkins.size_of; whole px at the drawing zoom, at least 1); on small bodies they are pushed
apart to keep a 1 px gap (_spread).
Drawing: World.draw (pygame) draws a body's eyes right after the body; gl_world's WorldPacker adds them as circle
stamps on the layer just above their body (lighting on: an unlit, shadowless material, so they stay black and
neither glow nor cast shadows). Both read `of_world`, cached per world until its bodies change.
"""
import weakref

import numpy as np

import pumpkins

COL = (0, 0, 0)

_cache = weakref.WeakKeyDictionary()  # world -> (key, Eyes or None, P)


def chosen(bodies):
    """Indices of the bodies that get eyes (the pumpkins; sorted)."""
    return [k for k, b in enumerate(bodies) if pumpkins.is_pumpkin(b) and not getattr(b, "fixed", False)]


def _mvc(poly, x):
    """Mean value coordinates (n,) of point x (2,) in the polygon poly (2, n): sum(w * poly) == x."""
    d = poly - x[:, None]
    r = np.maximum(np.hypot(d[0], d[1]), 1e-9)
    e = np.roll(d, -1, axis=1)
    t = np.tan(np.arctan2(d[0] * e[1] - d[1] * e[0], (d * e).sum(axis=0)) / 2)
    w = (np.roll(t, 1) + t) / r
    return w / w.sum()


WEIGHTS = np.stack([_mvc(pumpkins.TEMPLATE, e) for e in pumpkins.EYES.T])  # (2, n): one row per eye


def rig(body):
    """(weights (2, n): one row per eye, half width) for a pumpkin; kept on the body (`eye_rig`)."""
    got = getattr(body, "eye_rig", None)
    if got is not None and got[0].shape == WEIGHTS.shape:
        return got
    body.eye_rig = (WEIGHTS, pumpkins.size_of(body))
    return body.eye_rig


class Eyes:
    """The eyes of one world's bodies: body (m,) body index per eye, gidx (m, n) its body's point indices in
    World.P, w (m, n) their weights, size (m,) the body's half width (world units); rows[k]: body k's eyes."""

    def __init__(self, world, ks):
        n = pumpkins.N
        rigs = [rig(world.bodies[k]) for k in ks]
        self.body = np.repeat(np.asarray(ks, np.int64), 2)
        self.w = np.concatenate([r[0] for r in rigs]) if rigs else np.zeros((0, n))
        self.size = np.repeat([r[1] for r in rigs], 2).astype(np.float64)
        self.gidx = np.asarray(world.offsets, np.int64)[self.body][:, None] + np.arange(n)
        self.rows = {k: (2 * i, 2 * i + 2) for i, k in enumerate(ks)}

    def radii(self, zoom, sel=slice(None)):
        return np.maximum(np.round(self.size[sel] * zoom * pumpkins.EYE_RADIUS), 1.0)


def _spread(c, r):
    """Eye pairs (columns 2i, 2i + 1 of c (2, m) px) pushed apart about their midpoint until their discs (pygame's:
    2r px across) are at least 1 px apart."""
    a, b = c[:, 0::2], c[:, 1::2]
    mid, half = (a + b) / 2, (b - a) / 2
    d = np.maximum(np.hypot(half[0], half[1]), 1e-9)
    half = half * np.maximum(1.0, (r[0::2] + 0.5) / d)
    out = np.empty_like(c)
    out[:, 0::2], out[:, 1::2] = mid - half, mid + half
    return out


def of_world(world):
    """The world's Eyes (None: no pumpkins), rebuilt when its bodies change (World._rebuild rebinds P); per frame
    one key comparison."""
    if not world.bodies:
        return None
    key = (id(world.P), len(world.bodies))
    hit = _cache.get(world)
    if hit is not None and hit[0] == key and hit[2] is world.P:
        return hit[1]
    ks = chosen(world.bodies)
    eyes = Eyes(world, ks) if ks else None
    _cache[world] = (key, eyes, world.P)
    return eyes


def centres(world, P, zoom, mask=None):
    """(centres (2, m) float px, radii (m,) px, body (m,)) of the eyes from P (2, N) = World.P x zoom (or a shifted
    copy of it), only those of bodies in `mask` (bool per body) if given; None: no eyes."""
    e = of_world(world)
    if e is None:
        return None
    sel = slice(None) if mask is None else mask[e.body]
    r = e.radii(zoom, sel)
    return _spread((P[:, e.gidx[sel]] * e.w[sel]).sum(axis=2), r), r, e.body[sel]


def draw(surface, eyes, k, points, zoom):
    """pygame: body k's eyes (if it has any) for its points (2, n) (world units; a periodic copy's shifted ones)."""
    import pygame
    rows = eyes.rows.get(k)
    if rows is None:
        return
    sel = slice(*rows)
    rad = eyes.radii(zoom, sel)
    c = _spread(((np.asarray(points) * zoom)[:, None, :] * eyes.w[sel]).sum(axis=2), rad)
    for (x, y), r in zip(c.T, rad):
        pygame.draw.circle(surface, COL, (x, y), r)
