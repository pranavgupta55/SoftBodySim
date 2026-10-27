"""Lights for the lit renderer. A world holds them in `world.lights` (a list of Light) with `world.ambient` (linear
RGB); worlds without those attributes have none (see of_world). Positions are in world units (render px = world
units x zoom); directions are unit vectors in world space, +y down like the rest of the sim.

sanitize(light) is the one place a light's values are checked: everything that draws lights (gl_lights, gl_glass's
caustics) takes active(world), the sanitised enabled lights, so a NaN, a negative radius, a string colour or a zero
direction from a hand-edited scene or a slider can't reach a shader."""
import logging
import math
from dataclasses import dataclass, fields, replace

import numpy as np

POINT, SPOT, DIRECTIONAL = "point", "spot", "directional"
KINDS = (POINT, SPOT, DIRECTIONAL)
DEFAULT_AMBIENT = (0.25, 0.25, 0.3)
MAX_COLOR = 64.0  # per colour component (linear RGB; > 1 for over-bright lights)
MAX_INTENSITY = 1e4
MAX_HEIGHT = 1e4
FALLOFF_RANGE = (0.01, 16.0)
MAX_SOFTNESS = 400.0  # world units; point / spot lights also at most half their radius
MAX_AMBIENT = 4.0

log = logging.getLogger(__name__)
_logged = set()  # id(light)s already reported by sanitize


@dataclass
class Light:
    kind: str = POINT
    pos: tuple = (0.0, 0.0)  # point / spot
    dir: tuple = (0.0, 1.0)  # spot axis / directional travel direction
    color: tuple = (1.0, 0.95, 0.85)  # linear RGB
    intensity: float = 1.0
    radius: float = 400.0  # world units where a point / spot light falls to 0
    falloff: float = 2.0  # exponent of (1 - d / radius)
    cone_inner: float = 20.0  # spot: full brightness inside this half-angle (degrees)
    cone_outer: float = 35.0  # spot: 0 outside this half-angle
    softness: float = 8.0  # shadow source size (world units): bigger = softer penumbra
    height: float = 60.0  # pseudo height above the plane (world units): lights faces via the bevel normals
    casts_shadows: bool = True
    enabled: bool = True

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        """A Light from a saved dict, its values coerced like sanitize's (never dropped; `pos` is kept as saved,
        as a tuple, so a malformed one can still be refused by the loader and is dropped at draw time)."""
        try:
            items = dict(d).items()
        except (TypeError, ValueError):
            items = ()
        known = {k: (tuple(v) if isinstance(v, list) else v) for k, v in items if k in cls.__dataclass_fields__}
        lt = cls()
        for k, v in known.items():
            setattr(lt, k, v)
        if isinstance(lt.pos, (tuple, list)):
            lt.pos = tuple(_float(c, float("nan")) for c in lt.pos)
        return _coerce(lt, lt.pos)

    def copy(self, **changes):
        return replace(self, **changes)


_DEFAULTS = {f.name: f.default for f in fields(Light)}


def _float(v, default):
    """float(v), or default when it won't convert (NaN converts: callers decide what it means)."""
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return default


def _num(v, name, lo, hi):
    """Field `name`'s value as a float in [lo, hi]: the default when it won't convert or is NaN (or infinite with no
    upper bound), infinities clamped."""
    f = _float(v, float("nan"))
    if math.isnan(f) or (hi is None and math.isinf(f)):
        f = float(_DEFAULTS[name])
    f = max(f, lo)
    return f if hi is None else min(f, hi)


def _vec(v):
    """A sequence of floats (NaN for components that won't convert); a scalar is one component; None if not a
    number or sequence."""
    if isinstance(v, (str, bytes)):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_float(c, float("nan")) for c in np.ravel(np.asarray(v, dtype=object))]
    f = _float(v, None)
    return None if f is None else [f]


def _color(v):
    c = _vec(v)
    if not c:
        c = list(_DEFAULTS["color"])
    if len(c) == 1:
        c = c * 3
    c = (c + [0.0, 0.0])[:3]
    return tuple(min(max(0.0 if math.isnan(x) else x, 0.0), MAX_COLOR) for x in c)


def _dir(v):
    d = _vec(v)
    if not d or len(d) < 2 or not all(math.isfinite(x) for x in d[:2]):
        return 0.0, 1.0
    n = math.hypot(d[0], d[1])
    return (d[0] / n, d[1] / n) if n > 0 and math.isfinite(n) else (0.0, 1.0)


def _pos(v):
    """(x, y) floats (the first two of a longer sequence), or None when there aren't two finite numbers."""
    p = _vec(v)
    if not p or len(p) < 2 or not all(math.isfinite(x) for x in p[:2]):
        return None
    return p[0], p[1]


def _bool(v, name):
    try:
        return bool(v)
    except Exception:  # noqa: BLE001 (an object whose __bool__ raises)
        return _DEFAULTS[name]


def _coerce(lt, pos):
    """A copy of lt with every field but kind / pos coerced and clamped (sanitize's rules); pos as given."""
    kind = lt.kind if isinstance(lt.kind, str) else _DEFAULTS["kind"]
    radius = _num(lt.radius, "radius", 0.0, None)
    soft_max = MAX_SOFTNESS if kind == DIRECTIONAL else min(0.5 * radius, MAX_SOFTNESS)
    outer = _num(lt.cone_outer, "cone_outer", 0.0, 180.0)
    inner = min(_num(lt.cone_inner, "cone_inner", 0.0, 180.0), outer)
    return Light(kind=kind, pos=pos, dir=_dir(lt.dir), color=_color(lt.color),
                 intensity=_num(lt.intensity, "intensity", 0.0, MAX_INTENSITY), radius=radius,
                 falloff=_num(lt.falloff, "falloff", *FALLOFF_RANGE), cone_inner=inner, cone_outer=outer,
                 softness=min(_num(lt.softness, "softness", 0.0, None), soft_max),
                 height=_num(lt.height, "height", 0.0, MAX_HEIGHT),
                 casts_shadows=_bool(lt.casts_shadows, "casts_shadows"), enabled=_bool(lt.enabled, "enabled"))


def _report(lt, why, level=logging.WARNING):
    if id(lt) not in _logged:
        _logged.add(id(lt))
        log.log(level, "light %s: %s", why, getattr(lt, "__dict__", lt))


def sanitize(lt):
    """A copy of light lt with every value finite and in range, or None when it can't be drawn: disabled, an unknown
    kind, or a point / spot light without a finite position. Never raises; logs once per light (id) it drops
    (warning) or changes (info). Clamps: colour 3 components in [0, MAX_COLOR] (a scalar is repeated, short ones
    padded with 0, RGBA cut); intensity [0, 1e4]; dir unit ((0, 1) when zero or not finite); radius >= 0; falloff [0.01, 16]; softness
    [0, min(radius / 2, 400)] (directional: [0, 400]); height [0, 1e4]; cones [0, 180] with inner <= outer. A value
    that won't convert to a float (or is NaN) takes the Light default."""
    try:
        if not _bool(getattr(lt, "enabled", False), "enabled"):
            return None
        kind = getattr(lt, "kind", None)
        if kind not in KINDS:
            _report(lt, f"unknown kind {kind!r}, not drawn")
            return None
        pos = _pos(lt.pos)
        if pos is None:
            if kind != DIRECTIONAL:
                _report(lt, "position not finite, not drawn")
                return None
            pos = (0.0, 0.0)
        out = _coerce(lt, pos)
    except Exception as e:  # noqa: BLE001 (sanitize must never raise: a bad light is dropped)
        _report(lt, f"unreadable ({type(e).__name__}: {e}), not drawn")
        return None
    if any(_differs(getattr(lt, f.name, None), getattr(out, f.name)) for f in fields(Light)):
        _report(lt, "values clamped", logging.INFO)  # (info: a slider at its end or a long dir is routine)
    return out


def _differs(a, b):
    if isinstance(b, tuple):
        try:
            return len(a) != len(b) or any(float(x) != y for x, y in zip(a, b))
        except (TypeError, ValueError):
            return True
    try:
        return a != b
    except Exception:  # noqa: BLE001
        return True


def active(world):
    """The world's lights that draw: sanitize()d, disabled and invalid ones left out."""
    out = []
    for lt in of_world(world)[0]:
        s = sanitize(lt)
        if s is not None:
            out.append(s)
    return out


def _ambient(a):
    c = _vec(a)
    if not c:
        return DEFAULT_AMBIENT
    if len(c) == 1:
        c = c * 3
    c = np.nan_to_num(np.array((c + [0.0, 0.0])[:3], np.float64), nan=0.0, posinf=MAX_AMBIENT, neginf=0.0)
    return tuple(float(x) for x in np.clip(c, 0.0, MAX_AMBIENT))


def of_world(world):
    """(lights, ambient) of a world; ([], DEFAULT_AMBIENT) when it has none. The ambient is 3 finite components in
    [0, MAX_AMBIENT]; the lights are as stored (active() sanitises them)."""
    lights = getattr(world, "lights", None)
    try:
        lights = list(lights) if lights is not None else []
    except TypeError:
        lights = []
    ambient = getattr(world, "ambient", None)
    return lights, (DEFAULT_AMBIENT if ambient is None else _ambient(ambient))
