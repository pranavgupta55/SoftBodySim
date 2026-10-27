"""Lights for the lit renderer. A world holds them in `world.lights` (a list of Light) with `world.ambient` (linear
RGB); worlds without those attributes have none (see of_world). Positions are in world units (render px = world
units x zoom); directions are unit vectors in world space, +y down like the rest of the sim."""
from dataclasses import dataclass, replace

POINT, SPOT, DIRECTIONAL = "point", "spot", "directional"
DEFAULT_AMBIENT = (0.25, 0.25, 0.3)


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
        known = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def copy(self, **changes):
        return replace(self, **changes)


def of_world(world):
    """(lights, ambient) of a world; ([], DEFAULT_AMBIENT) when it has none."""
    return list(getattr(world, "lights", None) or []), tuple(getattr(world, "ambient", None) or DEFAULT_AMBIENT)
