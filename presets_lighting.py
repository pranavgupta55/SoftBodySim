"""The "lighting demo" preset: a showcase of the lit renderer (materials.py, lights.py; research/lighting_contract.md).

A dark room that starts lit (world.lighting = True) with a body of every material preset: glass lenses (one fixed under
the spot light, one falling), a mirror wall, metal blocks, glowing bodies (emissive: light sources for GI), jelly, a
toon body, patterned stone blocks hanging in mid-air as shadow casters, and a pool of water octagons (FLUID) between
two stone walls. Lights: a warm point light, a coloured point light, a spot light and a weak directional "sun".
Gravity is on, so everything but the fixed pieces falls and settles. About 450 movable points, so it runs smoothly.

Without the renderer pieces (lighting off, or gl_world without effects) it is an ordinary world: everything here is
plain data, and it builds and steps headless.
"""
import math

import numpy as np

from entities import FixedBody, RigidBody, SoftBody, polygon_points, rect_points
from lights import DIRECTIONAL, POINT, SPOT, Light
from materials import PATTERN_CHECKER
from solver import World

BACKGROUND = [96, 98, 112]
AMBIENT = (0.30, 0.30, 0.36)
POOL_OCTAGON_R = 13.0  # water octagons' circumradius (world units)
POOL_ROWS = 3


def _unit(x, y):
    n = math.hypot(x, y)
    return (x / n, y / n)


def lens_points(cx, cy, width, thickness, n=7, angle=0.0):
    """A biconvex lens outline (2, 2n - 2): two circular arcs meeting at sharp ends `width` apart, `thickness` thick
    in the middle, its axis along x rotated by `angle` (radians)."""
    half_w, sag = width / 2, thickness / 2
    r = (half_w ** 2 + sag ** 2) / (2 * sag)  # radius of each arc
    a = math.asin(half_w / r)
    top = [(r * math.sin(t), -(r * math.cos(t) - (r - sag))) for t in np.linspace(-a, a, n)]
    bottom = [(x, -y) for x, y in reversed(top[1:-1])]
    pts = np.array(top + bottom).T
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return rot @ pts + np.array([[cx], [cy]])


def _rotated_rect(cx, cy, w, h, deg):
    pts = rect_points(0.0, 0.0, w, h)
    t = math.radians(deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return rot @ pts + np.array([[cx], [cy]])


def _material(body, preset, **over):
    body.material = preset
    body.material_over = dict(over)
    return body


def build_lighting_demo(world_width, world_height):
    from looping import add_world_box
    from scene import INSET, Endesga
    w, h = world_width, world_height
    world = World()
    add_world_box(world, w, h, INSET, col=Endesga.greyVD)
    world.background = list(BACKGROUND)
    world.show_grid = False
    world.lighting = True
    world.caustics = True  # gl_glass.CausticsEffect: the spot light focused through the lens
    world.ambient = AMBIENT
    floor = h - INSET
    bodies = []

    def fixed(pts, col, preset, **over):
        b = FixedBody(pts, point_col=col, stick_col=col, fill_col=col, point_radius=2,
                      stick_width=2 if pts.shape[1] == 2 else 1)
        bodies.append(_material(b, preset, **over))
        return b

    def rigid(pts, col, preset, **over):
        b = RigidBody(pts, point_col=col, stick_col=col, fill_col=col, point_radius=2)
        bodies.append(_material(b, preset, **over))
        return b

    def soft(pts, col, preset, sets=2, **over):
        b = SoftBody(pts, stiffness=0.1, skip_sets=sets, point_col=col, stick_col=col, fill_col=col, point_radius=2)
        bodies.append(_material(b, preset, **over))
        return b

    # ---- static pieces: shadow casters, the mirror wall, the fixed lens, the pool walls
    stone = Endesga.steel_grey
    fixed(rect_points(0.30 * w, 0.46 * h, 0.07 * w, 26), stone, "stone")  # a hanging slab
    fixed(rect_points(0.72 * w, 0.40 * h, 26, 0.13 * h), stone, "stone", pattern=PATTERN_CHECKER, pattern_strength=0.25)  # a pillar
    fixed(_rotated_rect(0.06 * w, 0.52 * h, 14, 0.36 * h, -12), Endesga.pale_steel, "mirror")  # the mirror wall
    fixed(lens_points(0.5 * w, 0.27 * h, 0.09 * w, 26), Endesga.very_light_blue, "glass")  # under the spot light
    pool_l, pool_r, pool_h, wall_w = 0.36 * w, 0.64 * w, 3.4 * POOL_OCTAGON_R * POOL_ROWS, 16.0
    for x in (pool_l - wall_w / 2, pool_r + wall_w / 2):
        fixed(rect_points(x, floor - pool_h / 2, wall_w, pool_h), Endesga.slate, "stone")

    # ---- the water pool: soft octagons packed between the pool walls (rows 2R apart, offset like a hex packing)
    R = POOL_OCTAGON_R + 4.0  # 8 units between neighbours (> 2 contact radii)
    x0, x1 = pool_l + R + 4.0, pool_r - R - 4.0
    rng = np.random.default_rng(3)
    lo, hi = np.array([18, 60, 130], dtype=float), np.array([90, 200, 240], dtype=float)
    for row in range(POOL_ROWS):
        cy = floor - R - 2.0 - row * math.sqrt(3) * R
        shift = R if row % 2 else 0.0
        n = int((x1 - x0 - shift) // (2 * R)) + 1
        for k in range(n):
            col = [int(c) for c in lo + rng.uniform() * (hi - lo)]
            soft(polygon_points(x0 + shift + 2 * R * k, cy, POOL_OCTAGON_R * rng.uniform(0.9, 1.0), 8,
                                angle=rng.uniform(0, math.pi / 4)), col, "water", sets=3)

    # ---- falling bodies, a preset each (a few twice)
    soft(polygon_points(0.15 * w, 0.18 * h, 28, 12), Endesga.grass_green, "jelly")
    soft(polygon_points(0.24 * w, 0.10 * h, 24, 12), Endesga.orchid, "jelly")
    soft(polygon_points(0.84 * w, 0.14 * h, 26, 6), Endesga.amber, "toon")
    soft(polygon_points(0.20 * w, 0.62 * h, 18, 12), Endesga.orange, "glow", emissive=(1.0, 0.55, 0.2),
         emissive_strength=1.5)
    soft(polygon_points(0.86 * w, 0.58 * h, 18, 12), Endesga.cyan, "glow", emissive=(0.3, 0.9, 1.0),
         emissive_strength=1.5)
    rigid(rect_points(0.78 * w, 0.70 * h, 48, 48), Endesga.pale_steel, "metal")
    rigid(polygon_points(0.92 * w, 0.36 * h, 24, 6, angle=0.3), Endesga.amber, "metal",
          tint=(1.0, 0.8, 0.45), tint_strength=0.6)  # brass
    rigid(rect_points(0.62 * w, 0.14 * h, 56, 30), Endesga.clay, "stone")
    rigid(lens_points(0.40 * w, 0.12 * h, 70, 20), Endesga.very_light_blue, "glass")
    rigid(polygon_points(0.12 * w, 0.82 * h, 22, 5), Endesga.red, "glossy")
    soft(polygon_points(0.92 * w, 0.84 * h, 22, 7), Endesga.sand, "matte")

    world.add_many(bodies)
    for wall in getattr(world, "walls", {}).values():  # the box walls would shade the whole room from the sun
        _material(wall, "matte", casts_shadow=False)

    world.lights = [
        Light(kind=POINT, pos=(0.22 * w, 0.42 * h), color=(1.0, 0.72, 0.42), intensity=1.3, radius=0.75 * w,
              falloff=1.3, softness=10.0, height=60.0),  # warm
        Light(kind=POINT, pos=(0.82 * w, 0.40 * h), color=(0.45, 0.55, 1.0), intensity=1.1, radius=0.6 * w,
              falloff=1.3, softness=6.0, height=50.0),  # cool blue
        Light(kind=SPOT, pos=(0.5 * w, 0.05 * h), dir=_unit(0.0, 1.0), color=(1.0, 0.97, 0.9), intensity=1.8,
              radius=0.9 * h, cone_inner=12.0, cone_outer=22.0, softness=4.0, height=80.0),  # through the lens
        # a weak sun (pos: only where its gizmo sits)
        Light(kind=DIRECTIONAL, pos=(0.64 * w, 0.06 * h), dir=_unit(0.35, 1.0), color=(0.85, 0.9, 1.0), intensity=0.15,
              softness=2.0, height=200.0),
    ]
    return world
