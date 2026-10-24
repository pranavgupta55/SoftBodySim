"""Newton's cradle preset: a row of identical 20-sided rigid balls, each hung from a fixed frame by two strong ropes.

Two parallel ropes per ball (a parallelogram linkage) rather than a V: in 2D a V to one hook makes each ball a rigid
pendulum that tilts as it swings, and one rope makes a wobbly double pendulum. With parallel ropes a ball translates
along an arc without turning, so neighbours always meet face to face, like a real cradle seen from the front.
"""
import math

import numpy as np

from entities import EnvConfig, FixedBody, Hook, RigidBody, Rope, polygon_points, rect_points
from solver import World

BALLS = 5
SIDES = 20
PULLED = 1  # balls pulled out at the start (from the left)
PULL_ANGLE = 45.0  # degrees from vertical; keep it below the rope attachment angle (see ATTACH)
ATTACH = 3  # ropes attach this many vertices either side of the top vertex: 54 degrees off vertical for 20 sides
RESTITUTION = 1.0  # ball-ball bounciness (see Body.restitution); 1 = perfectly elastic
# per-world physics settings (World.env): no damping or friction, more substeps for crisp impacts
ENV = {"substeps": 8, "damping": 0.0, "friction": 0.0}


def build_newtons_cradle(world_width, world_height, balls=BALLS, pulled=PULLED, angle=PULL_ANGLE,
                         restitution=RESTITUTION, env=None, inset=None):
    from scene import INSET, Endesga  # here, not at import time: scene imports this module for PRESETS
    inset = INSET if inset is None else inset
    world = World()
    world.env = dict(ENV if env is None else env)
    r_contact = world.env.get("contact_radius", EnvConfig.contact_radius)

    # world box, as in the default scene
    corners = [(inset, inset), (world_width - inset, inset), (world_width - inset, world_height - inset),
               (inset, world_height - inset)]
    for k in range(4):
        world.add(FixedBody(np.array([corners[k], corners[(k + 1) % 4]]).T, point_col=Endesga.greyD,
                            stick_col=Endesga.greyD, point_radius=1, stick_width=2, fill_col=Endesga.greyD))

    # Centred in the world. Sizes follow the world height; the ropes are shortened if the swing (plus the frame
    # posts) wouldn't fit the width.
    avail = world_width
    radius = min(0.04 * world_height, 0.035 * avail)
    pitch = 2 * radius + r_contact  # centre spacing: neighbours rest exactly at contact distance
    swing = math.sin(math.radians(max(angle, 10.0)))
    rope_len = min(0.4 * world_height, ((avail - 2 * inset - 40 - (balls - 1) * pitch) / 2 - radius - 30) / swing)
    top = 0.2 * world_height  # hook line
    cx0 = avail / 2 - (balls - 1) / 2 * pitch
    half = math.sin(math.radians(360 / SIDES * ATTACH)) * radius  # attachment half spacing
    drop = math.cos(math.radians(360 / SIDES * ATTACH)) * radius  # attachments sit this far above the centre

    # frame: a top bar over the hooks, two posts outside the swing and a base
    reach = rope_len * swing + radius + 30
    left, right = cx0 - reach, cx0 + (balls - 1) * pitch + reach
    bar_y, base_y = top - 14, top + rope_len + drop + radius + 60
    frame = dict(point_col=Endesga.greyD, stick_col=Endesga.greyL, point_radius=1, fill_col=Endesga.slate)
    world.add(FixedBody(rect_points((left + right) / 2, bar_y - 4, right - left + 16, 8), **frame))
    for x in (left, right):
        world.add(FixedBody(rect_points(x, (bar_y + base_y) / 2, 8, base_y - bar_y), **frame))
    world.add(FixedBody(rect_points((left + right) / 2, base_y + 4, right - left + 60, 8), **frame))

    cols = (Endesga.steel_grey, Endesga.pale_steel)
    rad = math.radians(angle)
    for k in range(balls):
        hx = cx0 + k * pitch
        th = -rad if k < pulled else 0.0  # pulled balls swing out to the left, rigidly together
        cx, cy = hx + rope_len * math.sin(th), top + rope_len * math.cos(th) + drop
        ball = RigidBody(polygon_points(cx, cy, radius, SIDES), point_col=cols[k % 2], stick_col=Endesga.greyL,
                         point_radius=2, fill_col=cols[k % 2])
        ball.restitution = restitution
        world.add(ball)
        for idx, dx in ((SIDES - ATTACH, -half), (ATTACH, half)):
            hook = Hook((hx + dx, top), col=Endesga.greyL)
            world.add_hook(hook)
            world.add_rope(Rope(hook, (ball, idx), rest_length=rope_len, stiffness=1.0, col=Endesga.light_brown))
    return world
