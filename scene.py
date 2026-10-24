"""Shared setup for main_opengl.py: palette, view settings and the test scene.

Importing this module has no side effects beyond defining things (no display, no game loop).
"""
import numpy as np

from entities import RigidBody, SoftBody, FixedBody, Rope, rect_points, polygon_points
from solver import World


class Endesga:
    maroon_red = [87, 28, 39]
    lighter_maroon_red = [127, 36, 51]
    dark_green = [9, 26, 23]
    light_brown = [191, 111, 74]
    black = [19, 19, 19]
    grey_blue = [66, 76, 110]
    cream = [237, 171, 80]
    white = [255, 255, 255]
    greyL = [200, 200, 200]
    grey = [150, 150, 150]
    greyD = [100, 100, 100]
    greyVD = [50, 50, 50]
    very_light_blue = [199, 207, 221]
    my_blue = [32, 36, 46]
    debug_red = [255, 96, 141]
    sebastian_lague_purple = [70, 74, 124]
    sebastian_lague_light_purple = [137, 133, 181]
    network_green = [64, 128, 67]
    network_red = [127, 45, 41]
    # The rest of the ENDESGA 32 palette by Endesga (https://lospec.com/palette-list/endesga-32), in its order;
    # its white (#ffffff) is already `white` above
    rust_red = [190, 74, 47]  # be4a2f
    burnt_orange = [215, 118, 67]  # d77643
    sand = [234, 212, 170]  # ead4aa
    tan = [228, 166, 114]  # e4a672
    clay = [184, 111, 80]  # b86f50
    brown = [115, 62, 57]  # 733e39
    dark_brown = [62, 39, 49]  # 3e2731
    crimson = [162, 38, 51]  # a22633
    red = [228, 59, 68]  # e43b44
    orange = [247, 118, 34]  # f77622
    amber = [254, 174, 52]  # feae34
    yellow = [254, 231, 97]  # fee761
    grass_green = [99, 199, 77]  # 63c74d
    green = [62, 137, 72]  # 3e8948
    forest_green = [38, 92, 66]  # 265c42
    deep_teal = [25, 60, 62]  # 193c3e
    navy_blue = [18, 78, 137]  # 124e89
    sky_blue = [0, 153, 219]  # 0099db
    cyan = [44, 232, 245]  # 2ce8f5
    pale_steel = [192, 203, 220]  # c0cbdc
    steel_grey = [139, 155, 180]  # 8b9bb4
    slate = [90, 105, 136]  # 5a6988
    dark_slate = [58, 68, 102]  # 3a4466
    midnight_blue = [38, 43, 68]  # 262b44
    near_black = [24, 20, 37]  # 181425
    neon_red = [255, 0, 68]  # ff0044
    plum = [104, 56, 108]  # 68386c
    orchid = [181, 80, 136]  # b55088
    salmon = [246, 117, 122]  # f6757a
    peach = [232, 183, 150]  # e8b796
    caramel = [194, 133, 105]  # c28569


class FireCols:
    cols = [[255, 242, 157], [255, 212, 124], [255, 169, 70], [254, 128, 3], [230, 106, 37], [179, 46, 19], [126, 19, 6], [70, 2, 14]]
    colsReversed = [[70, 2, 14], [126, 19, 6], [179, 46, 19], [230, 106, 37], [254, 128, 3], [255, 169, 70], [255, 212, 124], [255, 242, 157]]


class ViewConfig:
    """View settings: debug toggles (keys G and V) and the world -> render surface zoom."""
    show_grid = True  # G: draw the collision spatial hash grid under the bodies
    filled = True  # V: filled polygons with a black border instead of points + springs
    zoom = 0.5  # render pixels per world unit: 0.5 shows twice as much world each way


INSET = 5  # the world box walls sit this many world units in from each edge


def build_scene(world_width, world_height, inset=INSET):
    """Test scene: world box, ground platform, rigid square, soft pentagon and dodecagons, ropes and a hollow box."""
    world_center = [world_width / 2, world_height / 2]
    world = World()
    # World box: four 2-point fixed walls (segments) sitting 5px in from each screen edge
    corners = [(inset, inset), (world_width - inset, inset), (world_width - inset, world_height - inset), (inset, world_height - inset)]
    for k in range(4):
        world.add(FixedBody(np.array([corners[k], corners[(k + 1) % 4]]).T, point_col=Endesga.greyD, stick_col=Endesga.greyD,
                            point_radius=1, stick_width=2, fill_col=Endesga.greyD))
    ground_top = world_height - 40
    ground_thickness = 6
    world.add(FixedBody(rect_points(world_center[0], ground_top + ground_thickness / 2, world_width * 0.5, ground_thickness),
                        point_col=Endesga.greyD, stick_col=Endesga.greyL, point_radius=2, fill_col=Endesga.grey))
    square_half = 20
    square_y = ground_top - square_half - 40
    world.add(RigidBody(rect_points(world_center[0], square_y, 2 * square_half, 2 * square_half),
                        point_col=Endesga.debug_red, stick_col=Endesga.greyL, point_radius=3, fill_col=Endesga.debug_red))
    world.add(SoftBody(polygon_points(world_center[0] + 4, square_y - square_half - 50, 20, 6),
                       stiffness=0.1, point_col=Endesga.network_green, stick_col=Endesga.very_light_blue, point_radius=3,
                       fill_col=Endesga.network_green))

    # Dodecagons (12 points, 2 sets of skip springs) with springs as strong as the hexagon, 25% weaker and 50% weaker
    hexagon_stiffness = 0.1
    for x_frac, y, strength, col in ((0.32, ground_top - 100, 1.0, Endesga.cream),
                                     (0.68, ground_top - 100, 0.75, Endesga.sebastian_lague_light_purple),
                                     (0.60, max(ground_top - 180, inset + 25), 0.5, Endesga.lighter_maroon_red)):  # clamp: short screens
        world.add(SoftBody(polygon_points(world_width * x_frac, y, 18, 12),
                           stiffness=hexagon_stiffness * strength, skip_sets=2, point_col=col, stick_col=Endesga.very_light_blue, point_radius=3, fill_col=col))

    # Ropes: a small rigid box hung from the top wall by two ropes in a V, and a rope from the square to the cream dodecagon
    square, cream_dodecagon = world.bodies[5], world.bodies[7]
    hang_x, box_half = world_width * 0.62, 8  # clear of the container (left) and the HUD cards (top right)
    anchor = (hang_x, inset)
    hanging_box = RigidBody(rect_points(hang_x, inset + 50 + box_half, 2 * box_half, 2 * box_half),
                            point_col=Endesga.cream, stick_col=Endesga.greyL, point_radius=2)
    world.add(hanging_box)
    for corner in (0, 1):  # top-left, top-right
        world.add_rope(Rope(anchor, (hanging_box, corner), col=Endesga.light_brown))
    world.add_rope(Rope((square, 0), (cream_dodecagon, 3), col=Endesga.light_brown))

    # Hollow rigid box near the left wall with a soft dodecagon inside it; drag the box to slosh it around.
    # The container is added first so it draws underneath its contents.
    box_cx, box_cy = inset + 50, world_height * 0.3
    world.add(RigidBody(rect_points(box_cx, box_cy, 90, 70), point_col=Endesga.light_brown, stick_col=Endesga.greyL, point_radius=3))
    world.add(SoftBody(polygon_points(box_cx, box_cy, 20, 12), stiffness=hexagon_stiffness, skip_sets=2,
                       point_col=Endesga.network_red, stick_col=Endesga.very_light_blue, point_radius=3))
    return world


# Scene presets: name -> builder(world_width, world_height) -> World. Imported last: preset modules import from here.
from presets_cradle import build_newtons_cradle  # noqa: E402

PRESETS = {
    "default": build_scene,
    "newtons cradle": build_newtons_cradle,
}
