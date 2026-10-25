"""Shared setup for main_opengl.py: palette, view settings and the test scene.

Importing this module has no side effects beyond defining things (no display, no game loop).
"""
import math
import os

import numpy as np

from entities import RigidBody, SoftBody, FixedBody, Rope, rect_points, polygon_points
from looping import add_world_box
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
    """View settings: debug toggles (keys G, V and T) and the world -> render surface zoom."""
    show_grid = True  # G: draw the collision spatial hash grid under the bodies
    filled = True  # V: filled polygons with a black border instead of points + springs
    trails = False  # T: fading motion trails behind moving bodies (trails.py)
    zoom = 0.5  # render pixels per world unit: 0.5 shows twice as much world each way
    # share of the screen width taken by the right-hand sidebar (all the cards); the world fills the rest. The sidebar
    # is at least wide enough for legible cards (ui_style.sidebar_rect), e.g. 158 px (12%) on a 1280 px screen
    sidebar_frac = 0.10


INSET = 5  # the world box walls sit this many world units in from each edge


def build_scene(world_width, world_height, inset=INSET):
    """Test scene: world box, ground platform, rigid square, soft pentagon and dodecagons, ropes and a hollow box."""
    world_center = [world_width / 2, world_height / 2]
    world = World()
    # World box: four tagged 2-point fixed walls (segments) sitting 5px in from each screen edge (looping.add_world_box)
    add_world_box(world, world_width, world_height, inset, col=Endesga.greyD)
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
    hang_x, box_half = world_width * 0.62, 8  # clear of the container (left)
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


# ENDESGA 32 (https://lospec.com/palette-list/endesga-32), for random body colors
ENDESGA_32 = [Endesga.rust_red, Endesga.burnt_orange, Endesga.sand, Endesga.tan, Endesga.clay, Endesga.brown,
              Endesga.dark_brown, Endesga.crimson, Endesga.red, Endesga.orange, Endesga.amber, Endesga.yellow,
              Endesga.grass_green, Endesga.green, Endesga.forest_green, Endesga.deep_teal, Endesga.navy_blue,
              Endesga.sky_blue, Endesga.cyan, Endesga.white, Endesga.pale_steel, Endesga.steel_grey, Endesga.slate,
              Endesga.dark_slate, Endesga.midnight_blue, Endesga.near_black, Endesga.neon_red, Endesga.plum,
              Endesga.orchid, Endesga.salmon, Endesga.peach, Endesga.caramel]
# ...minus the darkest few, which vanish against the background (Endesga.my_blue)
BRIGHT_ENDESGA_32 = [c for c in ENDESGA_32 if sum(c) > 230]

DIAMONDS_SEED = 12  # its tiling colors, shape placement and palette picks (and the random layout, without one)
DIAMOND_SIDE = 28.8  # edge length of its diamonds (world units)
DIAMOND_CHAIN_WIDTH = 120.0  # a cluster of touching diamonds is at most this wide (tall is fine): no long ledges
CUP_GAP = 40.0  # diamonds whose corners are closer than this can't form a cup either
# shape spawning, for a new layout (each saved layout keeps its own: the file's "spawn", over these defaults)
DIAMONDS_SPAWN = {
    "shapes": 168,  # falling soft shapes
    "sizes": [6.4, 14.4],  # their circumradius range (uniform)
    "sides": [6, 7],  # their side counts, taken in turn: hexagons and heptagons, half each
    "density": 1.2 * 210 / (1800 * 1169),  # diamonds per square world unit of a random layout (252 on 1800x1169)
}
# the diamonds folder: a world per saved layout (diamonds_map.save_layout), "diamonds/<file name>"; each was picked by
# running variants and keeping the one that left the shapes most evenly spread (its note says how)
DIAMONDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "diamonds")
DIAMONDS_LAYOUT = os.path.join(DIAMONDS_DIR, "evolved 1.json")  # the default one


def diamonds_spawn(layout=None, spawn=None):
    """The shape spawning settings for a diamonds world: DIAMONDS_SPAWN, then the layout file's own, then `spawn`."""
    from diamonds_map import layout_spawn
    out = dict(DIAMONDS_SPAWN)
    if isinstance(layout, str):
        out.update(layout_spawn(layout))
    out.update(spawn or {})
    return out


def build_diamonds_scene(world_width, world_height, seed=DIAMONDS_SEED, inset=INSET, layout=DIAMONDS_LAYOUT,
                         remove=0, add=0, add_seed=0, spawn=None):
    """"diamonds" preset (fullscreen in main_opengl: the panel floats over it; grid off): fixed diamonds from a
    tumbling-blocks tiling (edge DIAMOND_SIDE, its top row at the top edge; never two that form a cup,
    diamonds_map.forms_cup, even across the wraps), and soft shapes (hexagons and heptagons, half each) of random
    sizes that fall through them: how many, what sizes and sides come from diamonds_spawn (the layout file's own
    settings over DIAMONDS_SPAWN, then `spawn`). Colors come from palettes.py (the Palette card switches them,
    background and all). Every side loops, with no walls: what falls off the bottom comes back in at the top and what
    leaves one side comes in the other (looping.py). Diamonds lie wholly on screen, anywhere; the tiling's top faces
    touch the top edge and its last full row ends less than a row above the bottom edge, so rows meet across the
    top / bottom wrap about a row apart.
    layout: a saved layout file (DIAMONDS_LAYOUT) or (tile keys, world size it was made for); its diamonds that fit
    this world are used, and a bigger world than the one it was made for is topped up with random ones past it.
    None: a seeded random layout of the spawn settings' diamond density. remove / add / add_seed: a variant of it with that many diamonds
    dropped and added (diamonds_map.pick_diamonds)."""
    from diamonds_map import lattice_diamonds, load_layout, pick_diamonds, place_hexagons
    from palettes import DEFAULT, apply_palette

    spawn = diamonds_spawn(layout, spawn)

    world = World()
    add_world_box(world, world_width, world_height, inset)
    for side in ("top", "bottom", "left", "right"):
        world.set_looping(side, True)
    world.show_grid = False  # the G grid overlay starts off here (main_opengl keeps G per world)
    tiles = lattice_diamonds(world_width, world_height, DIAMOND_SIDE, seed=seed, y0=DIAMOND_SIDE)
    keys, size = load_layout(layout) if isinstance(layout, str) else layout if layout is not None else (None, None)
    diamonds = pick_diamonds(tiles, world_width, world_height, seed=seed, size=DIAMOND_SIDE, gap=CUP_GAP,
                             max_width=DIAMOND_CHAIN_WIDTH, layout=keys, layout_size=size, remove=remove, add=add,
                             add_seed=add_seed, count=int(round(spawn["density"] * world_width * world_height)))
    rng = np.random.default_rng(seed + 2)  # each body's palette_pick: which of its role's colors it gets
    bodies = []  # added in one go at the end (one repack instead of one per body)
    for corners, _ in diamonds:
        body = FixedBody(corners, point_radius=2)
        body.palette_role, body.palette_pick = "diamond", int(rng.integers(1 << 30))
        bodies.append(body)
    r_min, r_max = spawn["sizes"]
    sides = spawn["sides"]
    for k, (cx, cy, r, angle) in enumerate(place_hexagons([c for c, _ in diamonds], world_width, world_height,
                                                          spawn["shapes"], seed=seed, r_min=r_min, r_max=r_max,
                                                          side=inset)):
        n = int(sides[k % len(sides)])  # side counts taken in turn
        body = SoftBody(polygon_points(cx, cy, r, n, angle=-math.pi / 2 + angle), stiffness=0.1, skip_sets=2,
                        point_radius=2)
        body.palette_role, body.palette_pick = "hexagon", int(rng.integers(1 << 30))  # (role: the falling shapes)
        bodies.append(body)
    world.add_many(bodies)
    for wall in world.walls.values():
        wall.palette_role = "wall"
    apply_palette(world, DEFAULT)
    return world


# Preset scenes by name: builder(world_width, world_height) -> World. main_opengl.py --scene NAME picks one.
# Imported last: preset modules import from here.
from presets_cradle import build_newtons_cradle  # noqa: E402
from presets_basic import PRESETS as BASIC_PRESETS  # noqa: E402

PRESETS = {
    "default": build_scene,
    "newtons cradle": build_newtons_cradle,
}
PRESETS.update(BASIC_PRESETS)  # blank, random walls, looping (presets_basic.py)


def diamonds_presets():
    """{"diamonds/<name>": builder} for each saved layout in DIAMONDS_DIR (a folder on the Worlds panel)."""
    from functools import partial
    names = sorted(f[:-5] for f in os.listdir(DIAMONDS_DIR) if f.endswith(".json"))
    return {f"diamonds/{n}": partial(build_diamonds_scene, layout=os.path.join(DIAMONDS_DIR, n + ".json"))
            for n in names}


# the diamonds folder goes right after the default scene (first in the Worlds panel's list)
PRESETS = {"default": PRESETS["default"], **diamonds_presets(), **{k: v for k, v in PRESETS.items() if k != "default"}}
# presets built across the whole window with no sidebar; H shows / hides it as a panel floating over the world
FULLSCREEN_PRESETS = {name for name in PRESETS if name.startswith("diamonds/")}
