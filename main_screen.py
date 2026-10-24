import pygame
import math
import time
import random
from text import drawText
from fontDict import fonts
import numpy as np
from entities import RigidBody, SoftBody, FixedBody, Rope, rect_points, polygon_points
from solver import World

pygame.init()

# ---------------- Setting up the screen, assigning some global variables, and loading text fonts
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
clock = pygame.time.Clock()
fps = 60
scaleDownFactor = 3
screen_width = int(screen.get_width() / scaleDownFactor)
screen_height = int(screen.get_height() / scaleDownFactor)
screen_center = [screen_width / 2, screen_height / 2]
screen2 = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenT = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenT.set_alpha(100)
screenUI = pygame.Surface((screen_width, screen_height)).convert_alpha()
timer = 0
shake = [0, 0]
shake_strength = 3
montserratRegularAdaptive = fonts[f"regular{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]
montserratExtralightAdaptive = fonts[f"extralight{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]
montserratBoldAdaptive = fonts[f"bold{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]
montserratThinAdaptive = fonts[f"thin{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]


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


class FireCols:
    cols = [[255, 242, 157], [255, 212, 124], [255, 169, 70], [254, 128, 3], [230, 106, 37], [179, 46, 19], [126, 19, 6], [70, 2, 14]]
    colsReversed = [[70, 2, 14], [126, 19, 6], [179, 46, 19], [230, 106, 37], [254, 128, 3], [255, 169, 70], [255, 212, 124], [255, 242, 157]]


# Defining some more variables to use in the game loop
oscillating_random_thing = 0
ShakeCounter = 0
toggle = True
click = False


class ViewConfig:
    """Debug view toggles (keys G and V)."""
    show_grid = False  # G: draw the collision spatial hash grid under the bodies
    filled = False  # V: filled polygons with a black border instead of points + springs


# ---------------- Test scene: ground platform, rigid square, soft pentagon dropped above the square
world = World()
# World box: four thin fixed walls whose inner faces sit 5px in from each screen edge
inset, wall = 5, 6
for cx, cy, w, h in ((screen_width / 2, inset - wall / 2, screen_width + 2 * wall, wall),  # top
                     (screen_width / 2, screen_height - inset + wall / 2, screen_width + 2 * wall, wall),  # bottom
                     (inset - wall / 2, screen_height / 2, wall, screen_height + 2 * wall),  # left
                     (screen_width - inset + wall / 2, screen_height / 2, wall, screen_height + 2 * wall)):  # right
    world.add(FixedBody(rect_points(cx, cy, w, h), point_col=Endesga.greyD, stick_col=Endesga.greyD, point_radius=1, fill_col=Endesga.greyD))
ground_top = screen_height - 40
ground_thickness = 6
world.add(FixedBody(rect_points(screen_center[0], ground_top + ground_thickness / 2, screen_width * 0.5, ground_thickness),
                    point_col=Endesga.greyD, stick_col=Endesga.greyL, point_radius=2, fill_col=Endesga.grey))
square_half = 20
square_y = ground_top - square_half - 40
world.add(RigidBody(rect_points(screen_center[0], square_y, 2 * square_half, 2 * square_half),
                    point_col=Endesga.debug_red, stick_col=Endesga.greyL, point_radius=3, fill_col=Endesga.debug_red))
world.add(SoftBody(polygon_points(screen_center[0] + 4, square_y - square_half - 50, 20, 6),
                   stiffness=0.1, point_col=Endesga.network_green, stick_col=Endesga.very_light_blue, point_radius=3,
                   fill_col=Endesga.network_green))

# Dodecagons (12 points, 2 sets of skip springs) with springs as strong as the hexagon, 25% weaker and 50% weaker
hexagon_stiffness = 0.1
for x_frac, y, strength, col in ((0.32, ground_top - 100, 1.0, Endesga.cream),
                                 (0.68, ground_top - 100, 0.75, Endesga.sebastian_lague_light_purple),
                                 (0.60, ground_top - 180, 0.5, Endesga.lighter_maroon_red)):
    world.add(SoftBody(polygon_points(screen_width * x_frac, y, 18, 12),
                       stiffness=hexagon_stiffness * strength, skip_sets=2, point_col=col, stick_col=Endesga.very_light_blue, point_radius=3, fill_col=col))

# Ropes: a small rigid box hung from the top wall by two ropes in a V, and a rope from the square to the cream dodecagon
square, cream_dodecagon = world.bodies[5], world.bodies[7]
hang_x, box_half = screen_width * 0.12, 8
anchor = (hang_x, inset)
hanging_box = RigidBody(rect_points(hang_x, inset + 50 + box_half, 2 * box_half, 2 * box_half),
                        point_col=Endesga.cream, stick_col=Endesga.greyL, point_radius=2)
world.add(hanging_box)
for corner in (0, 1):  # top-left, top-right
    world.add_rope(Rope(anchor, (hanging_box, corner), col=Endesga.light_brown))
world.add_rope(Rope((square, 0), (cream_dodecagon, 3), col=Endesga.light_brown))

# Hollow rigid box near the left wall with a soft dodecagon inside it; drag the box to slosh it around.
# The container is added first so it draws underneath its contents.
box_cx, box_cy = inset + 50, screen_height * 0.3
world.add(RigidBody(rect_points(box_cx, box_cy, 90, 70), point_col=Endesga.light_brown, stick_col=Endesga.greyL, point_radius=3))
world.add(SoftBody(polygon_points(box_cx, box_cy, 20, 12), stiffness=hexagon_stiffness, skip_sets=2,
                   point_col=Endesga.network_red, stick_col=Endesga.very_light_blue, point_radius=3))

# ---------------- Main Game Loop
last_time = time.time()
running = True
while running:

    # ---------------- Reset Variables and Clear screens
    mx, my = pygame.mouse.get_pos()
    mx, my = mx / scaleDownFactor, my / scaleDownFactor
    screen.fill(Endesga.my_blue)
    screen2.fill(Endesga.my_blue)
    screenT.fill((0, 0, 0, 0))
    screenUI.fill((0, 0, 0, 0))
    dt = time.time() - last_time
    dt *= fps
    last_time = time.time()
    timer -= 1 * dt
    shake = [0, 0]
    oscillating_random_thing += math.pi / fps * dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
            world.grab((mx, my))
        if event.type == pygame.MOUSEBUTTONUP:
            click = False
            world.release()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                toggle = not toggle
            if event.key == pygame.K_g:
                ViewConfig.show_grid = not ViewConfig.show_grid
            if event.key == pygame.K_v:
                ViewConfig.filled = not ViewConfig.filled
        if event.type == pygame.KEYUP:
            pass

    # ---------------- Updating Screen
    world.drag_to((mx, my))
    world.step(dt / fps)  # dt is in frames here, World wants seconds
    if ViewConfig.show_grid:
        world.draw_grid(screen2)
    world.draw(screen2, filled=ViewConfig.filled)

    if toggle:
        items = {round(clock.get_fps()): None,
                 "on" if ViewConfig.show_grid else "off": "G grid",
                 "filled" if ViewConfig.filled else "points": "V view",
                 }
        for i, label in enumerate(items.keys()):
            string = str(label)
            if items[label] is not None:
                string = f"{items[label]}: " + string
            drawText(screenUI, Endesga.debug_red, montserratRegularAdaptive, 5, screen_height - (30 + 25 * i) / (scaleDownFactor ** (1 / 1.8)), string, Endesga.black, int(3 / scaleDownFactor) + int(3 / scaleDownFactor) < 1, antiAliasing=False)
        pygame.mouse.set_visible(False)
        pygame.draw.circle(screenUI, Endesga.black, (mx + 1, my + 1), 2, 1)
        pygame.draw.circle(screenUI, Endesga.white, (mx, my), 2, 1)
    screen.blit(pygame.transform.scale(screen2, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (shake[0], shake[1]))
    screen.blit(pygame.transform.scale(screenT, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (shake[0], shake[1]))
    screen.blit(pygame.transform.scale(screenUI, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (0, 0))
    pygame.display.update()
    clock.tick(fps)
