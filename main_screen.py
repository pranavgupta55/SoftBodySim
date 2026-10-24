import pygame
import math
import time
import random
from text import drawText
from fontDict import fonts
import numpy as np
from scene import Endesga, FireCols, ViewConfig, INSET, build_scene
from editor import Editor
from profiler import Profiler
from perf_hud import PerfHUD
import ui_style

pygame.init()

# ---------------- Setting up the screen, assigning some global variables, and loading text fonts
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
clock = pygame.time.Clock()
fps = 60
scaleDownFactor = 2
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


# Defining some more variables to use in the game loop
oscillating_random_thing = 0
ShakeCounter = 0
toggle = True
click = False


# The world is measured in its own units; the render surfaces above are its size times the zoom
world_width, world_height = screen_width / ViewConfig.zoom, screen_height / ViewConfig.zoom
world_center = [world_width / 2, world_height / 2]


# ---------------- Test scene (scene.py, shared with main_opengl.py)
world = build_scene(world_width, world_height)
inset = INSET

# The editor's UI lives on screenUI; it reads the G / V states from ViewConfig for the info card
editor = Editor(world, Endesga, (screen_width, screen_height), (world_width, world_height), scaleDownFactor, inset,
                ViewConfig.zoom, view=ViewConfig)
profiler = Profiler()  # frame timings for the compute card (sections below; World.step times its own phases)
world.profiler = profiler
perf_hud = PerfHUD(profiler)

# ---------------- Main Game Loop
last_time = time.time()
running = True
while running:
    profiler.frame()
    profiler.begin("draw")

    # ---------------- Reset Variables and Clear screens
    mx, my = pygame.mouse.get_pos()
    mx, my = mx / scaleDownFactor, my / scaleDownFactor  # render-surface pixels (cursor, UI)
    wx, wy = mx / ViewConfig.zoom, my / ViewConfig.zoom  # world units (physics)
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

    profiler.switch("events")
    editor.show_card = toggle  # the info card (and its buttons) only exists while the HUD is shown
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if editor.handle_event(event):  # info card buttons, editor panels and placement come first
            continue
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
            world.grab((wx, wy))
        if event.type == pygame.MOUSEBUTTONUP:
            click = False
            world.release()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                toggle = not toggle
                editor.show_card = toggle
            if event.key == pygame.K_g:
                ViewConfig.show_grid = not ViewConfig.show_grid
            if event.key == pygame.K_v:
                ViewConfig.filled = not ViewConfig.filled
        if event.type == pygame.KEYUP:
            pass

    # ---------------- Updating Screen
    profiler.switch("physics")
    world.drag_to((wx, wy))
    if not editor.paused:
        world.step(dt / fps)  # dt is in frames here, World wants seconds
    profiler.switch("draw")
    if ViewConfig.show_grid:
        world.draw_grid(screen2, zoom=ViewConfig.zoom)
    world.draw(screen2, filled=ViewConfig.filled, zoom=ViewConfig.zoom)
    editor.draw_world(screen2, screenT, (mx, my))
    editor.draw_ui(screenUI, (mx, my))  # panels, buttons and hints, under the HUD cards

    profiler.switch("ui")
    if toggle:
        info_card = editor.draw_info_card(screenUI, (mx, my))  # first card of the top-right column
        perf_hud.draw(screenUI, info_card.bottom + ui_style.CARD_GAP)  # compute card underneath
    editor.draw_message(screenUI)  # over the cards
    if toggle:
        pygame.mouse.set_visible(False)
        pygame.draw.circle(screenUI, Endesga.black, (mx + 1, my + 1), 2, 1)
        pygame.draw.circle(screenUI, Endesga.white, (mx, my), 2, 1)
    profiler.switch("present")
    screen.blit(pygame.transform.scale(screen2, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (shake[0], shake[1]))
    screen.blit(pygame.transform.scale(screenT, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (shake[0], shake[1]))
    screen.blit(pygame.transform.scale(screenUI, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (0, 0))
    pygame.display.update()
    profiler.switch("idle", device="idle")  # the frame cap's sleep
    clock.tick(fps)
