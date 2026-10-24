import pygame
pygame.init()

styles = ["regular", "bold", "thin", "extralight"]
sizes = [i for i in range(100)]

fonts = {
    f"{style}{size}": pygame.font.Font(f"fonts/Montserrat-{style.capitalize()}.ttf", size) for style in styles for size in sizes}
