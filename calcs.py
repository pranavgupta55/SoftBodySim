import math
import colorsys
import random
import numpy as np


def createRadialGradientSurface(pygame, finalSize=(512, 512), circularSmoothnessSteps=3, starterSize=(3, 3), baseColor=(0, 0, 0, 200), centerColor=(180, 30, 255, 255)):
    """
    Creates a radial gradient surface.

    Parameters:
      pygame: base library
      finalSize: (width, height) for the final surface.
      circularSmoothnessSteps: Number of segments to rotate/blend.
      starterSize: Size of the small starter surface (e.g. (3, 3)).
      baseColor: RGBA color for the edges.
      centerColor: RGBA color for the center.

    Returns:
      A pygame.Surface containing the radial gradient.
    """
    tempSurface = pygame.Surface(finalSize, pygame.SRCALPHA)
    tempSurface.set_alpha(255)

    # Adjust the colors by dividing each channel by circularSmoothnessSteps.
    color1 = pygame.Color(*baseColor)
    color2 = pygame.Color(*centerColor)
    for attr in ('r', 'g', 'b', 'a'):
        setattr(color1, attr, getattr(color1, attr) // circularSmoothnessSteps)
        setattr(color2, attr, getattr(color2, attr) // circularSmoothnessSteps)

    # Create the starter surface.
    starter = pygame.Surface(starterSize, pygame.SRCALPHA)
    starter.fill(color1)
    centerRect = pygame.Rect(starterSize[0] // 2, starterSize[1] // 2, 1, 1)
    starter.fill(color2, centerRect)

    # Scale up the starter.
    gradSurface = pygame.transform.smoothscale(starter, finalSize)

    # Rotate and blend copies.
    for i in range(circularSmoothnessSteps):
        angle = (360.0 / circularSmoothnessSteps) * i
        rotated = pygame.transform.rotate(gradSurface, angle)
        posRect = pygame.Rect((0, 0), finalSize)
        areaRect = pygame.Rect(0, 0, finalSize[0], finalSize[1])
        areaRect.center = (rotated.get_width() // 2, rotated.get_height() // 2)
        tempSurface.blit(rotated, posRect, area=areaRect, special_flags=pygame.BLEND_RGBA_ADD)

    pygame.draw.circle(tempSurface, baseColor, (tempSurface.get_width() / 2, tempSurface.get_width() / 2), tempSurface.get_width(), int(tempSurface.get_width() / 2))

    return tempSurface


def drawRoundedLine(pygame, s, p1, p2, c, w):
    p1v = pygame.math.Vector2(p1)
    p2v = pygame.math.Vector2(p2)
    lv = (p2v - p1v).normalize()
    lnv = pygame.math.Vector2(-lv.y, lv.x) * (w / 2)
    pts = [p1v + lnv, p2v + lnv, p2v - lnv, p1v - lnv]
    pygame.draw.polygon(s, c, pts)
    pygame.draw.circle(s, c, (int(p1v.x), int(p1v.y)), round(w / 2))
    pygame.draw.circle(s, c, (int(p2v.x), int(p2v.y)), round(w / 2))


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def ang(start_pos, end_pos):
    return math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])


def normalize_angle(angle):
    return angle % (2 * math.pi)


def collide_circle(point1, point2, d):
    return math.sqrt(math.pow(math.fabs(point1[0] - point2[0]), 2) + math.pow(math.fabs(point1[1] - point2[1]), 2)) < d


def brightness(color, shift):
    output = [0, 0, 0]
    for i, c in enumerate(color):
        output[i] = int(c * shift) + 1
        if output[i] > 255:
            output[i] = 255
    return output


def contrast(color, shift):
    output = [0, 0, 0]
    avg = sum(color) / len(color)
    for i, c in enumerate(color):
        if c > avg:
            output[i] = int(c * (1 + shift))
        if c < avg:
            output[i] = int(c * (1 - shift))
        if c == avg:
            output[i] = c
    for i, o in enumerate(output):
        if o >= 255:
            output[i] = 255
    return output


def shift_hue(color, hue_shift):
    # Convert RGB color to HSV color
    r, g, b = color
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

    # Shift the hue
    h = (h + hue_shift) % 1

    # Convert HSV color back to RGB color
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    r, g, b = int(r * 255), int(g * 255), int(b * 255)

    return r, g, b


def randomCol(weighting=None):
    if weighting == 'red' or weighting == 'r':
        return [random.randint(127, 255), random.randint(0, 127), random.randint(0, 127)]
    if weighting == 'green' or weighting == 'g':
        return [random.randint(0, 127), random.randint(127, 255), random.randint(0, 127)]
    if weighting == 'blue' or weighting == 'b':
        return [random.randint(0, 127), random.randint(0, 127), random.randint(127, 255)]
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


def linearGradient(colors, normalizedZero2One):
    normalizedZero2One = clip(0, 1 - 1e-3, normalizedZero2One)
    percent = (len(colors) - 1) * normalizedZero2One % 1
    index = min(int((len(colors) - 1) * normalizedZero2One), len(colors) - 1)
    return [(int(colors[index][i] + percent * (colors[index + 1][i] - colors[index][i]))) for i in range(3)]


def setOpacity(color, newOpacity):
    return color[0], color[1], color[2], newOpacity


def normalize(value, minValue, maxValue, doesCap=False):
    output = (value - minValue) / (maxValue - minValue)
    if doesCap:
        if output > 1:
            output = 1
        if output < 0:
            output = 0
    return output


def clip(minVal, maxVal, val):
    return min(maxVal, max(minVal, val))


def point_to_line(point, two_points_of_line):
    point_x, point_y = point
    x1, y1 = two_points_of_line[0]
    x2, y2 = two_points_of_line[1]
    dx = x2 - x1
    dy = y2 - y1
    parameterization = ((point_x - x1) * dx + (point_y - y1) * dy) / (dx ** 2 + dy ** 2)
    if parameterization < 0:
        closest_x, closest_y = x1, y1
    elif parameterization > 1:
        closest_x, closest_y = x2, y2
    else:
        closest_x = x1 + parameterization * dx
        closest_y = y1 + parameterization * dy

    d = math.sqrt((point_x - closest_x) ** 2 + (point_y - closest_y) ** 2)
    return d, [closest_x, closest_y]


def circumcircle(vertices):
    # Extract the vertices
    (x1, y1), (x2, y2), (x3, y3) = vertices

    # Calculate the coordinates of the circumcircle center
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    center_x = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / D
    center_y = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / D

    # Calculate the radius of the circumcircle
    radius = math.sqrt((x1 - center_x) ** 2 + (y1 - center_y) ** 2)

    return (center_x, center_y), radius


def random_sign():
    return random.randint(0, 1) * 2 - 1


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def reLu(x):
    return (x > 0) * x


def reLu_prime(x):
    return x > 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def draw_arrow(screen, start, end, color, pygameModel, thickness=3, arrowhead_length=10, arrowhead_angle=25):
    # Draw the main line (shaft of the arrow)
    pygameModel.draw.line(screen, color, start, end, thickness)

    # Calculate the angle of the line
    angle = math.atan2(end[1] - start[1], end[0] - start[0])

    # Calculate the two points of the arrowhead
    angle1 = angle + math.radians(arrowhead_angle)
    angle2 = angle - math.radians(arrowhead_angle)

    arrowhead_point1 = (end[0] - arrowhead_length * math.cos(angle1), end[1] - arrowhead_length * math.sin(angle1))

    arrowhead_point2 = (end[0] - arrowhead_length * math.cos(angle2), end[1] - arrowhead_length * math.sin(angle2))

    # Draw the two lines for the arrowhead
    pygameModel.draw.line(screen, color, end, arrowhead_point1, thickness)
    pygameModel.draw.line(screen, color, end, arrowhead_point2, thickness)


def search(direc, node, max_sizes, all_blocks):
    new_nodes = []
    searches = [[True, True], [True, True]]
    searches[direc // 2][1 - (direc % 2)] = False
    if searches[0][0]:
        if (node[0] - 1) >= 0:
            if all_blocks[node[0] - 1][node[1]] == 0:
                new_nodes.append([node[0] - 1, node[1]])
    if searches[0][1]:
        if (node[0] + 1) < max_sizes[0]:
            if all_blocks[node[0] + 1][node[1]] == 0:
                new_nodes.append([node[0] + 1, node[1]])
    if searches[1][0]:
        if (node[1] - 1) >= 0:
            if all_blocks[node[0]][node[1] - 1] == 0:
                new_nodes.append([node[0], node[1] - 1])
    if searches[1][1]:
        if (node[1] + 1) < max_sizes[1]:
            if all_blocks[node[0]][node[1] + 1] == 0:
                new_nodes.append([node[0], node[1] + 1])
    return new_nodes


def floodFillStep(queue, max_sizes, all_blocks):
    for node in reversed(queue[0]):
        for direc in range(4):
            s = search(direc, node, max_sizes, all_blocks)
            for new_node in s:
                if (new_node not in queue[0]) and (new_node not in queue[1]):
                    queue[0].append(new_node)
        queue[1].append(node)
        queue[0].remove(node)
    return queue


def rectRotation(center, w, h, a=0):
    # tr tl bl br
    cords = [[center[0] + ((w / 2) * math.cos(a)) - ((h / 2) * math.sin(a)), center[1] + ((w / 2) * math.sin(a)) + ((h / 2) * math.cos(a))], [center[0] - ((w / 2) * math.cos(a)) - ((h / 2) * math.sin(a)), center[1] - ((w / 2) * math.sin(a)) + ((h / 2) * math.cos(a))], [center[0] - ((w / 2) * math.cos(a)) + ((h / 2) * math.sin(a)), center[1] - ((w / 2) * math.sin(a)) - ((h / 2) * math.cos(a))],
             [center[0] + ((w / 2) * math.cos(a)) + ((h / 2) * math.sin(a)), center[1] + ((w / 2) * math.sin(a)) - ((h / 2) * math.cos(a))]]
    return cords


def ellipsePointCollision(pos, ellipseCenter, ellipseHeightRad, ellipseWidthRad):
    return (((ellipseCenter[0] - pos[0]) ** 2) / (ellipseHeightRad ** 2) + ((ellipseCenter[1] - pos[1]) ** 2) / (ellipseWidthRad ** 2)) <= 1
