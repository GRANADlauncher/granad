from dataclasses import dataclass
import numpy as np

def rotate_vertices(vertices, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    return np.dot(vertices, rotation_matrix)

@dataclass
class Shapes:
    triangle: np.ndarray
    triangle_rotated: np.ndarray
    rectangle: np.ndarray
    rectangle_rotated: np.ndarray
    hexagon: np.ndarray
    hexagon_rotated: np.ndarray
    rhomboid: np.ndarray
    rhomboid_rotated: np.ndarray

# Equilateral triangle
triangle_vertices = np.array([
    (0, np.sqrt(3)/3),
    (-0.5, -np.sqrt(3)/6),
    (0.5, -np.sqrt(3)/6),
    (0, np.sqrt(3)/3),
])

# Rectangle
rectangle_vertices = np.array([
    (-1, -0.5),
    (1, -0.5),
    (1, 0.5),
    (-1, 0.5),
    (-1, -0.5),
])

# Hexagon
n = 6
s = 1
angle = 2 * np.pi / n
hexagon_vertices = np.array([
    (s * np.cos(i * angle), s * np.sin(i * angle)) for i in [ x for x in range(n) ] + [0]
])

# Rhomboid
base = 2
height = 1
angle = np.radians(30)
rhomboid_vertices = np.array([
    (0, 0),
    (base, 0),
    (base + height * np.sin(angle), height * np.cos(angle)),
    (height * np.sin(angle), height * np.cos(angle)),
    (0, 0),
])

# Creating the dataclass instance
shapes = Shapes(
    triangle=triangle_vertices,
    triangle_rotated=rotate_vertices(triangle_vertices, 90),
    rectangle=rectangle_vertices,
    rectangle_rotated=rotate_vertices(rectangle_vertices, 90),
    hexagon=hexagon_vertices,
    hexagon_rotated=rotate_vertices(hexagon_vertices, 90),
    rhomboid=rhomboid_vertices,
    rhomboid_rotated=rotate_vertices(rhomboid_vertices, 90)
)
