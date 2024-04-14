from dataclasses import dataclass
import jax.numpy as jnp

# TODO: this is kind of messy

def rotate_vertices(vertices, angle_degrees):
    angle_radians = jnp.radians(angle_degrees)
    rotation_matrix = jnp.array([
        [jnp.cos(angle_radians), -jnp.sin(angle_radians)],
        [jnp.sin(angle_radians), jnp.cos(angle_radians)]
    ])
    return jnp.dot(vertices, rotation_matrix)

# Equilateral triangle
triangle_vertices = jnp.array([
    (0, jnp.sqrt(3)/3),
    (-0.5, -jnp.sqrt(3)/6),
    (0.5, -jnp.sqrt(3)/6),
    (0, jnp.sqrt(3)/3),
])

# Rectangle
rectangle_vertices = jnp.array([
    (-1, -0.5),
    (1, -0.5),
    (1, 0.5),
    (-1, 0.5),
    (-1, -0.5),
])

# Hexagon
n = 6
s = 1
angle = 2 * jnp.pi / n
hexagon_vertices = jnp.array([
    (s * jnp.cos(i * angle), s * jnp.sin(i * angle)) for i in [ x for x in range(n) ] + [0]
])

# Rhomboid
base = 2
height = 1
angle = jnp.radians(30)
rhomboid_vertices = jnp.array([
    (0, 0),
    (base, 0),
    (base + height * jnp.sin(angle), height * jnp.cos(angle)),
    (height * jnp.sin(angle), height * jnp.cos(angle)),
    (0, 0),
])


class Shapes:
    triangle=triangle_vertices
    triangle_rotated=rotate_vertices(triangle_vertices, 90)
    rectangle=rectangle_vertices
    rectangle_rotated=rotate_vertices(rectangle_vertices, 90)
    hexagon=hexagon_vertices
    hexagon_rotated=rotate_vertices(hexagon_vertices, 90)
    rhomboid=rhomboid_vertices
    rhomboid_rotated=rotate_vertices(rhomboid_vertices, 90)
