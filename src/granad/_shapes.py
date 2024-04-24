from functools import wraps

import jax.numpy as jnp

def _rotate_vertices(vertices, angle_degrees):
    angle_radians = jnp.radians(angle_degrees)
    rotation_matrix = jnp.array(
        [
            [jnp.cos(angle_radians), -jnp.sin(angle_radians)],
            [jnp.sin(angle_radians), jnp.cos(angle_radians)],
        ]
    )
    return jnp.dot(vertices, rotation_matrix)


def _edge_type(vertex_func):
    @wraps(vertex_func)
    def wrapper(*args, **kwargs):
        shift = kwargs.pop( "shift", [0,0])
        vertices = vertex_func(
            *args, **{key: val for key, val in kwargs.items() if key != "armchair"}
        )
        if "armchair" in kwargs and kwargs["armchair"] == True:
            vertices = _rotate_vertices(vertices, 90)
        return vertices + jnp.array( shift )

    return wrapper


@_edge_type
def Triangle(side_length):
    vertices = side_length * jnp.array(
        [
            (0, jnp.sqrt(3) / 3),
            (-0.5, -jnp.sqrt(3) / 6),
            (0.5, -jnp.sqrt(3) / 6),
            (0, jnp.sqrt(3) / 3),
        ]
    )
    return vertices


@_edge_type
def Rectangle(length_x, length_y):
    vertices = jnp.array(
        [
            (-1 * length_x, -0.5 * length_y),
            (1 * length_x, -0.5 * length_y),
            (1 * length_x, 0.5 * length_y),
            (-1 * length_x, 0.5 * length_y),
            (-1 * length_x, -0.5 * length_y),
        ]
    )
    return vertices


@_edge_type
def Hexagon(length):
    n = 6
    s = 1
    angle = 2 * jnp.pi / n
    vertices = length * jnp.array(
        [
            (s * jnp.cos(i * angle), s * jnp.sin(i * angle))
            for i in [x for x in range(n)] + [0]
        ]
    )
    return vertices


@_edge_type
def Rhomboid(base, height):
    angle = jnp.radians(30)
    vertices = jnp.array(
        [
            (0, 0),
            (base, 0),
            (base + height * jnp.sin(angle), height * jnp.cos(angle)),
            (height * jnp.sin(angle), height * jnp.cos(angle)),
            (0, 0),
        ]
    )
    return vertices
