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
    """
    Decorator that extends a vertex-generating function by adding the ability to shift and optionally rotate the vertices.

    Parameters:
        vertex_func (function): A function that generates vertices of a shape.

    Returns:
        function: A wrapped function that now accepts additional keyword arguments:
                  'shift' (list of float): Adjusts the x and y coordinates by specified amounts.
                  'armchair' (bool): If True, rotates the shape to the armchair orientation, typically by 90 degrees.

    Usage:
        This decorator is used to add functionality to basic geometric shape functions, allowing for
        easy manipulation of the shape's position and orientation in a plane.
    """
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
    """
    Generates the vertices of an equilateral triangle given the side length.

    The triangle is oriented such that one vertex points upwards and the base is horizontal.
    This function is designed to be used with the @_edge_type decorator, which adds functionality
    to shift the triangle's position or rotate it based on additional 'shift' and 'armchair'
    parameters passed to the function.

    Parameters:
        side_length (float): The length of each side of the triangle, specified in angstroms.

    Returns:
        jax.numpy.ndarray: An array of shape (4, 2), representing the vertices of the triangle,
                           including the starting vertex repeated at the end to facilitate
                           drawing closed shapes.

    Example:
        # Create a triangle with side length of 1.0 angstrom, no shift or rotation
        triangle = Triangle(1.0)

        # Create a triangle with side length of 1.0 angstrom, shifted by [1, 1] units
        triangle_shifted = Triangle(1.0, shift=[1, 1])

        # Create a triangle with side length of 1.0 angstrom, rotated by 90 degrees (armchair orientation)
        triangle_rotated = Triangle(1.0, armchair=True)
    """
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
    """
    Generates the vertices of a rectangle given the lengths along the x and y dimensions.

    The rectangle is centered at the origin, and the function is designed to be used with
    the @_edge_type decorator, allowing for positional shifts and rotations (if specified).

    Parameters:
        length_x (float): The length of the rectangle along the x-axis, specified in angstroms.
        length_y (float): The length of the rectangle along the y-axis, specified in angstroms.

    Returns:
        jax.numpy.ndarray: An array of shape (5, 2), representing the vertices of the rectangle,
                           starting and ending at the same vertex to facilitate drawing closed shapes.

    Example:
        # Rectangle with length 2.0 and height 1.0 angstroms
        rectangle = Rectangle(2.0, 1.0)
    """
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
    """
    Generates the vertices of a regular hexagon given the side length.

    The hexagon is oriented such that one vertex points upwards and the function is designed
    to be used with the @_edge_type decorator for positional adjustments and rotations.

    Parameters:
        length (float): The length of each side of the hexagon, specified in angstroms.

    Returns:
        jax.numpy.ndarray: An array of shape (7, 2), representing the vertices of the hexagon,
                           including the starting vertex repeated at the end for drawing closed shapes.

    Example:
        # Hexagon with side length of 1.0 angstrom
        hexagon = Hexagon(1.0)
    """
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
    """
    Generates the vertices of a rhomboid given the base length and height.

    The rhomboid is initially oriented with the base along the x-axis, and one angle being 30 degrees,
    designed to be adjusted for position and orientation using the @_edge_type decorator.

    Parameters:
        base (float): The length of the base of the rhomboid, specified in angstroms.
        height (float): The vertical height of the rhomboid, specified in angstroms.

    Returns:
        jax.numpy.ndarray: An array of shape (5, 2), representing the vertices of the rhomboid,
                           starting and ending at the same vertex to complete the shape.

    Example:
        # Rhomboid with base 2.0 angstroms and height 1.0 angstrom
        rhomboid = Rhomboid(2.0, 1.0)
    """
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