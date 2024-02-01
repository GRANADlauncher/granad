import granad

import jax.numpy as jnp


def show_layout(layout_list):
    """Iterates over all layout arrays given in layout_lists and plots the geometry
    without indicating orbital sizes.

    :param layout_list: list of layout arrays
    """
    for layout in layout_list:
        # build stack
        sb = granad.StackBuilder()

        # add graphene
        graphene = granad.Lattice(
            shape=layout,
            lattice_type=granad.LatticeType.HONEYCOMB,
            lattice_edge=granad.LatticeEdge.ZIGZAG,
            lattice_constant=2.46,
        )
        sb.add("pz", graphene)

        sb.show3D()


show_layout(
    [
        granad.Rectangle(10, 8),  # only "zigzag" can be used with rectangle
        granad.Rhomboid(10, 8),
        granad.Hexagon(8),
        granad.Triangle(8),
    ]
)
