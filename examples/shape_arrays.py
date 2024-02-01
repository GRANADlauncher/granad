import granad

import jax.numpy as jnp
import matplotlib.pyplot as plt


sb = granad.StackBuilder()
graphene = granad.Lattice(
    shape=granad.Triangle(7.4),
    lattice_type=granad.LatticeType.HONEYCOMB,
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_constant=2.46,
)
sb.add("pz", graphene)
sb.show2D()
