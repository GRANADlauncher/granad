import granad

import jax.numpy as jnp

# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Triangle(3),
    lattice_type=granad.LatticeType.HONEYCOMB,
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_constant=2.46,
)
sb.add("pz", graphene)

sb.show3D()
