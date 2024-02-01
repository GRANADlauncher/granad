import granad

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import lax

# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Triangle(7.4),
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_type=granad.LatticeType.HONEYCOMB,
    lattice_constant=2.46,
)
sb.add("pz", graphene)

hopping_graphene = granad.LatticeCoupling(
    orbital_id1="pz", orbital_id2="pz", lattice=graphene, couplings=[0, -2.66]
)
sb.set_hopping(hopping_graphene)

coulomb_graphene = granad.LatticeCoupling(
    orbital_id1="pz",
    orbital_id2="pz",
    lattice=graphene,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
sb.set_coulomb(coulomb_graphene)

# create the stack object
stack = sb.get_stack( from_state = 0, to_state = 2)

# compute induced field corresponding to the state of the stack at the origin
positions = jnp.zeros((1,3))
field = granad.induced_field( stack, positions )


