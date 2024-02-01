import jax
import jax.numpy as jnp

import granad

# build stack
sb = granad.StackBuilder()

# add graphene
chain = granad.Lattice(
    shape=granad.Chain(10),
    lattice_type=granad.LatticeType.CHAIN,
    lattice_edge=granad.LatticeEdge.NONE,
    lattice_constant=1,
)
sb.add("A", chain)

hopping = granad.LatticeCoupling(
    orbital_id1="A", orbital_id2="A", lattice=chain, couplings=[0, -2.66]
)
sb.set_hopping(hopping)

coulomb = granad.LatticeCoupling(
    orbital_id1="A",
    orbital_id2="A",
    lattice=chain,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
sb.set_coulomb(coulomb)

# create the stack object
stack = sb.get_stack()

# visualize the first "excited state"
granad.show_eigenstate2D(stack, show_state=1)
