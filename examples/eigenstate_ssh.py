import jax
import jax.numpy as jnp
from jax import lax

import granad

# build stack
sb = granad.StackBuilder()

# add graphene
chain = granad.Lattice(
    shape=granad.Chain(5),
    lattice_type=granad.LatticeType.SSH,
    lattice_edge=granad.LatticeEdge.NONE,
    lattice_constant=1,
)
sb.add("A", chain)

hopping = granad.LatticeCoupling(
    orbital_id1="A", orbital_id2="A", lattice=chain, couplings=[0, -2.66, -1.0]
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
stack_dimerized = sb.get_stack()

# show energies
granad.show_energies( stack_dimerized )

# we now modify the positions to get edge states
orbs_tmp = []
for i, orb in enumerate(sb.orbitals):
    if i % 2:
        position = orb.position + jnp.array([0.4, 0.0, 0.0])
        orbs_tmp.append(orb.replace( position = position) )
    else:
        orbs_tmp.append( orb )
sb.orbitals = orbs_tmp            

stack_edge = sb.get_stack()

# show energies
granad.show_energies( stack_edge )
