import granad

import jax
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

# add adatom in top position over the 0-th atom in benzene ring
pos = sb.get_positions()
top_position = pos[0] + jnp.array([0.0, 0.0, 1.0])
spot = granad.Spot(position=top_position)
sb.add("A", spot)
sb.add("B", spot, occupation=0)

# set graphene couplings
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


# set adatom couplings
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_hopping(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=2))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=2))

# set adatom and graphene coupling
lattice_spot_hopping = granad.LatticeSpotCoupling(
    lattice_id="pz", spot_id="A", couplings=[1, 0.5]
)
sb.set_hopping(lattice_spot_hopping)
lattice_spot_coulomb = granad.LatticeSpotCoupling(
    lattice_id="pz",
    spot_id="A",
    couplings=[2],
    coupling_function=lambda d: 1 / d + 0j,
)
sb.set_coulomb(lattice_spot_coulomb)

lattice_spot_hopping = granad.LatticeSpotCoupling(
    lattice_id="pz", spot_id="B", couplings=[1, 0.5]
)
sb.set_hopping(lattice_spot_hopping)
lattice_spot_coulomb = granad.LatticeSpotCoupling(
    lattice_id="pz",
    spot_id="B",
    couplings=[2],
    coupling_function=lambda d: 1 / d + 0j,
)
sb.set_coulomb(lattice_spot_coulomb)

stack = sb.get_stack()

granad.show_energies(stack)
granad.show_eigenstate3D(stack)
