import granad

import jax
import jax.numpy as jnp


def energy_gs(position):

    # build stack
    sb = granad.StackBuilder()

    # add graphene
    graphene = granad.Lattice(
        shape=granad.Triangle(7.4),
        lattice_type=granad.LatticeType.HONEYCOMB,
        lattice_edge=granad.LatticeEdge.ARMCHAIR,
        lattice_constant=2.46,
    )
    sb.add("pz", graphene)

    # add adatom in top position over the 0-th atom in benzene ring
    spot = granad.Spot(position=position)
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
        couplings=[3.0, 2.0, 1.0],
        coupling_function=jax.jit(lambda d: 1 / d + 0j),
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
    distance_coupling_ag = granad.DistanceCoupling(
        orbital_id1="pz",
        orbital_id2="A",
        coupling_function=jax.jit(lambda d: 1 / d + 0j),
    )
    distance_coupling_bg = granad.DistanceCoupling(
        orbital_id1="pz",
        orbital_id2="B",
        coupling_function=jax.jit(lambda d: 1 / d + 0j),
    )

    sb.set_hopping(distance_coupling_ag)
    sb.set_hopping(distance_coupling_bg)
    sb.set_coulomb(distance_coupling_ag)
    sb.set_coulomb(distance_coupling_bg)

    stack = sb.get_stack()

    return stack.energies[:4].sum()


pos = [2.5, 1.0, 0.1]
gamma = 0.01
dx, dy = 1, 1

# obtain gradient wrt to the position
dx, dy, _ = jax.jacfwd(energy_gs)(pos)
