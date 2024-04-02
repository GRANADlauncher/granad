import granad
import jax
from jax import lax
import jax.numpy as jnp
from granad import _density_matrix

if __name__ == '__main__':
    # build stack
    sb = granad.StackBuilder()

    # add graphene
    graphene = granad.Lattice(
    shape=granad.Triangle(4.1),
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_type=granad.LatticeType.HONEYCOMB,
    lattice_constant=2.46,
    )
    sb.add("pz_up", graphene)
    
    # make it half full
    sb.add("pz_down", graphene, occupation = 0)
    
    uu = granad.LatticeCoupling(
        orbital_id1="pz_up", orbital_id2="pz_up", lattice=graphene, couplings=[0, -2.66]
    )
    dd = granad.LatticeCoupling(
        orbital_id1="pz_down", orbital_id2="pz_down", lattice=graphene, couplings=[0, -2.66]
    )
    ud = granad.LatticeCoupling(
        orbital_id1="pz_up", orbital_id2="pz_down", lattice=graphene, couplings=[0, -2.66]
    )

    sb.set_hopping(uu)
    sb.set_hopping(dd)
    sb.set_hopping(ud)

    cuu = granad.LatticeCoupling(
        orbital_id1="pz_up",
        orbital_id2="pz_up",
        lattice=graphene,
        couplings=[16.522, 8.64, 5.333],
        coupling_function=lambda d: 14.399 / d + 0j,
    )
    cdd = granad.LatticeCoupling(
        orbital_id1="pz_down",
        orbital_id2="pz_down",
        lattice=graphene,
        couplings=[16.522, 8.64, 5.333],
        coupling_function=lambda d: 14.399 / d + 0j,
    )
    cud = granad.LatticeCoupling(
        orbital_id1="pz_up",
        orbital_id2="pz_down",
        lattice=graphene,
        couplings=[16.522, 8.64, 5.333],
        coupling_function=lambda d: 14.399 / d + 0j,
    )
    sb.set_coulomb(cuu)
    sb.set_coulomb(cdd)
    sb.set_coulomb(cud)

    # create the stack object
    stack = sb.get_stack( from_state = 0, to_state = 0,  doping = 0, spin_degenerate = False )
    print(stack.rho_0.diagonal(), stack.homo)
