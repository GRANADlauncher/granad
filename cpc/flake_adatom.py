import granad

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

# adapted from the main code for simplicity
def show_energies(stack: granad.Stack, ax):
    """Depicts the energy and occupation landscape of a stack (energies are plotted on the y-axis ordered by size)

    :param stack: stack object
    """
    # plt.colorbar(
    ax.scatter(
        jnp.arange(stack.energies.size),
        stack.energies,
        c=stack.electrons * jnp.diag(stack.rho_0.real),
        )
    #     label="ground state occupation",
    # )
    ax.set_xlabel("eigenstate number")
    ax.set_ylabel("energy (eV)")

def show_eigenstate2D(
        stack,
        ax,
        show_state: int = 0,
        show_orbitals: list[str] = None,
        indicate_size: bool = True,
):
    show_orbitals = stack.unique_ids if show_orbitals is None else show_orbitals
    for orb in show_orbitals:
        idxs = jnp.nonzero(stack.ids == stack.unique_ids.index(orb))[0]
        im = ax.scatter(
            *zip(*stack.positions[idxs, :2]),
            s=6000 * jnp.abs(stack.eigenvectors[idxs, show_state]),
            alpha = 0.7,
            c = None, #stack.eigenvectors[show_state, idxs].real,
            label=orb,
        )
    ax.set_aspect("equal", "box")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.legend()
    return im

def setup( ):
    sb = granad.StackBuilder()
    graphene = granad.Lattice(
        shape=granad.Triangle(9),
        lattice_type=granad.LatticeType.HONEYCOMB,
        lattice_edge=granad.LatticeEdge.ARMCHAIR,
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
        coupling_function=lambda d: 14.3989878 / d + 0j,
    )
    sb.set_coulomb(coulomb_graphene)

    pos = sb.get_positions()
    top_position = pos[0] + jnp.array([-1.0, 0.0, 0.0])
    spot = granad.Spot(position=top_position)
    sb.add("A", spot)
    sb.add("B", spot)

    # onsite hopping
    sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
    sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
    sb.set_hopping(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=2))

    # onsite coulomb
    sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=1))
    sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
    sb.set_coulomb(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=1))
    
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

    return sb

# we want a division like this:
# four plots side-by-side on the left, each showing the energy langscape with doping
# two plots on top of each other on the right, neighboured by a colorbar, showing the density matrix
# in site and energy space
canvas = gridspec.GridSpec(1, # one row
                           1, # one column
                           wspace=0.05, # separated by a bit of white space
                           hspace=0.0, 
                           # width_ratios=[0.5,1] 
                           )

sb = setup()
stack = sb.get_stack()
# show_energies( stack, plt.subplot( canvas[0] ) )

show_eigenstate2D( stack, plt.subplot(canvas[0]) )
    
# plt.show()
plt.savefig('flake_adatom.pdf')
