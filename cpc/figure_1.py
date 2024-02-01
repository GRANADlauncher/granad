import granad

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

# adapted from the main code for simplicity
def show_eigenstate2D(
    stack: granad.Stack,
    show_state : int,
    ax
):
    """Shows a 2D scatter plot of how selected orbitals in a stack contribute to an eigenstate.
    In the plot, orbitals are annotated with a color. The color corresponds either to the contribution to the selected eigenstate or to the type of the orbital.
    Optionally, orbitals can be annotated with a number corresponding to the hilbert space index.

    :param stack: object representing system state
    """
    show_orbitals = stack.unique_ids
    for orb in show_orbitals:
        idxs = jnp.nonzero(stack.ids == stack.unique_ids.index(orb))[0]
        im = ax.scatter(
            *zip(*stack.positions[idxs, :][:, [0,1]]),
            s=40,
            c=stack.eigenvectors[idxs, show_state],
            label=orb,
        )
    ax.axis("equal")
    return im

# adapted from the main code for simplicity
def show_energies(stack: granad.Stack, ax):
    """Depicts the energy and occupation landscape of a stack (energies are plotted on the y-axis ordered by size)

    :param stack: stack object
    """
    # plt.colorbar(
    sc = ax.scatter(
        jnp.arange(stack.energies.size),
        stack.energies,
        c=stack.electrons * jnp.diag(stack.rho_0.real),
        )
    #     label="ground state occupation",
    # )
    ax.set_xlabel("eigenstate number")
    ax.set_ylabel("energy (eV)")
    return sc


def setup():
    sb = granad.StackBuilder()
    graphene = granad.Lattice(
        shape=granad.Hexagon(8),
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

    return sb

# we want a division like this:
# four plots side-by-side on the left, each showing the energy langscape with doping
# two plots on top of each other on the right, neighboured by a colorbar, showing the density matrix
# in site and energy space

canvas = gridspec.GridSpec(2, # two rows
                           1, # one column
                           wspace=0.0, # separated by a bit of white space
                           hspace=0.2,
                           height_ratios = [0.5, 1]
                           )

c_part = gridspec.GridSpecFromSubplotSpec(1, # one rows
                                          2, # two columns
                                          subplot_spec = canvas[0],
                                          wspace=0.05, # separated by a bit of white space
                                          hspace=0.0,
                                          width_ratios = [1, 0.05]
                                          )


d_part = gridspec.GridSpecFromSubplotSpec(1, # one rows
                                          2, # two columns
                                          subplot_spec = canvas[1],
                                          wspace=0.05, # separated by a bit of white space
                                          hspace=0.0,
                                          width_ratios = [1, 0.05]
                                          )

places = gridspec.GridSpecFromSubplotSpec(3, # two rows
                                          2, # three columns
                                          subplot_spec=d_part[0], 
                                          wspace=0.0, # separated by a bit of white space
                                          hspace=0.1,
                                          )
places = [plt.subplot(i) for i in places]


sb = setup()
stack = sb.get_stack()

ax = plt.subplot(c_part[0])
sc = show_energies( stack, ax)
colorbar_axis = plt.subplot(c_part[1])
cb = Colorbar(ax = colorbar_axis, mappable = sc, orientation = 'vertical' )

for i in range(6):
    places[i].get_xaxis().set_visible(False)
    places[i].get_yaxis().set_visible(False)
    places[i].spines['top'].set_visible(False)
    places[i].spines['right'].set_visible(False)
    places[i].spines['bottom'].set_visible(False)
    places[i].spines['left'].set_visible(False)


    sc = show_eigenstate2D( stack, i,  places[i] )
    
# now, colorbar
colorbar_axis = plt.subplot(d_part[1])
cb = Colorbar(ax = colorbar_axis, mappable = sc, orientation = 'vertical' )
# plt.show()
plt.savefig("fig_1_c_d.pdf")
