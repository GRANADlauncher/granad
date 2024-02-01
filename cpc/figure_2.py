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



def setup( edge_type ):
    sb = granad.StackBuilder()
    graphene = granad.Lattice(
        shape=granad.Triangle(9),
        lattice_type=granad.LatticeType.HONEYCOMB,
        lattice_edge=edge_type,
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
canvas = gridspec.GridSpec(1, # one row
                           3, # two columns
                           wspace=0.05, # separated by a bit of white space
                           hspace=0.0, 
                           width_ratios=[1,0.8,0.1] # equal width
                           )
energy_landscapes = gridspec.GridSpecFromSubplotSpec(2,
                                                     1, 
                                                     subplot_spec=canvas[0], # energy landscapes go to the left
                                                     wspace=0,
                                                     hspace=0.3,
                                                     height_ratios=[1,1]
                                                     )
energy_landscapes = [plt.subplot(i) for i in energy_landscapes]

density_picture = gridspec.GridSpecFromSubplotSpec(2,
                                                   1, # two of them atop each other
                                                   subplot_spec=canvas[1], # densities to the right
                                                   wspace=0,
                                                   hspace=0.1)
density_picture = [plt.subplot(i) for i in density_picture]


stacks = []
# first, make doping plot
for i,edge_type in enumerate( [granad.LatticeEdge.ARMCHAIR, granad.LatticeEdge.ZIGZAG] ):
    # if i:
    #     energy_landscapes[i].get_xaxis().set_visible(False)
    #     energy_landscapes[i].get_yaxis().set_visible(False)

    sb = setup( edge_type )
    stack = sb.get_stack()
    show_energies( stack, energy_landscapes[i] )
    stacks.append(stack)
    
# now, plot density matrices
ax1, ax2 = density_picture
ax1.matshow( stacks[0].electrons * granad.to_site_basis(stacks[0], stacks[0].rho_0).real, vmin = 0, vmax = 2 )
ax1.axis('off')
sc = ax2.matshow( stacks[1].electrons * granad.to_site_basis( stacks[1], stacks[1].rho_0.real).real, vmin = 0, vmax = 2 )
ax2.axis('off')
colorbar_axis = plt.subplot(canvas[2])
cb = Colorbar(ax = colorbar_axis, mappable = sc, orientation = 'vertical' )

# plt.show()
plt.savefig('figure_2.pdf')
