import granad

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

# adapted from the main code for simplicity
def show_occupations(
    stack: granad.Stack,
    oc : jax.Array,
    ax
):
    """Shows a 2D scatter plot of how selected orbitals in a stack contribute to an eigenstate.
    In the plot, orbitals are annotated with a color. The color corresponds either to the contribution to the selected eigenstate or to the type of the orbital.
    Optionally, orbitals can be annotated with a number corresponding to the hilbert space index.

    :param stack: object representing system state
    :param oc:
    :param ax:
    """
    show_orbitals = stack.unique_ids 
    for orb in show_orbitals:
        idxs = jnp.nonzero(stack.ids == stack.unique_ids.index(orb))[0]
        im = ax.scatter(
            *zip(*stack.positions[idxs, :][:, [0,1]]),
            s = 40,
            c= oc,
            label=orb,
            vmin = 1,
            vmax = 2,
        )
    ax.set_aspect("equal", "box")
    return im

def setup():
    sb = granad.StackBuilder()
    graphene = granad.Lattice(
        shape=granad.Hexagon(7.4),
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


# building the stack
sb = setup()
stack = sb.get_stack( doping = 0, from_state = 0, to_state = 0 )
rho = np.array(granad.to_site_basis( stack, stack.rho_0 ))
diag = np.diag(rho)*stack.electrons
# idxs = np.arange( stack.electrons )[ (int(stack.electrons/2)) : (int(stack.electrons/2) + 10)]
idxs = [45,53,61,68,52,60, 31, 38, 30, 83, 75, 82]
diag[idxs] = 2.0
diag /= (stack.electrons + len(idxs))
np.fill_diagonal(rho, diag)
stack = stack.replace( rho_0 = stack.eigenvectors.conj().T @ rho @ stack.eigenvectors, electrons = stack.electrons + len(idxs)) 

# electric field
amplitudes = [0.0, 0, 0]
frequency = 1
peak = 2
fwhm = 0.5
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions, peak, fwhm
)

# propagate in time
time_axis = jnp.linspace(0, 10, 4000)

stack, occs = granad.evolution(
    stack, time_axis, field_func, granad.relaxation(1), postprocess=jnp.diag
)


# we want a division like this:
# 4 plots above, 4 plots below, to the side a colorbar
canvas = gridspec.GridSpec(2, # one row
                           1, # two columns
                           wspace=0.0, # separated by a bit of white space
                           hspace=0.0, 
                           height_ratios=[0.02, 1.0] 
                           )
occupations = gridspec.GridSpecFromSubplotSpec(1, 
                                               4, 
                                               subplot_spec=canvas[1], # occupations on the left
                                               wspace=0.05,
                                               hspace=0,
                                               )
occupations = [plt.subplot(i) for i in occupations]

colorbar_space = gridspec.GridSpecFromSubplotSpec(1,
                                            1, 
                                            subplot_spec=canvas[0],
                                            wspace=0,
                                            hspace=0)
colorbar_space = [plt.subplot(i) for i in colorbar_space]

# run simulation, save occs in 8-element array
skip = 50
occs_plot = occs[::skip] * stack.electrons

# first, show all occupations
for i in range(4):
    
    occupations[i].get_xaxis().set_visible(False)
    occupations[i].get_yaxis().set_visible(False)
    occupations[i].spines['top'].set_visible(False)
    occupations[i].spines['right'].set_visible(False)
    occupations[i].spines['bottom'].set_visible(False)
    occupations[i].spines['left'].set_visible(False)
    occupations[i].set_title(f"t = {round(time_axis[::skip][i],2)}")
    sc = show_occupations( stack, occs_plot[i], occupations[i]  )
    
# now, show colorbar
cb = Colorbar(ax = colorbar_space[0], mappable = sc, orientation = 'horizontal')
# plt.show()
plt.savefig('time_evolution_hexagon.pdf', bbox_inches='tight')

plt.plot( time_axis, occs * stack.electrons )
plt.show()
