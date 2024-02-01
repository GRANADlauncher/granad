import granad
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar


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
            s=40,#6000 * jnp.abs(stack.eigenvectors[idxs, show_state]),
            # alpha = 0.7,
            c = jnp.abs(stack.eigenvectors[idxs, show_state]),
            label=orb,
        )
    ax.set_aspect("equal", "box")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    for idx in idxs:
        ax.annotate(
            str(idx),
            (
                stack.positions[idx, 0],
                stack.positions[idx, 1],
        ),
        )
    number_str = '%.1f' % round(stack.energies[show_state],2)
    ax.set_title(f'E = {number_str} eV')

    return im

canvas = gridspec.GridSpec(3, 
                           1, 
                           wspace=0.0, # separated by a bit of white space
                           hspace=0.5, 
                           height_ratios=[0.03,1,0.5] 
                           )
eigenstates = gridspec.GridSpecFromSubplotSpec(1,
                                               2, 
                                               subplot_spec=canvas[1],
                                               )

# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Triangle(10),
    lattice_type=granad.LatticeType.HONEYCOMB,
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_constant=2.46,
)
sb.add("pz", graphene)

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

# plot first 4 IP excited states
stack_graphene = sb.get_stack()
for i, c in enumerate(eigenstates):
    sc = show_eigenstate2D(stack_graphene, plt.subplot(c), show_state = i)
colorbar_axis = plt.subplot(canvas[0])
cb = Colorbar(ax = colorbar_axis, mappable = sc, orientation = 'horizontal' )
cb.ax.set_xlabel(r'$|a_{ij}|$', rotation = 0)
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')

# set adatom couplings
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=0))
sb.set_hopping(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=2))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=0))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=0))

ax = plt.subplot( canvas[2] )
pos = sb.get_positions()
top_position = pos[24] + jnp.array([0.0, -1.0, 0.0])
hollow_position = (pos[24] + pos[31]) / 2
energies = []
names = ['top', 'hollow']
lines = ['-', '-', '-']
for i, p in enumerate([ top_position, hollow_position ]):

    sb.orbitals = list(filter(lambda x : x.orbital_id == 'pz', sb.orbitals))
    spot = granad.Spot(position=p)
    sb.add("A", spot)
    sb.add("B", spot, occupation=0)
   
    # set adatom and graphene coupling
    lattice_spot_hopping = granad.LatticeSpotCoupling(
        lattice_id="pz", spot_id="A", couplings=[2]
    )
    sb.set_hopping(lattice_spot_hopping)
    lattice_spot_coulomb = granad.LatticeSpotCoupling(
        lattice_id="pz",
        spot_id="A",
        couplings=[0],
    )
    sb.set_coulomb(lattice_spot_coulomb)

    lattice_spot_hopping = granad.LatticeSpotCoupling(
        lattice_id="pz", spot_id="B", couplings=[2]
    )
    sb.set_hopping(lattice_spot_hopping)
    lattice_spot_coulomb = granad.LatticeSpotCoupling(
        lattice_id="pz",
        spot_id="B",
        couplings=[0],
        coupling_function=lambda d: 0j,
    )
    sb.set_coulomb(lattice_spot_coulomb)

    stack = sb.get_stack()
    energies.append(stack.energies)

    omegas = jnp.linspace( min(stack.energies) - 1, max(stack.energies)  + 1, 2000 )
    dos_omega = lambda w : granad.dos( stack, w )
    dos = jax.vmap( dos_omega, (0,), 0)(omegas)
    ax.plot( omegas, dos / dos.max(), lines[i], label = names[i] )

ax.set_xlabel(r'$\omega$ (eV)')
ax.set_ylabel('Normalized DOS')
ax.legend()
plt.savefig('example_static.pdf')
plt.close()

# omegas = jnp.linspace( min(stack.energies) - 1, max(stack.energies)  + 1, 2000 )
# result = []
# for w in omegas:
#     result.append( granad.dos( stack, w, 0.1 ) )
