import granad
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

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

# get the stack object
stack_graphene = sb.get_stack( from_state = 0, to_state = 0 )
amplitudes = [1.0, 0.5, 0]
frequency = 1
peak = 2
fwhm = 0.5
field_func = granad.electric_field_pulse(
    amplitudes,
    frequency,
    stack_graphene.positions,
    peak,
    fwhm
)

dissipation = granad.relaxation(0.1)

time_axis = jnp.linspace(0, 4, 10**5)

stack_graphene_new, occs = granad.evolution( stack_graphene,
                                             time_axis,
                                             field_func,
                                             dissipation = dissipation,
                                             postprocess = jnp.diag
                                            )

skip = 10
dip_moment = granad.induced_dipole_moment( stack_graphene, occs )
labels = ['x', 'y']
norm = dip_moment.max()
for i, d in enumerate(dip_moment.T[:2] / norm):
    label = labels[i]
    plt.plot( time_axis[::skip], d[::skip], label = rf'$p_{label}$' )
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$P_{ind}$ (normalized)')
plt.savefig('example_dynamic.pdf')
plt.close()
# granad.show_energy_occupations( stack_graphene, occs[::skip], time_axis[::skip], thresh = 1e-2)

