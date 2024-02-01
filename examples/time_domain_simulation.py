import granad

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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

# create the stack object
stack = sb.get_stack()

amplitudes = [1.5, 0, 0]
frequency = 2.75
peak = 0.2
fwhm = 0.5

# choose an x-polarized electric field propagating in z-direction
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions, peak, fwhm
)

# propagate in time
time_axis = jnp.linspace(0, 4, 10**5)

# run the simulation and extract occupations with a suitable postprocessing function
stack, occupations = granad.evolution(
    stack, time_axis, field_func, postprocess=lambda x : jnp.diag(stack.eigenvectors.conj().T @ x @ stack.eigenvectors) )

# plot energy occupations
granad.show_energy_occupations(stack, occupations[::1000], time_axis[::1000])
