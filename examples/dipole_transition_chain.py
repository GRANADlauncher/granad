import granad

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

# build stack
sb = granad.StackBuilder()

chain = granad.Lattice(
    shape=granad.Chain(20),
    lattice_type=granad.LatticeType.CHAIN,
    lattice_edge=granad.LatticeEdge.NONE,
    lattice_constant=1,
)
sb.add("pz", chain)

hopping = granad.LatticeCoupling(
    orbital_id1="pz", orbital_id2="pz", lattice=chain, couplings=[0, -2.66]
)
sb.set_hopping(hopping)

coulomb = granad.LatticeCoupling(
    orbital_id1="pz",
    orbital_id2="pz",
    lattice=chain,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
sb.set_coulomb(coulomb)

# adatom
pos = sb.get_positions()
top_position = jnp.array([0.0, 0.0, 1.0])
spot = granad.Spot(position=top_position)
sb.add("A", spot)
sb.add("B", spot, occupation=0)

# set adatom couplings
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_hopping(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=2))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=2))

# set adatom-chain hoppings
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

stack = sb.get_stack( from_state = 0, to_state = 1)

# electric field parameters
amplitudes = [0, 0, 1]
frequency = 0.1
peak = 0.05
fwhm = 0.5

# choose an x-polarized electric field propagating in z-direction
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions, peak, fwhm
)

# transition between "A" and "B"; dipole moment in z-direction
transition_function = granad.dipole_transitions( {("A", "B") : [0, 0, 1.0]}, stack  )

# propagate in time
time_axis = jnp.linspace(0, 1, 2000)

# run the simulation and extract occupations with a suitable postprocessing function
stack, occupations = granad.evolution(
    stack, time_axis, field_func, dissipation = granad.relaxation(0.1), transition = transition_function)

# show energy occupations
plt.plot(time_axis, stack.electrons*jnp.diagonal(occupations, axis1 = 1, axis2 = 2).real)
plt.show()
