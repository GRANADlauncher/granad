import granad

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

## custom function for performing the fourier transform
def get_fourier_transform(t_linspace, function_of_time):
    function_of_omega = np.fft.fft(function_of_time) / len(t_linspace)
    omega_axis = (
        2
        * np.pi
        * len(t_linspace)
        / np.max(t_linspace)
        * np.fft.fftfreq(function_of_omega.shape[-1])
    )
    return omega_axis, function_of_omega


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

amplitudes = [1, 0, 0]
frequency = 1
peak = 2
fwhm = 0.5

# choose an x-polarized electric field propagating in z-direction
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions[0, :], peak, fwhm
)

# propagate in time
time_axis = jnp.linspace(0, 10, 10**4)

new_stack, occupations = granad.evolution(
    stack,
    time_axis,
    field_func,
    granad.relaxation(0.1),
    postprocess = jnp.diag
)

# calculate dipole moment
dipole_moment = granad.induced_dipole_moment(stack, occupations)

#  perform fourier transform
omega_axis, dipole_omega = get_fourier_transform(time_axis, dipole_moment[:, 0])

# we also need the x-component of the electric field as a single function
electric_field = granad.electric_field(
    amplitudes, frequency, stack.positions[0, :]
)(time_axis)
_, field_omega = get_fourier_transform(time_axis, electric_field[0])

# plot result
omega_max = 100
component = 0
polarizability = dipole_omega / field_omega
spectrum = -omega_axis[:omega_max] * np.imag(polarizability[:omega_max])
plt.plot(omega_axis[:omega_max], np.abs(spectrum) ** (1 / 2))
plt.xlabel(r"$\hbar\omega$", fontsize=20)
plt.ylabel(r"$\sigma(\omega)$", fontsize=25)
plt.legend()
plt.show()
