# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: base
# ---

# ## Absorption spectrum of graphene nanoflakes
#
# This example demonstrates an advanced simulation. We will initialize a triangular graphene nanoflake, specify the couplings, simulate it under pulsed illumination and compute the absorption spectrum.
#
# ### Set up the Stack
#
# The setup is analogous to the first tutorial. We build the Stack using the StackBuilder, which needs to know the material (a triangular armchair graphene nanoflake of 7.4 Å) and the coupling between the pz-orbitals (hopping rates and Coulomb interaction).

import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# +
import granad

sb = granad.StackBuilder()

# geometry
triangle = granad.Triangle(7.4)
graphene = granad.Lattice(
    shape=triangle,
    lattice_type=granad.LatticeType.HONEYCOMB,
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_constant=2.46,
)
sb.add("pz", graphene)

# couplings
hopping_graphene = granad.LatticeCoupling(
    orbital_id1="pz",
    orbital_id2="pz",
    lattice=graphene,
    couplings=[0, -2.66],  # list of hopping amplitudes like [onsite, nn, ...]
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

stack = sb.get_stack()
# -

# We now define the electric field as an x-polarized pulse of frequency peaking at 2, with fwhm of 0.5.

amplitudes = [1, 0, 0]
frequency = 1
peak = 2
fwhm = 0.5
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions[0, :], peak, fwhm
)

# We assume the structure looses energy relaxing across all channels with a rate of 0.1

loss_function = granad.relaxation(0.1)

# We propagate the structure in time. The function evolution returns a tuple. First is a new stack with its initial state given by the last state in the time evolution.  Second is a solution object provided by the diffrax library GRANAD uses for solving the underlying ODE with Dormand-Prince's 5/4 method and a PID controller for adapting the step size. It contains the density matrices, accessible via its attribute soltuion.ys. The saveat argument controls at which times we want to sample the density matrix.

time_axis = jnp.linspace(0, 10, int(1e5))
import time

saveat = time_axis[::100]
start = time.time()
new_stack, sol = granad.evolution(
    stack,
    time_axis,
    field_func,
    loss_function,
    saveat=saveat,
    stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10),
)
print(time.time() - start)
# Unpack the solutions. sol.ys contains the density matrices stacked into an array of shape T x N x N, where T is the number of timesteps, N is the number of orbitals. We only need the diagonal elements for the occupation numbers.

occupations_new = jnp.diagonal(sol.ys, axis1=1, axis2=2)

# We can also calculate the solution without diffrax with the previous solver, equivalent to first order Runge-Kutta
_, occupations_old = granad.evolution_old(
    stack, time_axis, field_func, loss_function, postprocess=jnp.diag
)

# We now plot the absorption


# +
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


# dipole moments in fourier space
dipole_moment_new = granad.induced_dipole_moment(stack, occupations_new)
omega_axis_new, dipole_omega_new = get_fourier_transform(
    saveat, dipole_moment_new[:, 0]
)
dipole_moment_old = granad.induced_dipole_moment(stack, occupations_old)
omega_axis_old, dipole_omega_old = get_fourier_transform(
    time_axis, dipole_moment_old[:, 0]
)

# we also need the x-component of the electric field as a single function
electric_field = granad.electric_field(amplitudes, frequency, stack.positions[0, :])(
    saveat
)
_, field_omega_new = get_fourier_transform(saveat, electric_field[0])

electric_field = granad.electric_field(amplitudes, frequency, stack.positions[0, :])(
    time_axis
)
_, field_omega_old = get_fourier_transform(time_axis, electric_field[0])


omega_max = 100
component = 0
# alpha = p / E
polarizability = dipole_omega_new / field_omega_new
polarizability_old = dipole_omega_old / field_omega_old
# sigma ~ Im[alpha]
spectrum_old = -omega_axis_old[:omega_max] * np.imag(polarizability_old[:omega_max])
spectrum = -omega_axis_new[:omega_max] * np.imag(polarizability[:omega_max])
plt.plot(omega_axis_old[:omega_max], np.abs(spectrum_old) ** (1 / 2), label="old")
plt.plot(omega_axis_new[:omega_max], np.abs(spectrum) ** (1 / 2), "--", label="new")
plt.legend()
plt.xlabel(r"$\hbar\omega$", fontsize=20)
plt.ylabel(r"$\sigma(\omega)$", fontsize=25)
plt.show()
