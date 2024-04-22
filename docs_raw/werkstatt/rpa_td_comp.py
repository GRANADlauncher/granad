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

# ## RPA-absorption spectrum of graphene nanoflakes
#
# This example demonstrates an advanced simulation. We will initialize a triangular graphene nanoflake and compute its TB absorption spectrum within the RPA. This is a frequency-domain simulation, so no need for time propagation.
#
# ### Set up the Stack
#
# The setup is analogous to the first tutorial. We build the Stack using the StackBuilder, which needs to know the material (a triangular armchair graphene nanoflake of 7.4 Å) and the coupling between the pz-orbitals (hopping rates and Coulomb interaction).

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# +
import granad

sb = granad.StackBuilder()

# geometry
triangle = granad.Triangle(20.0)
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

# We now compute the RPA absorption for an x-polarized uniform electric field. We do so by first computing the polarizability and then aking its imaginary part. Similar to the previous function, we have to tell GRANAD some parameters: a phenomennological broadening, coulomb strength scaling factor. In addition, we make it fast, but take a lot of memory with hungry = True.

# +

omegas = jnp.linspace(0, 20, 200)
polarization = 0
tau = 5
coulomb_strength = 1.0
alpha = granad.rpa_polarizability_function(
    stack=stack,
    tau=tau,
    polarization=polarization,
    coulomb_strength=coulomb_strength,
    hungry=False,
)
absorption = jax.vmap(alpha)(omegas).imag * 4 * jnp.pi * omegas

# -

# We now turn to the TD simulations

# +

amplitudes = [1e-5, 0, 0]
frequency = 2.75
peak = 2
fwhm = 0.5
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions[0, :], peak, fwhm
)

time_axis = jnp.linspace(0, 40, int(1e5))
import time
saveat = time_axis[::10]
start = time.time()
new_stack, sol = granad.evolution(
    stack,
    time_axis,
    field_func,
    granad.relaxation( tau ),
    saveat=saveat,
    stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10),
)
print(time.time() - start)
# Unpack the solutions. sol.ys contains the density matrices stacked into an array of shape T x N x N, where T is the number of timesteps, N is the number of orbitals. We only need the diagonal elements for the occupation numbers.
occupations = jnp.diagonal(sol.ys, axis1=1, axis2=2)

# -

# We now plot the TD absorption


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
dipole_moment = granad.induced_dipole_moment(stack, occupations)

plt.plot(saveat, dipole_moment)
plt.show()

omega_axis, dipole_omega = get_fourier_transform(
    saveat, dipole_moment[:, 0]
)
# we also need the x-component of the electric field as a single function
electric_field = granad.electric_field(amplitudes, frequency, stack.positions[0, :])(
    saveat
)
_, field_omega = get_fourier_transform(saveat, electric_field[0])


component = 0
lower, upper = omegas.min(),omegas.max()
idxs = jnp.argwhere(jnp.logical_and(lower < omega_axis, omega_axis < upper))        
omega_axis,dipole_omega,field_omega=omega_axis[idxs],dipole_omega[idxs],field_omega[idxs]

# plot result
polarizability = dipole_omega / field_omega
spectrum = -omega_axis * jnp.imag(polarizability)        
plt.plot(omegas, absorption / absorption.max(), '-', label = 'RPA' )
plt.plot(omega_axis, jnp.abs(spectrum) / jnp.max(jnp.abs(spectrum)), '--', label = 'TD')
plt.xlabel(r"$E$ [eV]", fontsize=20)
plt.ylabel(r"$\sigma$", fontsize=25)
plt.legend()
# plt.show()
plt.savefig(f"rpa_td_comp.pdf")
plt.close()

# -
