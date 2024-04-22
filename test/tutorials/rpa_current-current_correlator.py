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

# ## RPA current-current correlator
#
# This example demonstrates an advanced simulation. We will initialize a triangular graphene nanoflake and compute its TB absorption spectrum within the RPA. This is a frequency-domain simulation, so no need for time propagation.
#
# ### Set up the Stack
#
# The setup is analogous to the first tutorial. We build the Stack using the StackBuilder, which needs to know the material (a triangular armchair graphene nanoflake of 7.4 Å) and the coupling between the pz-orbitals (hopping rates and Coulomb interaction).

# +


def rpa_polarizability_function(
    stack, tau, polarization, coulomb_strength, phi_ext=None, hungry=True
):
    def _polarizability(omega):
        ro = sus(omega) @ phi_ext
        return -pos @ ro

    pos = stack.positions[:, polarization]
    phi_ext = pos if phi_ext is None else phi_ext
    sus = rpa_susceptibility_function(stack, tau, coulomb_strength, hungry)
    return _polarizability


def rpa_susceptibility_function(stack, tau, coulomb_strength, hungry=True):
    def _rpa_susceptibility(omega):
        x = sus(omega)
        return x @ jnp.linalg.inv(one - c @ x)

    sus = bare_susceptibility_function(stack, tau, hungry)
    c = stack.coulomb * coulomb_strength
    one = jnp.identity(stack.hamiltonian.shape[0])

    return _rpa_susceptibility


def bare_susceptibility_function(stack, tau, hungry=True):

    def _sum_subarrays(arr):
        """Sums subarrays in 1-dim array arr. Subarrays are defined by n x 2 array indices as [ [start1, end1], [start2, end2], ... ]"""
        arr = jnp.r_[0, arr.cumsum()][indices]
        return arr[:, 1] - arr[:, 0]

    def _susceptibility(omega):

        def susceptibility_element(site_1, site_2):
            # calculate per-energy contributions
            prefac = eigenvectors[site_1, :] * eigenvectors[site_2, :] / delta_omega
            f_lower = prefac * (energies - omega_low) * occupation
            g_lower = prefac * (energies - omega_low) * (1 - occupation)
            f_upper = prefac * (-energies + omega_up) * occupation
            g_upper = prefac * (-energies + omega_up) * (1 - occupation)
            f = (
                jnp.r_[0, _sum_subarrays(f_lower)][mask]
                + jnp.r_[0, _sum_subarrays(f_upper)][mask2]
            )
            g = (
                jnp.r_[0, _sum_subarrays(g_lower)][mask]
                + jnp.r_[0, _sum_subarrays(g_upper)][mask2]
            )
            b = jnp.fft.ihfft(f, n=2 * f.size, norm="ortho") * jnp.fft.ihfft(
                g[::-1], n=2 * f.size, norm="ortho"
            )
            Sf1 = jnp.fft.hfft(b)[:-1]
            Sf = -Sf1[::-1] + Sf1
            eq = 2.0 * Sf / (omega - omega_grid_extended + 1j / (2.0 * tau))
            return -jnp.sum(eq)

        if hungry:
            return jax.vmap(
                jax.vmap(susceptibility_element, (0, None), 0), (None, 0), 0
            )(sites, sites)
        return jax.lax.map(
            lambda i: jax.lax.map(lambda j: susceptibility_element(i, j), sites), sites
        )

    # unpacking
    energies = stack.energies.real
    eigenvectors = stack.eigenvectors.real
    occupation = jnp.diag(stack.rho_0).real * stack.electrons / stack.spin_degeneracy
    sites = jnp.arange(energies.size)
    freq_number = 2**12
    omega_max = jnp.real(max(stack.energies[-1], -stack.energies[0])) + 0.1
    omega_grid = jnp.linspace(-omega_max, omega_max, freq_number)

    # build two arrays sandwiching the energy values: omega_low contains all frequencies bounding energies below, omega_up bounds above
    upper_indices = jnp.argmax(omega_grid > energies[:, None], axis=1)
    omega_low = omega_grid[upper_indices - 1]
    omega_up = omega_grid[upper_indices]
    delta_omega = omega_up[0] - omega_low[0]

    omega_dummy = jnp.linspace(-2 * omega_grid[-1], 2 * omega_grid[-1], 2 * freq_number)
    omega_3 = omega_dummy[1:-1]
    omega_grid_extended = jnp.insert(omega_3, int(len(omega_dummy) / 2 - 1), 0)

    # indices for grouping energies into contributions to frequency points.
    # e.g. energies like [1,1,2,3] on a frequency grid [0.5, 1.5, 2.5, 3.5]
    # the contribution array will look like [ f(1, eigenvector), f(1, eigenvector'), f(2, eigenvector_2), f(3, eigenvector_3) ]
    # we will have to sum the first two elements
    # we do this by building an array "indices" of the form: [ [0, 2], [2,3], [3, 4] ]
    omega_low_unique, indices = jnp.unique(omega_low, return_index=True)
    indices = jnp.r_[jnp.repeat(indices, 2)[1:], indices[-1] + 1].reshape(
        omega_low_unique.size, 2
    )

    # mask for inflating the contribution array to the full size given by omega_grid.
    comparison_matrix = omega_grid[:, None] == omega_low_unique[None, :]
    mask = (jnp.argmax(comparison_matrix, axis=1) + 1) * comparison_matrix.any(axis=1)
    comparison_matrix = omega_grid[:, None] == jnp.unique(omega_up)[None, :]
    mask2 = (jnp.argmax(comparison_matrix, axis=1) + 1) * comparison_matrix.any(axis=1)
    return _susceptibility


import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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

# We now compute the RPA absorption for an x-polarized uniform electric field. We do so by first computing the polarizability and then aking its imaginary part. Similar to the previous function, we have to tell GRANAD some parameters: a phenomennological broadening, coulomb strength scaling factor. In addition, we make it fast, but take a lot of memory with hungry = True.

# +
omegas = jnp.linspace(0, 20, 200)
polarization = 0
tau = 0.1
coulomb_strength = 1.0
alpha = granad.rpa_polarizability_function(
    stack=stack,
    tau=tau,
    polarization=polarization,
    coulomb_strength=coulomb_strength,
    hungry=True,
)
absorption = jax.lax.map(alpha, omegas).imag * 4 * jnp.pi * omegas
plt.plot(omegas, absorption)
plt.show()
# -
