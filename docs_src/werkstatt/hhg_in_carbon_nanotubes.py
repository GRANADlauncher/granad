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

# ## HHG in carbon nanotubes
#
# In this example, we will study the interaction of carbon nanotubes with light in TD.
#
# ### Define functions
#
# Carbon nanotubes are slightly unusual: we will abuse the StackBuilder object to inject the orbitals we need directly from CNT coordinates and then build the stack. We thus need a function to construct the CNT coordinates and functions for the electric potentials corresponding to the external incident field.

# +

import os
import time

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import granad

# some globals to make re-runs less tedious
PLOT_GEOMETRY_ENERGY = False
# parameters like (n,m,unit_cells)
PARAMS = {
    "semiconducting": (5, 1, 2),
    "metallic_1": (4, 4, 10),
    "metallic_2": (4, 4, 60),
    "metallic": (6, 2, 5),
}
DIR = "/scratch/local/ddams/cnt_hhg"
FREQUENCY = 2.3

if not os.path.isdir(DIR):
    os.mkdir(DIR)


## non-flat geometries
# port from https://github.com/rgaveiga/nanotube/blob/main/nanotube.f90
# IMPORTANT: CC-DISTANCE=1.0
def nanotube(n, m, ncells):
    hcd = jnp.gcd(n, m)
    l = m * m + n * m + n * n
    rm = jnp.sqrt(3.0 * l)
    radius = rm / (2.0 * jnp.pi)

    dr = hcd if (n - m) % (3 * hcd) else 3 * hcd
    length = 3.0 * jnp.sqrt(l) / dr
    nc = int(2 * l / dr)
    phi = jnp.pi * (m + n) / l
    t = (n - m) / (2.0 * jnp.sqrt(l))
    for p1 in range(n):
        if (hcd + p1 * m) % n == 0:
            p2 = (hcd + p1 * m) / n
            break

    alpha = jnp.pi * (m * (2.0 * p2 + p1) + n * (2.0 * p1 + p2)) / l
    h = (3.0 * hcd) / (2.0 * jnp.sqrt(l))
    x = [radius]
    y = [0]
    z = [0]
    u = [0]
    x.append(radius * jnp.cos(phi))
    y.append(radius * jnp.sin(phi))
    z.append(t)
    u.append(u[0] + phi)

    for i in range(2, 2 * hcd):
        x.append(radius * jnp.cos(u[i - 2] + (2.0 * jnp.pi) / hcd))
        y.append(radius * jnp.sin(u[i - 2] + (2.0 * jnp.pi) / hcd))
        z.append(z[i - 2])
        u.append(u[i - 2] + (2.0 * jnp.pi) / hcd)

    for i in range(2 * hcd, 2 * nc * ncells):
        x.append(radius * jnp.cos(u[i - (2 * hcd)] + alpha))
        y.append(radius * jnp.sin(u[i - (2 * hcd)] + alpha))
        z.append(z[i - (2 * hcd)] + h)
        u.append(u[i - (2 * hcd)] + alpha)
    print(f"{n}, {m}, {ncells} :{len(x)} atoms")
    x_, y_, z_ = abs(min(x)), abs(min(y)), abs(min(z))
    return [[x0 + x_ for x0 in x], [y0 + y_ for y0 in y], [z0 + z_ for z0 in z]]


# -

# We now define a function to create a stack for a given nanotube setup and look at the energy landscape of one concrete stack

# +


def get_stack(params):
    sb = granad.StackBuilder()

    x, y, z = nanotube(*params)
    sb.orbitals += [
        granad.Orbital(orbital_id="pz", position=[x[i], y[i], z[i]])
        for i in range(len(x))
    ]

    # couplings
    hopping_nt = granad.DistanceCoupling(
        orbital_id1="pz",
        orbital_id2="pz",
        coupling_function=granad.gaussian_coupling(1e-1, [1.0], [-2.66]),
    )
    sb.set_hopping(hopping_nt)
    coulomb_nt = granad.DistanceCoupling(
        orbital_id1="pz",
        orbital_id2="pz",
        coupling_function=lambda d: 16.522 / (jnp.sqrt(d**2 + 1)),
    )
    sb.set_coulomb(coulomb_nt)

    return sb.get_stack()


# -

# ### Dipole moment

# We study the dipole moment in a TD simulation. First, we define functions for running the simulation and plotting

# +


def get_fourier_transform(t_linspace, function_of_time):
    function_of_omega = jnp.fft.fft(function_of_time) / len(t_linspace)
    omega_axis = (
        2
        * jnp.pi
        * len(t_linspace)
        / jnp.max(t_linspace)
        * jnp.fft.fftfreq(function_of_omega.shape[-1])
    )
    return omega_axis, function_of_omega


def plot_results(params, interpolate=False):

    data = jnp.load(f"{DIR}/{params}.npz")
    dipole_moment = data["dipole_moment"]
    electric_field = data["electric_field"]
    saveat = data["saveat"]
    positions = data["positions"]

    #  perform fourier transform
    for component in [2]:
        if interpolate:
            dip = jnp.interp(
                jnp.linspace(saveat.min(), saveat.max(), 2 * saveat.size),
                saveat,
                dipole_moment[:, component],
            )
        else:
            dip = dipole_moment[:, component]
        omega_axis, dipole_omega = get_fourier_transform(saveat, dip)
        _, field_omega = get_fourier_transform(saveat, electric_field[component])

        lower, upper = 0, 20
        idxs = jnp.argwhere(jnp.logical_and(lower < omega_axis, omega_axis < upper))
        omega_axis, dipole_omega, field_omega = (
            omega_axis[idxs],
            dipole_omega[idxs],
            field_omega[idxs],
        )

        arr = [i * FREQUENCY for i in range(1, 8)]
        for x in arr:
            plt.axvline(x=x, color="r", linestyle="--")

        # plot result
        plt.plot(omega_axis, jnp.abs(dipole_omega), "-")
        plt.xlabel(r"$E$ [eV]", fontsize=20)
        plt.ylabel(r"$d$", fontsize=25)
        plt.savefig(f"cnt_dip_{params}.pdf")
        plt.close()

        polarizability = dipole_omega / field_omega
        spectrum = -omega_axis * jnp.imag(polarizability)
        plt.plot(omega_axis, spectrum / jnp.max(jnp.abs(spectrum)))
        plt.savefig(f"abs_{params}.pdf")
        plt.close()


def plot_results_real_space(params):

    data = jnp.load(f"{DIR}/{params}.npz")
    dipole_moment = data["dipole_moment"]
    electric_field = data["electric_field"]
    saveat = data["saveat"]
    positions = data["positions"]

    #  perform fourier transform
    for component in range(1):
        plt.plot(saveat, dipole_moment[:, component], "--")
        plt.xlabel(r"$t$", fontsize=20)
        plt.ylabel(r"$d$", fontsize=25)
        plt.savefig(f"cnt_dip_rs_{params}.pdf")
        plt.close()


def td_sim(params):

    name = f"{DIR}/{params}.npy"

    if os.path.isfile(name):
        return

    stack = get_stack(params)

    if PLOT_GEOMETRY_ENERGY:
        # granad.show_eigenstate3D(stack, indicate_size = True )
        granad.show_energies(stack, name=f"energies_{params}.pdf")

    # simulation duration
    time_axis = jnp.linspace(0, int(400), int(1e5))
    saveat = time_axis[::10]
    print(saveat.size)

    amplitudes = [0, 0, 1e-5]
    peak = 2
    fwhm = 0.5
    field_func = granad.electric_field_pulse(
        amplitudes, FREQUENCY, stack.positions[0, :], peak, fwhm
    )

    tau = 5
    start = time.time()
    # run the simulation and extract occupations with a suitable postprocessing function
    stack, sol = granad.evolution(
        stack,
        time_axis,
        field_func,
        dissipation=granad.relaxation(tau),
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(atol=1e-10, rtol=1e-10),
    )
    print(time.time() - start)
    print(sol.result == diffrax.RESULTS.successful)

    # calculate dipole moment
    dipole_moment = granad.induced_dipole_moment(
        stack, jnp.diagonal(sol.ys, axis1=1, axis2=2)
    )

    electric_field = field_func(saveat)

    # Save the arrays to an .npz file
    jnp.savez(
        f"{DIR}/{params}.npz",
        dipole_moment=dipole_moment,
        electric_field=electric_field,
        saveat=saveat,
        positions=stack.positions,
    )

    plot_results(params)

    plot_results_real_space(params)


# -

# We run the simulations

# +

setup_sc = [
    (5, 0, 1),
    (5, 0, 2),
    (5, 0, 3),
    (5, 0, 4),
    (5, 0, 5),
    (5, 0, 6),
    (5, 0, 7),
    (5, 0, 8),
    (5, 0, 9),
    (10, 0, 1),
    (10, 0, 2),
    (10, 0, 3),
    (10, 0, 4),
    (10, 0, 5),
    (10, 0, 6),
    (10, 0, 7),
    (10, 0, 8),
    (10, 0, 9),
    (7, 3, 1),
    (7, 3, 2),
    (7, 3, 3),
    (7, 3, 4),
    (7, 3, 5),
    (7, 3, 6),
    (7, 3, 7),
    (10, 5, 1),
    (10, 5, 2),
    (10, 5, 3),
    (10, 5, 4),
    (10, 5, 5),
]
setup_metallic = [
    (8, 2, 1),
    (8, 2, 2),
    (8, 2, 3),
    (8, 2, 4),
    (8, 2, 5),
    (8, 2, 6),
    (8, 2, 7),
    (8, 2, 8),
    (8, 2, 9),
]

setup = setup_sc + setup_metallic

for i, params in enumerate(setup):
    if not params[0] == 7:
        continue
    td_sim(params)

# -
