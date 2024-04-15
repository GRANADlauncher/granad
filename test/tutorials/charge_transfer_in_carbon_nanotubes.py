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

# ## Charge transfer in Carbon Nanotubes
#
# In this example, we will study current flow through carbon nanotubes due to the application of an external electric field.
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
DIR = "/scratch/local/ddams/cnt_current"
DIR = "/home/david/cnt_current"
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

# ### Current

# We study the time-varying current in a TD simulation. First, we define as usual a function for running the simulation

# +


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

    # Save the arrays to an .npz file
    jnp.savez(
        f"{DIR}/{params}.npz",
        rhos=sol.ys,
        electric_field=field_func(saveat),
        saveat=saveat,
    )


# -

# We run the simulations for a few semiconducting and metallic CNTs

# +

setup_sc = [(5, 0, 4), (7, 3, 1), (7, 3, 2)]
setup_metallic = [
    (8, 2, 1),
    (8, 2, 2),
]

setup = setup_sc + setup_metallic

for i, params in enumerate(setup):
    td_sim(params)

# -

# We now want to visualize the induced current. We do so by looking at pairs of sites in the CNT. We first define a function that does this.

# +


def plot_results_real_space(params, sites):
    # reconstruct the stack
    stack = get_stack(params)

    # compute the velocity/current operator (in GRANAD's units, they are identical)
    v = granad.velocity_operator(stack)
    electrons = stack.electrons
    rho_stat = granad.to_site_basis(stack, stack.rho_stat)

    data = jnp.load(f"{DIR}/{params}.npz")
    rhos = data["rhos"]
    electric_field = data["electric_field"]
    saveat = data["saveat"]

    # pick the slices of the position operator that we want to keep
    for collection in sites:
        j = [
            sum(v[c, i] * (rhos[c, :] - rho_stat[c]) for c in collection)
            for i in range(3)
        ]
        # jnp.einsum( 'ijk,kjr->ir', -electrons*(rhos - rho_stat[None,:,:]), v[s] )

        #  perform fourier transform
        for component in range(3):
            plt.plot(saveat, j[component], "--")
            plt.xlabel(r"$t$", fontsize=20)
            plt.ylabel(r"$d$", fontsize=25)
            plt.savefig(f"cnt_current_rs_{params}.pdf")
            plt.close()


# define the sites we want to compute the current through
sites = []
for i, params in enumerate(setup):
    plot_results_real_space(params, sites)

# -
