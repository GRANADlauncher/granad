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

# ## RPA absorption for Carbon nanotubes
#
# In this example, we will study the interaction of carbon nanotubes with light in the RPA.
#
# ### Define functions
#
# Carbon nanotubes are slightly unusual: we will abuse the StackBuilder object to inject the orbitals we need directly from CNT coordinates and then build the stack. We thus need a function to construct the CNT coordinates and functions for the electric potentials corresponding to the external incident field.

# +

import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import granad
from flax import struct

@struct.dataclass
class Parameters:
    n : int
    m : int
    l : int
    polarization : int
    doping : int = 0
    coulomb_strength : float = 1.0
    hungry : bool = False
    omega_min : float = 4.0
    omega_max : float = 12.0
    omega_steps : int = 5
    skip : int = 100
    tau : float = 5.0
    has_edge_states : bool = False
    center : float = None
    width : float = None
    steepness : float = 1
    save_me : str = "cnt"
    do_plots : bool = False

# DIR = "/scratch/local/ddams/cnt_rpa"
# DIR = "cnt"

def to_str(params):
    ns = [ f"{x:.2e}" for x in [params.n, params.m, params.l,params.polarization] ]
    return '_'.join(ns)    

# TODO: remove this
p = Parameters( n = 0, m = 0, l = 40, polarization = 0 )
if not os.path.isdir(p.save_me):
    os.mkdir(p.save_me)

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
    return [x, y, z]

# -

# We now define a function to create a stack for a given nanotube setup and look at the energy landscape of one concrete stack

# +


def get_nanotube(params):
    sb = granad.StackBuilder()

    x, y, z = nanotube(params.n, params.m, params.l)
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

# ### Absorption

# We study the absorption properies within the RPA

# +

# Nanotube Type	Chirality (n, m)	Electrical Properties
# Zigzag	(n, 0)	Metallic if n is a multiple of 3, otherwise semiconductor
# Armchair	(n, n)	Metallic
# Chiral	(n, m)	Metallic if n - m is a multiple of 3, otherwise semiconductor

def rpa( params ):

    # dont repeat simulations
    if os.path.isfile( to_str(params) ):
        return

    omegas = jnp.linspace(params.omega_min, params.omega_max, params.omega_steps)
    tau = params.tau
    coulomb_strength = params.coulomb_strength
    stack = get_nanotube(params)
    polarization = params.polarization

    if params.do_plots:
        granad.show_eigenstate3D(stack, name = f"eigenstate_{to_str(params)}" )
        granad.show_energies(stack, name=f"energies_{to_str(params)}.pdf")

    start = time.time()
    alpha = granad.rpa_polarizability_function(
        stack=stack,
        tau=tau,
        polarization=polarization,
        coulomb_strength=coulomb_strength,
        hungry=params.hungry,
    )
    pol = jnp.abs(jax.vmap(alpha)(omegas))
    print(time.time() - start)
    np.save(f'{params.save_me}/{to_str(params)}.npy', pol)

def plot_rpa( params_list ):    
    for params in params_list:
        omegas = jnp.linspace(params.omega_min, params.omega_max, params.omega_steps)
        pol = np.load(f'{params.save_me}/{to_str(params)}.npy')    
        plt.plot(omegas, pol / jnp.max(pol), label=f"n, m, l = {params.n},{params.m},{params.l}")
        plt.xlabel(r"$\omega$ [eV]")
        plt.ylabel(r"$|\alpha| / |\alpha_{max}|$")

    plt.legend()
    plt.savefig(f"nt.pdf")
    plt.close()

params_zigzag = [ Parameters( n = i, m = 0, l = 10, polarization = 0 ) for i in [4, 8, 12 ]  ]
params_armchair = [ Parameters( n = i, m = i, l = 10,  polarization = 0 ) for i in [4, 8, 12] ]
params_tot = params_zigzag + params_armchair

params_armchair = []
params_zigzag = [params_zigzag[0]]

for p in params_tot:
    rpa( p )
    
plot_rpa( params_zigzag )
        
# -
