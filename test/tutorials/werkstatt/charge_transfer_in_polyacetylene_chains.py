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

# ## Charge transfer in Polyacetylene Chains
#
# In this example, we will study current flow through Polyacetylene Chains due to the application of an external electric field.
#
# ### Define functions
#

# We first define necessary functions

# +

import os
import time
import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import granad
from flax import struct

@struct.dataclass
class Parameters:
    field_strength: float
    geom : float
    t_0 : float
    n_atoms : int
    doping : int = 0
    time_start : float = 0.0
    time_end : float = 1e3
    time_steps : int = int(1e5)
    skip : int = 100
    tau : float = 5.0
    has_edge_states : bool = False
    center : float = None
    width : float = None
    steepness : float = 1
    save_me : str = "/scratch/local/ddams/poly_current"
    do_plots : bool = False

def to_str(params):
    return f'{params.field_strength}_{params.geom}_{params.n_atoms}_{params.doping}'
    

# TODO: remove this
p = Parameters( field_strength = 1e-5, geom = 0.05, t_0 =  1.0, n_atoms = 500) 
if not os.path.isdir(p.save_me):
    os.mkdir(p.save_me)

    
def stack_poly( params ):
    """constructs a polyacetylene chain with specified parameters"""

    n, doping, geom, t_0, edge  = params.n_atoms, params.doping, params.geom, params.t_0, params.has_edge_states
    
    fac = 1 if edge else -1
    displacement = 0.2
    positions = jnp.arange(n) + fac * displacement * (jnp.arange(n) % 2) + 3
    distances = [1.0 - displacement, 1.0 + displacement]

    sb = granad.StackBuilder()
    sb.orbitals += [
        granad.Orbital(orbital_id="pz", position=[p, 0, 0]) for p in positions
    ]

    hoppings = [t_0 + 2 * geom, t_0 - 2 * geom]
    cf = granad.gaussian_coupling(1e-1, distances, hoppings)

    # couplings
    hopping_nt = granad.DistanceCoupling(
        orbital_id1="pz", orbital_id2="pz", coupling_function=cf
    )
    sb.set_hopping(hopping_nt)

    coulomb_nt = granad.DistanceCoupling(
        orbital_id1="pz",
        orbital_id2="pz",
        coupling_function=lambda d: 16.522 / (jnp.sqrt(d**2 + 1)),
    )
    sb.set_coulomb(coulomb_nt)
    stack = sb.get_stack(doping=doping)
    return stack


def get_time_saveat(params):
    time_axis = jnp.linspace(params.time_start, params.time_end, params.time_steps)
    saveat = time_axis[::params.skip]
    return time_axis, saveat

def get_electric_field_func(params):
    amplitudes = [params.field_strength, 0, 0]
    if params.center:
        # TODO: ramp
        return lambda t: jnp.array(amplitudes)[:, None] * ramp(t, params.steepness)
    return lambda t: jnp.array(amplitudes)[:, None] * ramp(t, params.steepness)
    
def td_sim(params):    
    time_axis, saveat = get_time_saveat(params)
    field_func = get_electric_field_func( params )    
    stack = stack_poly(params)
    
    if params.do_plots:
        plt.plot(saveat, field_func(saveat).T)
        plt.savefig(f"efield_{to_str(params)}.pdf")
        plt.close()    
        granad.show_energies(stack, name=f"chains_energies_{to_str(params)}.pdf")
        granad.show_eigenstate3D(stack, indicate_size = True, name=f"chains_eigenstate_{to_str(params)}.pdf" )

    start = time.time()
    # run the simulation and extract occupations with a suitable postprocessing function
    stack, sol = granad.evolution(
        stack,
        time_axis,
        field_func,
        dissipation=granad.relaxation(params.tau),
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(atol=1e-10, rtol=1e-10),
    )
    print(time.time() - start)
    print(sol.result == diffrax.RESULTS.successful)

    # Save the arrays to an .npz file
    jnp.savez(
        f"{params.save_me}/{to_str(params)}.npz",
        rhos=sol.ys
    )


@jax.jit    
def current_global( electrons, rhos, rho_stat, v ):
    return jnp.einsum(
                    "ijk,kjr->ir", -electrons * (rhos - rho_stat[None, :, :]), v
                )

def current_local( electrons, rhos, rho_stat, v, x, y ):
    return 2 * (v[x, y, component] * (rhos[:, y, x] - rho_stat[y, x]))
    
def plot_results_real_space(params_list, sites = []):
    for params in params_list:
        data = jnp.load(f"{params.save_me}/{to_str(params)}.npz")
        rhos = data["rhos"]

        _, saveat = get_time_saveat(params)
        field_func = get_electric_field_func( params )
        electric_field = field_func( saveat )

        stack = stack_poly(params)
        v = granad.velocity_operator(stack)
        rho_stat = granad.to_site_basis(stack, stack.rho_stat)

        for site in sites:
            current = current_global( stack.electrons, rhos, rho_stat, v, *site ).imag
            plt.plot(saveat, current, "--", label=f"doping={params.doping}, sites = {sites}")
            plt.xlabel(r"$t$")
            plt.ylabel(r"$j$")            

        if not sites:
            current = current_global( stack.electrons, rhos, rho_stat, v )[:,0].real
            plt.plot(saveat, current, "--", label=f"doping={params.doping}, total current")
            plt.xlabel(r"$t$")
            plt.ylabel(r"$j$")
    
    plt.legend()
    plt.savefig(f"current.pdf")
    plt.close()


def rectangle(x, center, width, steepness):
    return 0.5 * (
        jnp.tanh(steepness * (x - center + width / 2))
        - jnp.tanh(steepness * (x - center - width / 2))
    )


def ramp(x, steepness):
    return jnp.tanh(steepness * x)


# -

# ### Current

# We study the time-varying current in a TD simulation. 

# +

# ssh_params = [0.22399999999999998, 2.5, False]
metal_params = [0.0, 1.0, False]
ssh_params = [0.05, 1.0, False]

# ssh
p = Parameters( field_strength = 1e-5, geom = 0.05, t_0 =  1.0, n_atoms = 80, do_plots = True)
params_list = [ p.replace( field_strength = f) for f in [1e-5, 5*1e-5, 1e-4, 5*1e-4, 1e-3, 5*1e-3] ]
for params in params_list:
    td_sim(params)
sites = []
plot_results_real_space(params_list, sites)

# metal
p = Parameters( field_strength = 1e-5, geom = 0.00, t_0 =  1.0, n_atoms = 300)
params_list = [ p.replace( field_strength = f) for f in [1e-5, 5*1e-5, 1e-4, 5*1e-4, 1e-3, 5*1e-3] ]
for params in params_list:
    td_sim(params)
sites = []
plot_results_real_space(params_list, sites)

    
# define the sites we want to compute the current through in the way (from, to)
# sites = [ (39,40) ]
# sites = [ (-2, -1) ]
# sites = []
# plot_results_real_space(params_list, sites)

# -
