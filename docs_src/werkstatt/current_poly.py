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

# ## Bond currents in polyacetylene
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
    frequency : float = 2.75
    peak : float = 2.0
    fwhm : float = 0.5
    time_start : float = 0.0
    time_end : float = 20
    time_steps : int = int(1e5)
    skip : int = 10
    tau : float = 5.0
    has_edge_states : bool = False
    center : float = None
    width : float = None
    steepness : float = 1
    save_me : str = "poly_results"
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
    frequency = params.frequency
    peak = params.peak
    fwhm = params.fwhm

    field_func = granad.electric_field_pulse(
        amplitudes, frequency, jnp.array([0., 0., 0.]), peak, fwhm
    )
    return field_func

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
def expectation_value( electrons, rhos, rho_stat, operator  ):
    return jnp.einsum(
        "ijk,kjr->ir", -electrons * (rhos - rho_stat[None, :, :]), operator
    )

def expectation_value_local( electrons, rhos, rho_stat, v, x, y ):
    return 2 * (v[x, y, component] * (rhos[:, y, x] - rho_stat[y, x]))
    

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

def plot_results_real_space(params_list, sites = []):
    
    # we assume that all stacks in params list start identical
    stack = stack_poly(params_list[0])
    v = granad.velocity_operator(stack)
    x = granad.position_operator( stack )
    rho_stat = granad.to_site_basis(stack, stack.rho_stat)
    _, saveat = get_time_saveat(params_list[0])
    amplitudes = [params_list[0].field_strength, 0, 0]

    for params in params_list:
        data = jnp.load(f"{params.save_me}/{to_str(params)}.npz")
        rhos = data["rhos"]

        for site in sites:
            current = expectation_value_local( stack.electrons, rhos, rho_stat, v, *site ).imag
            plt.plot(saveat, current, "--", label=f"doping={params.doping}, sites = {sites}")

        if not sites:
            current = expectation_value( stack.electrons, rhos, rho_stat, v )[:,0].real
            omega_axis_1, current_omega = get_fourier_transform(
                saveat, current
            )

            dipole_moment = expectation_value( stack.electrons, rhos, rho_stat, x )[:,0].real
            omega_axis_2, dipole_omega = get_fourier_transform(
                saveat, dipole_moment
            )

    plt.plot(saveat, current / current.max(), "--", label=f"doping={params.doping}, total current")
    plt.plot(saveat, dipole_moment / dipole_moment.max(), "--", label=f"doping={params.doping}, total dipole moment")    
    plt.xlabel(r"$t$")
    plt.ylabel(r"$j$ and $p$")    
    plt.legend()
    plt.savefig(f"time_domain.pdf")
    plt.close()

    component = 0
    lower, upper = 0, 10    
    idxs_1 = jnp.argwhere(jnp.logical_and(lower < omega_axis_1, omega_axis_2 < upper))
    idxs_2 = jnp.argwhere(jnp.logical_and(lower < omega_axis_2, omega_axis_2 < upper))
    
    electric_field = get_electric_field_func(params)( saveat )
    _, field_omega = get_fourier_transform(saveat, electric_field[0])

    omega_axis_1,dipole_omega,field_omega_1=omega_axis_1[idxs_1],dipole_omega[idxs_1],field_omega[idxs_1]
    omega_axis_2,current_omega,field_omega_2=omega_axis_2[idxs_2],current_omega[idxs_2],field_omega[idxs_2]

    r = jnp.abs(current_omega) / omega_axis_2
    # i = jnp.abs(current_omega.imag) / omega_axis_2
    plt.plot(omega_axis_2, r / r.max(), '-', label = 'current' )
    # plt.plot(omega_axis_2, i / i.max(), '.', label = 'current i' )
    
    polarizability = dipole_omega / field_omega_1
    # spectrum = -omega_axis_1 * jnp.imag(polarizability)
    spectrum = jnp.abs( dipole_omega ) 
    plt.plot(omega_axis_1, jnp.abs(spectrum) / jnp.max(jnp.abs(spectrum)), '--', label = 'direct')
    plt.xlabel(r"$\omega$ [eV]", fontsize=20)
    plt.ylabel(r"$|p| / |p_{max}|$", fontsize=20)
    plt.legend()
    # plt.show()
    plt.savefig(f"dipole_direct_current_comparison.pdf")
    plt.close()

# -

# ### Current

# We study the time-varying current in a TD simulation. 

# +

# ssh_params = [0.22399999999999998, 2.5, False]
metal_params = [0.0, 1.0, False]
ssh_params = [0.05, 1.0, False]
run = True

# ssh
p = Parameters( field_strength = 1e-5, geom = 0.05, t_0 =  1.0, n_atoms = 100, do_plots = False)
fs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
fs = jnp.linspace(1e-5, 0.5, 100)
params_list = [ p.replace(geom = 0.00, field_strength = f) for f in fs ]
params_list = [ p.replace( field_strength = f) for f in fs ]
params_list = [ p ]
if run:
    for params in params_list:
        td_sim(params)
sites = []
plot_results_real_space(params_list, sites)

# -
