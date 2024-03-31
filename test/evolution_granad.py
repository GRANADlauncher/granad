import granad

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union

import time  # Import the time module for benchmarking
import pdb

def evolution_diffrax( 
    stack: granad.Stack,
    time: jax.Array,
    field: granad.FieldFunc,
    dissipation: granad.DissipationFunc = None,
    coulomb_strength: float = 1.0,
    transition: Callable = lambda c, h, e: h,    
    saveat = None, 
    solver = diffrax.Dopri5(), 
    rtol = 1e-8, 
    atol = 1e-8 ):    

    def rhs(time, rho, args):
        print(101)
        e_field, delta_rho = field(time), rho - rho_stat
        charge = -jnp.diag(delta_rho) * stack.electrons
        p_ext = jnp.sum(stack.positions * e_field.real.T, axis=1)
        p_ind = coulomb @ charge
        h_total = transition(charge, stack.hamiltonian, e_field) + jnp.diag(
            p_ext - p_ind
        )
        return -1j * (h_total @ rho - rho @ h_total) + dissipation(rho, rho_stat)

    coulomb = stack.coulomb * coulomb_strength
    rho_stat = stack.eigenvectors @ stack.rho_stat @ stack.eigenvectors.conj().T

    term = ODETerm(rhs)
    rho_init = stack.eigenvectors @ stack.rho_0 @ stack.eigenvectors.conj().T
    saveat = SaveAt( ts = time if saveat is None else saveat )
    stepsize_controller = PIDController(rtol=rtol, atol=atol)    
    sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=time[1] - time[0], y0=rho_init, saveat=saveat,
                    stepsize_controller=stepsize_controller)
    return (
        stack.replace(rho_0=stack.eigenvectors.conj().T @ sol.ys[-1] @ stack.eigenvectors),
        sol
    )

def sim(a, b):
    # build stack
    sb = granad.StackBuilder()

    # add graphene
    graphene = granad.Lattice(
        shape=granad.Rectangle(a, b),
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
    time_axis = jnp.linspace(0, 2, int(1e6))
    time_axis_old = time_axis
    time_axis_new = time_axis[::100]

    start_time = time.time()
    stack_new, sol = evolution_diffrax(
        stack,
        time_axis,
        field_func,
        granad.relaxation(0.1),
        saveat= time_axis_new,
        )
    time_new = time.time() - start_time 

    start_time = time.time()
    stack_old, occupations_old = granad.evolution(
        stack,
        time_axis,
        field_func,
        granad.relaxation(0.1),
        postprocess = jnp.diag
    )
    time_old = time.time() - start_time 

    # calculate dipole moment
    dipole_moment_old = granad.induced_dipole_moment(stack, occupations_old)
    dipole_moment_new = granad.induced_dipole_moment(stack, jnp.diagonal( sol.ys, axis1=1, axis2=2 ))

    plt.plot( time_axis_old, dipole_moment_old, label = 'old' )
    plt.plot( time_axis_new, dipole_moment_new, '--', label = 'new' )
    plt.legend()
    plt.savefig(f"comparison_{a}_{b}.pdf")
    plt.close()
    return (stack.electrons, time_old, time_new)

if __name__ == '__main__':
    orbs, old, new = [], [], []
    sizes = [4]
    for a in sizes:
        orb, time_old, time_new = sim( a, 4 )
        orbs.append( orb )
        old.append( time_old )
        new.append( time_new )

    plt.plot( orbs, old, label = "old" )
    plt.plot( orbs, new, label = "new" )
    plt.xlabel( "size" )
    plt.ylabel( "time" )
    plt.legend()
    plt.savefig("evolution_runtime_comparison.pdf")
    plt.close()
