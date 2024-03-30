import granad

import jax 
from jax import lax
import jax.numpy as jnp

import numpy as np
import pdb

import time
import pyfftw
pyfftw.interfaces.cache.enable()

from functools import partial 

import matplotlib.pyplot as plt

def get_polarizability_function(
    stack,
    tau,
    polarization,
    coulomb_strength,
    hungry = True
):
    def _polarizability( omega ):
        x = sus( omega )
        ro = x @ jnp.linalg.inv( one - c @ x ) @ phi_ext
        return -pos @ ro
    
    one = jnp.identity(stack.electrons)
    pos = stack.positions[:, polarization]
    phi_ext = pos
    c = stack.coulomb * coulomb_strength
    sus = get_susceptibility_function( stack, tau, hungry )
    return _polarizability 

def get_susceptibility_function(
    stack,
    tau,
    hungry = True
):

    def _sum_subarrays( arr ):
        """Sums subarrays in 1-dim array arr. Subarrays are defined by n x 2 array indices as [ [start1, end1], [start2, end2], ... ]
        """
        arr = jnp.r_[0, arr.cumsum()][indices]
        return ( arr[:,1] - arr[:,0] ) 

    def _susceptibility( omega ):

        def susceptibility_element( site_1, site_2 ):   

            # calculate per-energy contributions
            prefac = eigenvectors[site_1, :] * eigenvectors[site_2, :] / delta_omega 
            f_lower = prefac * (energies - omega_low) * occupation
            g_lower = prefac * (energies - omega_low)* (1 - occupation )
            f_upper = prefac * (-energies + omega_up)* occupation 
            g_upper = prefac * (-energies + omega_up) * (1 - occupation )

            # TODO: this is retarded, change this
            f = jnp.r_[0, jnp.ravel(jnp.column_stack(( _sum_subarrays( f_lower ), _sum_subarrays( f_upper ) ))) ][mask]
            g = jnp.r_[0, jnp.ravel(jnp.column_stack(( _sum_subarrays( g_lower ), _sum_subarrays( g_upper ) ))) ][mask]
            #f = jnp.r_[0, _sum_subarrays( f_lower ) ][ mask ] + jnp.roll( jnp.r_[0, _sum_subarrays( f_upper ) ][ mask ], 1)
            #g = jnp.r_[0, _sum_subarrays( g_lower )][ mask ] + jnp.roll( jnp.r_[0, _sum_subarrays( g_upper )][ mask ], 1)

            b = jnp.fft.ihfft( f, n = 2 * f.size, norm="ortho" ) * jnp.fft.ihfft( g[::-1], n = 2 * f.size, norm="ortho"  )
            Sf1 = jnp.fft.hfft(b)[:-1]            
            Sf = -Sf1[::-1] + Sf1
            eq = 2.0 * Sf / (omega - omega_grid_extended + 1j / (2.0 * tau))
            return -jnp.sum(eq)
        
        if hungry:
            return jax.vmap(jax.vmap(susceptibility_element, (0, None), 0), (None, 0), 0) (sites, sites)        
        return lax.map( lambda i : lax.map(lambda j: susceptibility_element(i, j), sites), sites )
        
    # unpacking
    energies = stack.energies.real
    eigenvectors = stack.eigenvectors.real
    occupation = jnp.diag(stack.rho_0).real * stack.electrons / 2
    sites = jnp.arange( energies.size )
    freq_number = 2**12
    omega_max = jnp.real(max(stack.energies[-1], -stack.energies[0])) + 0.1
    omega_grid = jnp.linspace(-omega_max, omega_max, freq_number)

    # build two arrays sandwiching the energy values: omega_low contains all frequencies bounding energies below, omega_up bounds above
    upper_indices = jnp.argmax( omega_grid > energies[:, None], axis = 1) 
    omega_low = omega_grid[ upper_indices - 1 ]
    omega_up = omega_grid[ upper_indices  ]
    delta_omega = omega_up[0] - omega_low[0]

    omega_dummy = jnp.linspace(-2 * omega_grid[-1], 2 * omega_grid[-1], 2 * freq_number)
    omega_3 = omega_dummy[1:-1]
    omega_grid_extended = jnp.insert(omega_3, int(len(omega_dummy) / 2 - 1), 0)

    # indices for grouping energies into contributions to frequency points.
    # e.g. energies like [1,1,2,3] on a frequency grid [0.5, 1.5, 2.5, 3.5] 
    # the contribution array will look like [ f(1, eigenvector), f(1, eigenvector'), f(2, eigenvector_2), f(3, eigenvector_3) ]
    # we will have to sum the first two elements
    # we do this by building an array "indices" of the form: [ [0, 2], [2,3], [3, 4] ] 
    omega_low_unique, indices = jnp.unique( omega_low, return_index = True )
    indices = jnp.r_[jnp.repeat(indices,2)[1:], indices[-1] + 1].reshape(omega_low_unique.size, 2)

    # TODO: this is retarded, change this
    # mask for inflating the contribution array to the full size given by omega_grid. 
    comparison_matrix = omega_grid[:, None] == jnp.ravel(jnp.column_stack((jnp.unique(omega_low),jnp.unique(omega_up))))[None, :]    
    mask = ( jnp.argmax(comparison_matrix, axis=1) + 1) * comparison_matrix.any(axis=1) 

    return _susceptibility

def sim(a, b):

    coulomb_strength: float = 1.0
    minimum_omega: float = 0.0
    maximum_omega : float = 40.0
    discretization: int = 200
    polarization = 0
    tau = 0.1

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
    orbs = len(sb.orbitals)

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

    start = time.time()
    omegas, absorption_old = granad.absorption(stack, polarization, maximum_omega, tau = tau, discretization = discretization )
    time_old = time.time() - start 

    # omegas = jnp.linspace(minimum_omega, maximum_omega, discretization)
    omegas = jnp.array( omegas )
    start = time.time()
    alpha = get_polarizability_function( stack = stack, tau = tau, polarization = polarization, coulomb_strength=coulomb_strength, hungry = True )
    # absorption_new = jax.vmap(alpha)( jnp.array(omegas) ).imag * 4 * jnp.pi * omegas  # this kills mem if batch is too large
    absorption_new = lax.map( alpha, omegas ).imag * 4 * jnp.pi * omegas
    time_new = time.time() - start

    fig, axs = plt.subplots(2,1)
    axs[0].plot( omegas, absorption_old, '-', label = 'old')
    axs[0].plot( omegas, absorption_new, '--', label = 'new')
    axs[1].plot( omegas, absorption_new - absorption_old, '-', label = 'delta')
    plt.savefig(f"absorption_comparison_{a}_{b}.pdf")
    plt.close()

    return (orbs, time_old, time_new)

if __name__ == '__main__':
    orbs, old, new = [], [], []
    sizes = [4, 10, 20, 30, 40, 50, 60, 70]
    sizes = [30, 40, 45, 50]
    #sizes = [4, 10]
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
    plt.savefig("runtime_comparison.pdf")
    plt.close()

    # print( np.linalg.norm(xis_new[-1] - xis_old[-1]) )
    # pdb.set_trace()    
    # energies in units of delta_omega [E_1, E_2] = [z_1 * delta_omega, z_2 * delta_omega ]
    # identify energies that are less then 1 * delta_omega apart =>  z_1 and z_2 differ by a decimal
    # _, idxs, degeneracies = jnp.unique( jnp.round(energies / delta_omega), return_index = True, return_counts = True )
    # energies = energies[idxs]
    # vecs = vecs[:, idxs]
        # omega_dummy = np.linspace(-2 * omega_max, 2 * omega_max, 2 * freq_number)
    # omega_3 = omega_dummy[1:-1]
    # omega_grid_extended = np.insert(omega_3, int(len(omega_dummy) / 2 - 1), 0)
    # delta_omega = omega_grid_extended[1] - omega_grid_extended[0]
    # omegas = np.linspace(minimum_omega, maximum_omega, discretization)
    # omega_step = omegas[1] - omegas[0]
    # occupation = np.diag(stack.rho_0).real * stack.electrons / 2
    # xis_old = []
    # # omegas = [ omegas[0] ]
    # for omega in range(len(omegas)):        
    #     xis_old.append( granad._susceptibility(
    #             np.array(stack.hamiltonian).real,
    #             np.array(stack.energies).real,
    #             np.array(stack.eigenvectors).real,
    #             occupation,
    #             omegas[omega],
    #             stack.electrons,
    #             tau,
    #             omega_grid,
    #             omega_grid_extended,
    #         ) )