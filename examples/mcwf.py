import granad

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

# 1. pick initial state vector
# 2. evolve state vector one time step according to (1 - iHd), where H = H_0 - i/2 \sum_i J^{\dagger}_iJ_i
# 3. calculate its norm and p = 1 - |psi'|, where p is change in quantum jump probability
# 4. random number r > p => no jump => psi' = psi'/sqrt(1 - p)
# 5. r < p => jump => psi' = sqrt(d/p) * J_m * psi', where J_m is chosen according to pm/p

# we measure time in units of 1/hbar and dissipation parameters in units of sqrt(hbar)
# this means, we effectively set hbar = 1

def evolution_mcwf( stack, time, field, dissipation_matrix, transition, state ):
    """Computes the MCWF evolution of an initial state.

    :param stack: stack instance
    :param time: time axis
    :param field: electric field function
    :param dissipation_matrix: matrix such that the entry :math:`M[i,j]` is the transition from state i to j
    :param transition: dipole transition function
    :param state: initial state
    """
    dt = time[1] - time[0]

    dissipation_array = np.sum( dissipation_matrix ** 2, axis = 1)
    hamiltonian_term = np.diag( dissipation_array )
    
    for t in time:
        # effective hamiltonian is given by H_eff = H - i/2 \sum_m J^{\dagger}_m J_m, 
        ham_eff = transition( stack.hamiltonian )
        state_new = (1-1j*dt*ham_eff) * state
        dps = jnp.abs(state)**2 * dissipation_array
        dp = dps.sum()
        
        if np.random.rand() < dp:
            # quantum jump => change states
            state = np.sqrt(dt/dp) * state_new * np.random.choice( foo, p = dps )
        else:
            # no quantum jump => normalize state
            state = new_state / np.sqrt( 1 - dp )        
            
    return state                            

# build stack
sb = granad.StackBuilder()

# adatom
pos = sb.get_positions()
top_position = jnp.array([0.0, 0.0, 1.0])
spot = granad.Spot(position=top_position)
sb.add("A", spot)
sb.add("B", spot, occupation=0)

# set adatom couplings
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_hopping(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=2))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=2))

stack = sb.get_stack( from_state = 0, to_state = 1)

# electric field parameters
amplitudes = [0, 0, 1]
frequency = 0.1
peak = 0.05
fwhm = 0.5

# choose an x-polarized electric field propagating in z-direction
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions, peak, fwhm
)

# transition between "A" and "B"; dipole moment in z-direction
transition_function = granad.dipole_transitions( {("A", "B") : [0, 0, 1.0]}, stack  )

# propagate in time
time_axis = jnp.linspace(0, 10, 1000)

# run the simulation and extract occupations with a suitable postprocessing function
stack, occupations = granad.evolution(
    stack, time_axis, field_func, dissipation = granad.relaxation(10), transition = transition_function, postprocess=jnp.diag
)

# show energy occupations
plt.plot(time_axis, occupations.real)
plt.show()
