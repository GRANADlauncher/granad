import granad

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import lax

def sigma( i, j, N ):
    vec1 = np.array([1 + 0j if l == i else 0.0 for l in range(N) ])
    vec2 = np.array([1 + 0j if l == j else 0.0 for l in range(N) ])
    return vec1[:, None] * vec2

def evolution(
    stack,
    time,
    dissipation,
    postprocess
):
    """Propagate a stack forward in time.

    :param stack: stack object
    :param time: time axis
    :param field: electric field function
    :param dissipation: dissipation function
    :param coulomb_strength: scaling factor applied to coulomb interaction strength
    :param postprocess: a function applied to the density matrix after each time step
    :returns: (stack with rho_0 set to the current state, array containing all results from postprocess at all timesteps)
    """

    def integrate(rho, time):
        delta_rho = rho - rho_stat
        h_total = stack.hamiltonian
        return (
            rho
            - 1j * dt * (h_total @ rho - rho @ h_total)
            + dt * dissipation(rho, rho_stat),
            postprocess(rho) if postprocess else rho,
        )

    dt = time[1] - time[0]
    coulomb = stack.coulomb * 1
    rho_stat = stack.eigenvectors @ stack.rho_stat @ stack.eigenvectors.conj().T
    # rho, rhos = jax.lax.scan(
    #     integrate, stack.eigenvectors @ stack.rho_0 @ stack.eigenvectors.conj().T, time
    # )
    rho_new = stack.eigenvectors @ stack.rho_0 @ stack.eigenvectors.conj().T
    rhos = []
    for i, t in enumerate(time):
        rho_new, rho_out = integrate( rho_new, time[i] )
        rhos.append( rho_out )
        
    return (
        stack.replace(rho_0=stack.eigenvectors.conj().T @ rho_new @ stack.eigenvectors),
        np.array(rhos),
    )


def _default_functional(rho_element):
    return lax.cond( jnp.logical_or(jnp.abs(rho_element.real) >= 2.0, rho_element.real < 0.0), lambda x : 0.0 + 0j, lambda x : 1 + 0j, rho_element )           


def lindblad(stack, gamma, saturation = _default_functional ):
    """Function for modelling dissipation according to the saturated lindblad equation. TODO: ref paper

    :param stack: object representing the state of the system
    :param gamma: symmetric (or lower triangular) NxN matrix. The element gamma[i,j] corresponds to the transition rate from state i to state j
    :param saturation: a saturation functional to apply, defaults to a sharp turn-off
    :returns: JIT-compiled closure that is needed for computing the dissipative part of the lindblad equation
    """
        
    commutator_diag = jnp.diag( gamma )
    gamma_matrix = gamma.astype(complex)
    saturation_vmapped = jax.vmap(saturation, 0, 0)

    def inner(r, rs):

        # convert rho to energy basis
        r = stack.eigenvectors.conj().T @ (r) @ stack.eigenvectors
        # r = stack.rho_0
        
        # extract occupations
        diag = jnp.diag(r) * stack.electrons
        #print(diag, saturation_vmapped( diag ))

        # apply the saturation functional
        gamma = gamma_matrix# * saturation_vmapped( diag )[:, None]
        a = np.diag( gamma.T @  np.diag(r) )
        mat = np.diag(np.sum( gamma, axis =  1))
        b = -1/2 * (mat @ r + r @ mat)
        val = a + b

        # N = 2
        # a_term = np.zeros_like( r )
        # b_term = np.zeros_like( r )
        # gamma = gamma_matrix #* saturation_vmapped( diag )[:, None]
        # for i in range( N ):
        #     for j in range( N ):
        #         a_term +=  gamma[i,j] * sigma(j, i, N) @ r @ sigma(i, j, N)
        #         mat = sigma(i, j, N) @ sigma(j, i, N)
        #         b_term +=  -1/2 * gamma[i,j] * ( mat @ r + r @ mat  )
        # val = a_term + b_term
        
        
        # # co += commutator_diag
        # ac = -0.5*jnp.diag(jnp.sum(gamma, axis=1))
        # # import pdb; pdb.set_trace()

        # val = jnp.diag(co @ jnp.diag(r)) + ac @ r + r @ ac
        # print( val[-1, -1], val[-2, -2] )
        # print( np.trace(val), np.diag(val) )

        return (
            stack.eigenvectors
            @ (
                val
            )
            @ stack.eigenvectors.conj().T
        )
    
    return inner

sb = granad.StackBuilder()

spot = granad.Spot(position=[0.0, 0.0, 0.0])
sb.add("A", spot)
sb.add("B", spot, occupation=0)

# onsite hopping
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_hopping(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=1))

# onsite coulomb
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=1))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=1))

stack = sb.get_stack()
granad.show_energies(stack)

# create the stack object
stack = sb.get_stack( from_state = 0, to_state = 1)
# granad.show_energies(stack)

amplitudes = [0.0, 0, 0]
frequency = 1
peak = 2
fwhm = 0.5

# choose an x-polarized electric field propagating in z-direction
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions[0, :], peak, fwhm
)

# propagate in time
gamma = 0.1 + 0j
time_axis = jnp.linspace(0, 20 * 1/gamma, 400)
time_axis = jnp.linspace(0, 1/gamma * 2, 100)

# allow transfer from higher energies to lower energies only if the
# two energy levels are not degenerate
diff = stack.energies[:,None] - stack.energies
gamma_matrix = gamma * jnp.logical_and( diff < 0, jnp.abs(diff) > stack.eps, )
gamma_matrix = jnp.array( [ [0., 0.1], [0.0, 0.] ] ).T

relaxation_function = lindblad( stack, gamma_matrix )
# relaxation_function = granad.relaxation( gamma )
new_stack, occupations = evolution(stack, time_axis, relaxation_function, postprocess = lambda x : jnp.diag( stack.eigenvectors.conj().T @ x @ stack.eigenvectors) )

# plot occupations as function of time
skip = 1
for i, occ in enumerate(occupations[::skip].T):
    plt.plot( time_axis[::skip], stack.electrons * occ.real, label = round(stack.energies[i],2) )
plt.legend()
plt.show()
