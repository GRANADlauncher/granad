import jax
import jax.numpy as jnp
from jax import Array

import granad

import matplotlib.pyplot as plt

from typing import Callable


## these functions would be integrated in GRANAD 
def peierls_phases(positions,
                   space : Callable,
                   time : Callable) -> Callable:
    """This function returns a closure. The closure takes a single point in time and returns 
    a :math:`N \times N` array containing Peierls phases.

    :param positions: positions between which to evaluate the Peierls phases. Most of the time, these will be orbital positions.
    :param space: space-dependent part of the vector potential. It should map a :math:`3 \times N` array to a :math:`3 \times N` array, where the first dimension corresponds to x,y,z and the second dimension corresponds to a sample point.
    :param time: time-dependent part of the vector potential. Should be a function of a single argument returning either a hermitian matrix or a single float.
    :returns: closure returning time-dependent matrix peierls phases matrix
    """

    def integration(i, j):                
        # get start and end point
        start, end = positions[i][:,None], positions[j][:,None]

        # discretize the path
        x = jnp.linspace(0, 1, 10000)

        # a straight line connecting the two points
        path = (end-start) * x + start
        
        # projected values of the vector potential
        y = space( path ).T @ (end-start)

        # numerical integration
        return jnp.trapz(y[:,0], x)

    # vectorize integration function 
    idxs = jnp.arange( positions.shape[0] )
    integration_vectorized = jax.vmap(jax.vmap(integration, (0, None), 0), (None, 0), 0)

    # static part of the peierls matrix multiplying the hamiltonian
    # it is given by static_phases_{ij} = exp( i \int_{r_i}^{r_j} dr A(r) )
    # the integral is taken along the straight line connecting orbitals i and j
    static_phases = jnp.exp( 1j * integration_vectorized( idxs, idxs ) )

    # dynamic part is simply given by multiplying the time dependence with the static phases
    return lambda t : static_phases * time( t )

def evolution_peierls(
        stack: granad.Stack,
        time: Array,
        peierls_phases : Callable,
        dissipation: granad.DissipationFunc = None,
        coulomb_strength: float = 1.0,
        postprocess: Callable[[Array], Array] = None,
) -> tuple[granad.Stack, Array]:
    """Propagate a stack forward in time.

    :param stack: stack object
    :param time: time axis
    :param peierls_phases: a function returning the peierls phases at a point in time as a :math:`N \times N` Array
    :param dissipation: dissipation function
    :param coulomb_strength: scaling factor applied to coulomb interaction strength
    :param postprocess: a function applied to the density matrix after each time step
    :returns: (stack with rho_0 set to the current state, array containing all results from postprocess at all timesteps)
    """

    def integrate(rho, time):
        delta_rho = rho - rho_stat
        charge = -jnp.diag(delta_rho) * stack.electrons
        p_ind = coulomb @ charge
        h_total = peierls_phases(time) * stack.hamiltonian + jnp.diag(p_ind)
        if dissipation:
            return (
                rho
                - 1j * dt * (h_total @ rho - rho @ h_total)
                + dt * dissipation(rho, rho_stat),
                postprocess(rho) if postprocess else rho,
            )
        else:
            return (
                rho - 1j * dt * (h_total @ rho - rho @ h_total),
                postprocess(rho) if postprocess else rho,
            )

    dt = time[1] - time[0]
    coulomb = stack.coulomb * coulomb_strength
    rho_stat = stack.eigenvectors @ stack.rho_stat @ stack.eigenvectors.conj().T
    rho, rhos = jax.lax.scan(
        integrate, stack.eigenvectors @ stack.rho_0 @ stack.eigenvectors.conj().T, time
    )

    return (
        stack.replace(rho_0=stack.eigenvectors.conj().T @ rho @ stack.eigenvectors),
        rhos,
    )

## define spatial part of vector potential
def vector_potential( r ):
    return jnp.ones_like(r) #r / jnp.linalg.norm(r, axis = 0)

# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Triangle(4.1),
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_type=granad.LatticeType.HONEYCOMB,
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
stack_graphene = sb.get_stack()

# peierls phases are a function of time returning a matrix
phases = peierls_phases(stack_graphene.positions,
                        space = vector_potential,
                        time = jnp.sin # harmonic time dependence of potential
                        )

time_axis = jnp.linspace(0, 4, 10**5)

stack_graphene_new, occs = evolution_peierls( stack_graphene,
                                              time_axis,
                                              phases,
                                              dissipation = granad.relaxation(0.1),
                                              postprocess = jnp.diag
                                             )

skip = 10
dip_moment = granad.induced_dipole_moment( stack_graphene, occs )
labels = ['x', 'y']
norm = dip_moment.max()
for i, d in enumerate(dip_moment.T[:2] / norm):
    label = labels[i]
    plt.plot( time_axis[::skip], d[::skip], label = rf'$p_{label}$' )
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$P_{ind}$ (normalized)')
plt.savefig('example_dynamic.pdf')
plt.close()
