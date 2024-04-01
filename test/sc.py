import granad
from jax import lax
import jax.numpy as jnp
from granad import _density_matrix

def get_self_consistent(
    stack: granad.Stack, iterations: int = 500, mix: float = 0.05, accuracy: float = 1e-6
) -> granad.Stack:
    """Get a stack with a self-consistent IP Hamiltonian.

    :param stack: a stack object
    :param iterations:
    :param mix:
    :param accuracy:
    :returns: a stack object
    """

    def _to_site_basis(ev, mat):
        return ev @ mat @ ev.conj().T

    def _phi( rho ):
        return stack.coulomb @ jnp.diag(rho - rho_uniform)
        
    def stop( args ):
        return jnp.logical_or( jnp.linalg.norm(args[1] - args[0]) < accuracy, args[2] > iterations )

    def loop( args ):        
        rho, rho_old, idx = args
        ham_new = stack.hamiltonian + _phi( rho ) * mix + _phi( rho_old  ) * (1 - mix)
        
        # diagonalize
        energies, eigenvectors = jnp.linalg.eigh( ham_new )

        # new density matrix
        rho_energy, _ = _density_matrix(
            energies, stack.electrons, 0, 0, stack.beta, stack.eps
        )

        return _to_site_basis(eigenvectors, rho_energy), rho, idx + 1


    system_dim = stack.energies.size
    rho_uniform = jnp.eye(system_dim) / system_dim 
    
    # first induced potential
    rho_old = jnp.zeros_like(stack.hamiltonian)
    rho = _to_site_basis( stack.eigenvectors, stack.rho_0 )

    rho, rho_old, _ = lax.while_loop( stop, loop, (rho, rho_old, 0)  )
    ham_new = h0 + _phi( rho ) * mix + _phi( rho_old  ) * (1 - mix)
    energies, eigenvectors = jnp.linalg.eigh( ham_new )
    rho_stat, homo = granad._density_matrix(
        energies, stack.electrons, 0, 0, stack.beta, stack.eps
    )
    _, homo = granad._density_matrix(
        energies,
        stack.electrons,
        stack.from_state,
        stack.to_state,
        stack.beta,
        stack.eps,
    )

    return stack.replace(
        hamiltonian=ham_new,
        rho_0=rho,
        rho_stat=rho_stat,
        energies=energies,
        eigenvectors=eigenvectors,
        homo=homo,
    )


if __name__ == '__main__':
    # build stack
    sb = granad.StackBuilder()

    # add graphene
    graphene = granad.Lattice(
        shape=granad.Chain(16),
        lattice_type=granad.LatticeType.CHAIN,
        lattice_edge=granad.LatticeEdge.NONE,
        lattice_constant=1.42,
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
    stack = sb.get_stack( from_state = 0, to_state = 2, doping = 1 )


    stack_old = granad.get_self_consistent( stack )
    stack_new = get_self_consistent( stack )
