import jax.numpy as jnp
from granad import *
from granad._plotting import *

def get_hubbard_flake(U):
    """Constructs a square lattice cut from the 2D hubbard model at half-filling

    Args:
       U : onsite coulomb repulsion for opposite spins    
    """

    t = -1. # nearest-neighbor hopping

    hubbard = (
        Material("Hubbard")
        .lattice_constant(1.0)
        .lattice_basis([
            [1, 0, 0],
            [0, 1, 0],
        ])
        .add_orbital_species("up", s = -1)
        .add_orbital_species("down", s = 1)
        .add_orbital(position=(0, 0), species = "up",  tag = "up")
        .add_orbital(position=(0, 0), species = "down",  tag = "down")   
        .add_interaction(
            "hamiltonian",
            participants=("up", "up"),
            parameters=[0.0, t],
        )
        .add_interaction(
            "hamiltonian",
            participants=("down", "down"),
            parameters=[0.0, t],
        )
        .add_interaction(
            "coulomb",
            participants=("up", "down"),
            parameters=[U]
            )
    )

    flake = hubbard.cut_flake( Rectangle(10, 4) )
    flake.set_open_shell()
    flake.set_electrons(len(flake)//2)
    
    return flake

# v = jax.random.randint(jax.random.key(1), (4,4), 0, 4)  * (jnp.arange(4) < 2)[None, :]
# print(v @ v.T)
# w = jax.random.randint(jax.random.key(1), (4,4), 0, 4)
# print(w[:, :2] @ w[:, :2].T)

U = -10
flake = get_hubbard_flake(U)
N = len(flake)

# decoupling  like U<cd cd> c c + U <c c> cd cd
def f_mean_field(rho, ham):
    # BdG is [[H, U], [U, H]], where H is s.p. and U is diagonal with U
    U_mat = jnp.diag(U * jnp.diagonal(rho, offset = -N))    
    return jnp.block([[ham, U_mat], [U_mat, ham]])

# constructing density matrix
def f_build(vecs, energies):
    v = vecs * (energies < 0)[None, :]
    return v @ v.T

rho_0 = jnp.ones((2*N, 2*N)).astype(complex)
flake.set_mean_field(f_mean_field = f_mean_field, f_build = f_build, rho_0 = rho_0)

# anomalous entries of the ground state density matrix
print(flake.stationary_density_matrix_e[:N, N:])
