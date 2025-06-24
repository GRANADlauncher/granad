import jax.numpy as jnp
from granad import *
from granad._plotting import *

# define spinful graphene
graphene_spinful = (
    Material("graphene_spinful")
    .lattice_constant(2.46)
    .lattice_basis([
        [1, 0, 0],
        [-0.5, jnp.sqrt(3)/2, 0]
    ])
    .add_orbital_species("pz+", atom='C')
    .add_orbital_species("pz-", atom='C')
    .add_orbital(position=(0, 0), tag="sublattice_1+", species="pz+")
    .add_orbital(position=(-1/3, -2/3), tag="sublattice_2+", species="pz+")
    .add_orbital(position=(0, 0), tag="sublattice_1-", species="pz-")
    .add_orbital(position=(-1/3, -2/3), tag="sublattice_2-", species="pz-")
    .add_interaction(
        "hamiltonian",
        participants=("pz+", "pz+"),
        parameters=[0.0, 2.55],
    )
    .add_interaction(
        "hamiltonian",
        participants=("pz-", "pz-"),
        parameters=[0.0, 2.55],
    )
    .add_interaction(
        "coulomb",
        participants=("pz+", "pz-"),
        parameters=[4],
    )
)

shape = Triangle(24, armchair = False)
# shape = Rhomboid(30, 30, armchair = False)
flake = graphene_spinful.cut_flake(shape) 
flake.set_open_shell()
flake.set_electrons(len(flake)//2)
flake.show_2d()

rho_0 = jnp.ones(len(flake)) 
rho_0 = jnp.diag(rho_0).astype(complex) 
flake.set_mean_field(mix = 0.4, iterations = 200, rho_0 = rho_0)
flake.show_energies(e_min = -1., e_max = 4)

occs = jnp.diag(flake.initial_density_matrix) * flake.electrons
spin_density = occs[:len(flake)//2] - occs[len(flake)//2:]
show_2d(flake[:len(flake)//2], display = spin_density)


sublattice_diff = len(flake.filter_orbs("sublattice_1+", Orbital)) -  len(flake.filter_orbs("sublattice_2+", Orbital))
total_spin = spin_density.sum()
print(sublattice_diff, total_spin)
