import jax.numpy as jnp
import matplotlib.pyplot as plt

from granad import *

def test_graphene():
    """plots band structure of graphene"""

    # we cut a small flake, equivalent to the bulk unit cell
    graphene = get_graphene()
    flake = graphene.cut_flake(grid_range = [(0,2), (0,1)])
    del flake[-1]
    del flake[1]

    # define lattice vectors
    lattice_vectors = 2.46 * jnp.array([
        [1, 0, 0],
        [-0.5, jnp.sqrt(3)/2, 0]
    ])
    bulk = Periodic(flake, lattice_vectors, 1)

    # high-symmetry path
    B, A = bulk.reciprocal_basis.T
    ks = jnp.concatenate(
        ( jnp.expand_dims(jnp.linspace(0,-1/3,100),1)*A + jnp.expand_dims(jnp.linspace(0,2/3,100),1)*B, # G -> K
          jnp.expand_dims(jnp.linspace(-1/3,0,100),1)*A + jnp.expand_dims(jnp.linspace(2/3,0.5,100),1)*B, # K -> M
          jnp.expand_dims(jnp.linspace(0.5,0,30),1)*B # M -> G          
        )).T

    # get eigenvalues along path
    vals, vecs = bulk.get_eigenbasis(ks)
    plt.plot(jnp.linspace(0, 1, ks.shape[1]), vals)
    plt.show()
    
# def test_goldene():
sk_file = "Au-Au.skf"
goldene = (Material("goldene")
           .lattice_constant(2.62)  # use 2.62 Å (exp) or 2.735 Å (DFT relaxed)
           .lattice_basis([
               [1, 0, 0],
               [-0.5, jnp.sqrt(3)/2, 0]
           ])
           .add_atom(atom = "Au", position=(0, 0), orbitals = ["s"])  # one Au atom per primitive cell (P6/mmm)
           .add_slater_koster_interaction("Au", "Au", sk_file, num_neighbors = 20)
           )

flake = goldene.cut_flake(Rectangle(0.5, 0.5), minimum_neighbor_number = 0)
lattice_vectors = 2.62 * jnp.array([
    [1, 0, 0],
    [-0.5, jnp.sqrt(3)/2, 0]
])
bulk = Periodic(flake, lattice_vectors, 4)

B, A = bulk.reciprocal_basis.T
ks = jnp.concatenate(
    ( jnp.expand_dims(jnp.linspace(0,-1/3,100),1)*A + jnp.expand_dims(jnp.linspace(0,2/3,100),1)*B, # G -> K
      jnp.expand_dims(jnp.linspace(-1/3,0,100),1)*A + jnp.expand_dims(jnp.linspace(2/3,0.5,100),1)*B, # K -> M
      jnp.expand_dims(jnp.linspace(0.5,0,100),1)*B # M -> G          
    )).T

vals, vecs = bulk.get_eigenbasis(ks)
plt.plot(jnp.linspace(0, 1, ks.shape[1]), vals)
plt.show()


mu = bulk.get_mu(vals, 0.5, 1)

ham = bulk.get_hamiltonian(ks)
plt.plot(jnp.linspace(0, 1, ks.shape[1]), vals)
plt.show()


metal = (
Material("Metal")
.lattice_constant(1.0)
.lattice_basis([
    [1, 0, 0],
    [0, 1, 0],
])
.add_orbital_species("up", s = -1)
.add_orbital(position=(0, 0), species = "up",  tag = "up")
.add_interaction(
    "hamiltonian",
    participants=("up", "up"),
    parameters=[0.0, 1.],
)
)

flake = metal.cut_flake(Rectangle(0.5, 0.5), minimum_neighbor_number = 0)

# band structure 1d metal
lattice_vectors =  jnp.array([
    [1, 0, 0],
])
bulk = Periodic(flake, lattice_vectors, 2)
# check we get a nice cosine
grid = jnp.linspace(0, 1, 100)
ks = bulk.reciprocal_basis * grid
ham = bulk.get_hamiltonian(ks)
vals, vecs = bulk.get_eigenbasis(ks)
plt.plot(grid, vals)
plt.show()

v = bulk.get_velocity_operator(ks, vecs)
mu = bulk.get_mu(vals)
omegas = jnp.linspace(0, 2, 100)
c_intra = bulk.get_ip_conductivity_intra(v, vals, omegas, 0, 10, relaxation_rate = 0.1)
cap = c_intra.imag
plt.plot(omegas, cap.diagonal(axis1=1, axis2=2)[:, 0])
plt.show()


# band structure graphene
graphene = get_graphene()
flake = graphene.cut_flake(grid_range = [(0,2), (0,1)])
del flake[-1]
del flake[1]

lattice_vectors = 2.46 * jnp.array([
    [1, 0, 0],
    [-0.5, jnp.sqrt(3)/2, 0]
])
bulk = Periodic(flake, lattice_vectors, 1)
B, A = bulk.reciprocal_basis.T
ks = jnp.concatenate(
    ( jnp.expand_dims(jnp.linspace(0,-1/3,100),1)*A + jnp.expand_dims(jnp.linspace(0,2/3,100),1)*B, # G -> K
      jnp.expand_dims(jnp.linspace(-1/3,0,100),1)*A + jnp.expand_dims(jnp.linspace(2/3,0.5,100),1)*B, # K -> M
      jnp.expand_dims(jnp.linspace(0.5,0,30),1)*B # M -> G          
    )).T

ks = bulk.get_kgrid_monkhorst_pack([200, 200])
energies, eigenvectors = bulk.get_eigenbasis(ks)    
# plt.plot(jnp.linspace(0, 1, ks.shape[1]), vals)
# plt.show()

ham = bulk.get_hamiltonian(ks)
v = bulk.get_velocity_operator(ks, eigenvectors)
mu = bulk.get_mu(energies)
omegas = 2.66 * jnp.linspace(0, 2, 100)
c_inter = bulk.get_ip_conductivity_inter(v, energies, omegas, mu, jnp.inf, relaxation_rate = 0.1)
cap = c_intra.imag / c_intra.imag[0]
plt.plot(omegas, cap.diagonal(axis1=1, axis2=2)[:, 0])
plt.plot(omegas, cap.diagonal(axis1=1, axis2=2)[:, 1])
plt.show()
