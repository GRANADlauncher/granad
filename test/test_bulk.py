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
