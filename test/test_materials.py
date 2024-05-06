from granad import *
from granad._numerics import *
import jax.numpy as jnp

def test_2d():
    for name in ["graphene", "MoS2", "hBN"]:
        orbs = MaterialCatalog.get(name).cut_flake( Triangle(10) )
        assert len( orbs ) > 0

def test_1d():
    for name in ["metal_1d", "ssh"]:
        orbs = MaterialCatalog.get(name).cut_flake( unit_cells = 2 )
        assert len( orbs ) > 0

def test_custom():

    t = 1. # nearest-neighbor hopping
    U = 0.1 # onsite coulomb repulsion
    hubbard = (
        Material("Hubbard")
        .lattice_constant(1.0)
        .lattice_basis([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        .add_orbital_species("up", s = -1)
        .add_orbital_species("down", s = 1)
        .add_orbital(position=(0, 0, 0), species = "up")
        .add_orbital(position=(0, 0, 0), species = "down")   
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

    orbs = hubbard.cut_flake( [(0,1), (0,1), (0,1)] )

    assert len(orbs) > 0



