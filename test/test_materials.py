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

def test_haldane():
    delta, t1, t2 = 0.2, -2.66, 1j + 1
    
    haldane =  (
        Material("graphene")
        .lattice_constant(2.46)
        .lattice_basis([
            [1, 0, 0],
            [-0.5, jnp.sqrt(3)/2, 0]
        ])
        .add_orbital_species("pz1", l=1, atom='C')
        .add_orbital_species("pz2", l=1, atom='C')
        .add_orbital(position=(0, 0), tag="sublattice_1", species="pz1")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz2")
        .add_interaction(
            "hamiltonian",
            participants=("pz1", "pz2"),
            parameters= [t1],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz1", "pz1"),            
            # a bit overcomplicated
            parameters=[                
                [0, 0, 0, 0], # onsite                
                # clockwise hoppings
                [-2.46, 0, 0, t2], 
                [2.46, 0, 0, jnp.conj(t2)],
                [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, t2],
                [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, t2],
                [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)]
            ],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz2", "pz2"),
            parameters=[                
                [0, 0, 0, delta], # onsite                
                # clockwise hoppings
                [-2.46, 0, 0, jnp.conj(t2)], 
                [2.46, 0, 0, t2],
                [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, t2],
                [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, t2]
            ],
        )
        .add_interaction(                
            "coulomb",
            participants=("pz1", "pz2"),
            parameters=[8.64],
            expression=ohno_potential(0)
        )
        .add_interaction(
            "coulomb",
            participants=("pz1", "pz1"),
            parameters=[16.522, 5.333],
            expression=ohno_potential(0)
        )
        .add_interaction(
            "coulomb",
            participants=("pz2", "pz2"),
            parameters=[16.522, 5.333],
            expression=ohno_potential(0)
        )
    )

    flake = haldane.cut_flake(Triangle(10, armchair = True))

    assert flake.hamiltonian[0, 2] == flake.hamiltonian[2, 1] == flake.hamiltonian[1, 0] == flake.hamiltonian[9, 10]  == flake.hamiltonian[10, 11] == flake.hamiltonian[11, 9], "circular hoppings don't match"
    assert jnp.all(flake.hamiltonian.conj().T == flake.hamiltonian), "hamiltonian non-hermitian"
    
