import jax.numpy as jnp
from granad import *
from granad._plotting import *

def peierls_coupling(B, t, r1, r2):
    plus = r1 + r2
    minus = r1 - r2
    return complex(t * jnp.exp(1j * B * minus[0] * plus[1]))

def get_graphene_b_field(B, t, shape):
    """obtain tb model for graphene in magnetic field from peierls substitution

    Args:
        B : B-field strength
        t : nn hopping
    
    """
    graphene_peierls = (
        Material("graphene_peierls")
        .lattice_constant(2.46)
        .lattice_basis([
            [1, 0, 0],
            [-0.5, jnp.sqrt(3)/2, 0]
        ])
        .add_orbital_species("pz", atom='C')
        .add_orbital(position=(0, 0), tag="sublattice_1", species="pz")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz")
        .add_interaction(
            "hamiltonian",
            participants=("pz", "pz"),
            parameters = [0],
        )
        .add_interaction(
            "coulomb",
            participants=("pz", "pz"),
            expression = ohno_potential(1.42)
        )
    )
    
    flake = graphene_peierls.cut_flake(shape)
    distances = jnp.round(jnp.linalg.norm(flake.positions - flake.positions[:, None], axis = -1), 4)
    nn = 1.5

    # this is slightly awkward: our coupling only depends on the distance vector, but peierls substitution in landau
    # gauge A = (0, Bx, 0) leads to peierls phase of \int_r1^r2 A dl = (y2 - y1) (x2 + x1) / 2
    # we patch this manually here, but this is very, very slow
    for i, orb1 in enumerate(flake):
        for j in range(i+1):
            if 0 < distances[i, j] <= nn:
                orb2 = flake[j]
                flake.set_hamiltonian_element(orb1, orb2, peierls_coupling(B, t, orb1.position, orb2.position))
    return flake

def make_plots():
    shape = Hexagon(30)
    flake = get_graphene().cut_flake(shape)
    flake.show_energies()
    flake.show_2d()

    # magnetic field of 1T / scaled by e * (1e-4 eV)**2
    flake = get_graphene_b_field(10 * 200 * 0.3 * 1e-8, -2.7, shape)
    flake.show_2d()
    flake.show_energies()
    omegas = jnp.linspace(0, 10)
    
    # show quantization
    dos = jax.vmap(lambda w : flake.get_dos(w, broadening = 0.05))(omegas)
    plt.plot(omegas, dos)
    plt.show()
