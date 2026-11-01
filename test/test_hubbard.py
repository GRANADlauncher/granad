import jax.numpy as jnp
import numpy as np
import pytest

from granad import Material

def get_hubbard(U):
    t = 1. # nearest-neighbor hopping
    # U => onsite coulomb repulsion for opposite spins

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

    return hubbard

@pytest.mark.parametrize('U', [0., 1., 4.])
@pytest.mark.parametrize('sign', [-1, 1])
def test_hubbard(U, sign):
    """The half-filled dimer has an analytical collinear UHF solution."""
    flake = get_hubbard(U).cut_flake(grid_range=[(0, 2), (0, 1)])
    flake.set_open_shell()
    flake.set_electrons(2)
    assert len(flake) == 4
    bare = np.asarray(flake.hamiltonian)
    coulomb = np.asarray(flake.coulomb)
    spins = np.array([orb.spin for orb in flake])
    stagger = np.where(np.asarray(flake.positions)[:, 0] == 0, 1., -1.)
    seed = jnp.diag(.5 + sign * .1 * spins * stagger).astype(complex)
    flake.set_mean_field(rho_0=seed, accuracy=1e-10)

    density = np.asarray(flake.stationary_density_matrix) * flake.electrons
    occupations = density.diagonal().real
    magnetization = occupations[spins == -1] - occupations[spins == 1]
    expected_m = np.sqrt(1 - (2 / U)**2) if U > 2 else 0.
    np.testing.assert_allclose(magnetization, -sign * expected_m * np.array([1., -1.]), rtol=0, atol=1e-8)
    energy = np.trace(bare @ density).real + .5 * occupations @ coulomb.real @ occupations
    expected_energy = -2 / U if U > 2 else -2 + U / 2
    np.testing.assert_allclose(energy, expected_energy, rtol=0, atol=1e-8)
    if U > 2:
        assert energy < -2 + U / 2
    np.testing.assert_allclose(np.trace(density), 2., rtol=0, atol=1e-10)
    np.testing.assert_allclose(density, density.conj().T, rtol=0, atol=1e-10)
    np.testing.assert_allclose(flake.initial_density_matrix, flake.stationary_density_matrix, rtol=0, atol=1e-10)
    np.testing.assert_allclose(flake.hamiltonian, bare + np.diag(coulomb @ occupations), rtol=0, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh(density)
    assert np.all(eigenvalues >= -1e-10) and np.all(eigenvalues <= 1 + 1e-10)
    

