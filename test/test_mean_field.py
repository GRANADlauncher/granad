import numpy as np
import pytest

from granad import Orbital, OrbitalList


def diagonal_system(energies, electrons, spin_degeneracy):
    orbitals = [Orbital() for _ in energies]
    system = OrbitalList(orbitals)
    if spin_degeneracy == 1:
        system.set_open_shell()
    system.set_electrons(electrons)
    for orbital, energy in zip(orbitals, energies):
        system.set_hamiltonian_element(orbital, orbital, energy)
    return system


@pytest.mark.parametrize('spin_degeneracy,electrons', [
    (1, 1), (1, 3), (2, 1), (2, 2), (2, 3), (2, 6), (2, 8),
])
def test_mean_field_particle_number_and_filling(spin_degeneracy, electrons):
    system = diagonal_system([-2., -.5, .7, 2.], electrons, spin_degeneracy)
    system.set_mean_field(mix=0., iterations=2, coulomb_strength=0.)
    expected = np.clip(electrons - spin_degeneracy * np.arange(4), 0, spin_degeneracy)
    density = np.asarray(system.stationary_density_matrix)
    np.testing.assert_allclose(density * electrons, np.diag(expected), rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.trace(density), 1., rtol=0, atol=1e-10)
    np.testing.assert_allclose(density, system.initial_density_matrix, rtol=0, atol=1e-10)


@pytest.mark.parametrize('spin_degeneracy', [1, 2])
def test_mean_field_complex_hopping(spin_degeneracy):
    system = diagonal_system([0., 0.], 1, spin_degeneracy)
    system.set_hamiltonian_element(system[0], system[1], 1j)
    system.set_mean_field(mix=0., iterations=2, coulomb_strength=0.)
    expected = np.array([[.5, -.5j], [.5j, .5]])
    density = np.asarray(system.stationary_density_matrix)
    np.testing.assert_allclose(density, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(density, density.conj().T, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.eigvalsh(density), [0., 1.], rtol=0, atol=1e-10)


@pytest.mark.parametrize('spin_degeneracy,electrons,expected', [
    (1, 2, [1., 1., 0., 0.]),
    (2, 3, [2., 1., 0., 0.]),
])
def test_mean_field_keeps_sequential_degenerate_filling(spin_degeneracy, electrons, expected):
    system = diagonal_system([-1., 0., 0., 2.], electrons, spin_degeneracy)
    system.set_mean_field(mix=0., iterations=2, coulomb_strength=0.)
    np.testing.assert_allclose(
        system.stationary_density_matrix * electrons, np.diag(expected), rtol=0, atol=1e-10,
    )


def test_mean_field_excitation_does_not_change_ground_state_filling():
    system = diagonal_system([-1., 1.], 1, 1)
    system.set_excitation(0, 1, 1)
    system.set_mean_field(mix=0., iterations=2, coulomb_strength=0.)
    np.testing.assert_allclose(system.stationary_density_matrix, np.diag([1., 0.]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(system.initial_density_matrix, np.diag([0., 1.]), rtol=0, atol=1e-10)


def test_mean_field_custom_builder_keeps_spin_scaling():
    system = diagonal_system([-1., 1.], 2, 2)

    def fill_upper(vecs, energies):
        return vecs[:, 1:2] @ vecs[:, 1:2].conj().T

    system.set_mean_field(mix=0., iterations=2, coulomb_strength=0., f_build=fill_upper)
    np.testing.assert_allclose(system.stationary_density_matrix, np.diag([0., 1.]), rtol=0, atol=1e-10)
