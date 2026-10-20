"""Small numerical guarantees from the tutorials, without notebook execution."""

import jax.numpy as jnp
import numpy as np
import pytest

from granad import Orbital, OrbitalList, Periodic, Pulse, get_chain
from granad.orbitals import TDResult


@pytest.fixture
def chain():
    return get_chain(hopping=-1., lattice_const=1.).cut_flake(unit_cells=6)


def test_chain_spectrum(chain):
    """j_defining_materials: a finite nearest-neighbor chain has a known spectrum."""
    assert len(chain) == 6
    expected = -2 * np.cos(np.pi * np.arange(1, 7) / 7)
    np.testing.assert_allclose(chain.energies, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(chain.hamiltonian, chain.hamiltonian.conj().T, rtol=0, atol=1e-10)


def test_basis_round_trip_and_expectation(chain):
    """d_basis_handling: complex matrices and expectations are basis independent."""
    hamiltonian = chain.hamiltonian
    np.testing.assert_allclose(chain.hamiltonian_e, np.diag(chain.energies), rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        chain.transform_to_site_basis(chain.transform_to_energy_basis(hamiltonian)),
        hamiltonian, rtol=0, atol=1e-10,
    )
    state = jnp.arange(1, 7) + 1j * jnp.arange(6)
    state = state / jnp.linalg.norm(state)
    densities = jnp.stack([chain.initial_density_matrix, jnp.outer(state, state.conj())])
    energy_densities = chain.transform_to_energy_basis(densities)
    np.testing.assert_allclose(
        chain.transform_to_site_basis(energy_densities), densities, rtol=0, atol=1e-10,
    )
    operator = jnp.diag(jnp.arange(6)) + 1j * (jnp.eye(6, k=1) - jnp.eye(6, k=-1))
    expected = chain.electrons * jnp.einsum('ij,tji->t', operator, densities)
    np.testing.assert_allclose(
        chain.electrons * jnp.einsum('ij,tji->t', chain.transform_to_energy_basis(operator), energy_densities),
        expected, rtol=0, atol=1e-10,
    )
    # The public observable API uses the electron's negative charge.
    np.testing.assert_allclose(
        chain.get_expectation_value(operator=operator, density_matrix=densities, induced=False),
        -expected, rtol=0, atol=1e-10,
    )


@pytest.mark.parametrize('model', ['scalar', 'lindblad'])
def test_exponential_relaxation(model):
    """la_lindblad: a single downward transition relaxes exponentially."""
    lower, upper = Orbital(), Orbital()
    system = OrbitalList([lower, upper])
    system.set_hamiltonian_element(upper, upper, 2.)
    system.set_electrons(1)
    system.set_excitation(0, 1, 1)
    gamma = .7
    rate = gamma if model == 'scalar' else jnp.array([[0., 0.], [gamma, 0.]])
    times = jnp.linspace(0., 2., 21)
    result = system.master_equation(
        end_time=2., grid=times, dt=.01, coulomb_strength=0.,
        relaxation_rate=rate, density_matrix=['full', 'occ_e'],
    )
    assert result is not None
    density, occupations = result.output
    expected = np.exp(-gamma * np.asarray(times))
    np.testing.assert_allclose(occupations[:, 1], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(occupations[:, 0], 1 - expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.trace(density, axis1=-2, axis2=-1), 1, rtol=0, atol=1e-6)
    np.testing.assert_allclose(density, density.conj().swapaxes(-1, -2), rtol=0, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(density) >= -1e-6)


def test_simulation_outputs_and_custom_processing(chain):
    """g_td_sim and l_master: all output paths describe the same density matrices."""
    operators = chain.dipole_operator
    postprocesses = chain.get_postprocesses([operators], ['full', 'occ_x', 'occ_e'])
    postprocesses['element'] = lambda rho, args: rho[:, 0, 1]
    times = jnp.linspace(0., 1., 21)
    result = chain.master_equation(
        end_time=1., grid=times, dt=.01, coulomb_strength=0.,
        illumination=Pulse(amplitudes=[.1, 0., 0.], frequency=2., peak=.5, fwhm=.4),
        postprocesses=postprocesses,
    )
    assert result is not None
    expectation, density, occ_x, occ_e, element = result.output
    assert expectation.shape == (21, 3)
    assert density.shape == (21, 6, 6)
    assert occ_x.shape == occ_e.shape == (21, 6)
    assert element.shape == (21,)
    np.testing.assert_allclose(result.time_axis, times, rtol=0, atol=1e-12)
    np.testing.assert_allclose(occ_x, chain.electrons * jnp.diagonal(density, axis1=-2, axis2=-1), rtol=0, atol=1e-10)
    energy_density = chain.transform_to_energy_basis(density)
    np.testing.assert_allclose(occ_e, chain.electrons * jnp.diagonal(energy_density, axis1=-2, axis2=-1), rtol=0, atol=1e-10)
    expected = chain.electrons * jnp.einsum('cij,tji->tc', operators, chain.stationary_density_matrix - density)
    np.testing.assert_allclose(expectation, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(element, density[:, 0, 1], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.final_density_matrix, density[-1], rtol=0, atol=1e-10)
    assert np.max(np.abs(expectation[:, 0])) > 1e-4


def test_fourier_frequency_amplitude_and_filtering():
    """i_linear_response: output and illumination transforms keep frequency and scale."""
    times = jnp.arange(128) * .125
    duration = 16.
    omega = 2 * np.pi * 5 / duration
    signal = 1.5 * jnp.cos(omega * times)
    result = TDResult(time_axis=times, output=[signal[:, None]], td_illumination=signal[:, None])
    frequencies, field = result.ft_illumination(jnp.inf, -jnp.inf)
    expected = np.zeros((128, 1), dtype=complex)
    expected[[5, -5], 0] = 1.5 * duration / 2
    np.testing.assert_allclose(field, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(frequencies[jnp.array([5, 123])], [omega, -omega], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.ft_output(jnp.inf, -jnp.inf)[0], expected, rtol=0, atol=1e-10)
    filtered_frequencies, filtered = result.ft_illumination(omega + .01, omega - .01)
    np.testing.assert_allclose(filtered_frequencies, [omega], rtol=0, atol=1e-10)
    np.testing.assert_allclose(filtered, [[12.]], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.ft_output(omega + .01, omega - .01)[0], filtered, rtol=0, atol=1e-10)


@pytest.mark.parametrize('hungry', [0, 1, 2])
def test_noninteracting_response_matches_transition_sum(chain, hungry):
    """i_linear_response: zero-Coulomb RPA matches the direct polarizability."""
    energies = np.asarray(chain.energies)
    vectors = np.asarray(chain.eigenvectors)
    occupations = np.asarray(chain.initial_density_matrix_e.diagonal()).real * chain.electrons
    dipole = vectors.conj().T @ np.diag(np.asarray(chain.positions)[:, 0]) @ vectors
    frequencies = jnp.array([0., .5, 1.5, 2.5])
    expected = np.array([
        -sum(
            (occupations[n] - occupations[m]) * abs(dipole[n, m]) ** 2
            / (omega + energies[n] - energies[m] + .2j)
            for n in range(6) for m in range(6)
        )
        for omega in np.asarray(frequencies)
    ])
    actual = chain.get_polarizability_rpa(
        frequencies, polarization=0, coulomb_strength=0., relaxation_rate=.2, hungry=hungry,
    )
    # RPA interpolates energies onto 4096 bins; allow their spacing relative to broadening.
    spacing = 2 * (np.max(np.abs(energies)) + .1) / 4095
    np.testing.assert_allclose(actual, expected, rtol=2 * spacing / .2, atol=1e-10)


def test_periodic_chain_band_and_reciprocal_basis():
    """n_periodic: the one-orbital chain has a cosine band and reciprocal basis."""
    lattice_constant = 1.3
    cell = get_chain(hopping=-1., lattice_const=lattice_constant).cut_flake(unit_cells=1)
    lattice = jnp.array([[lattice_constant, 0., 0.]])
    bulk = Periodic(cell, lattice, 1)
    momenta = jnp.array([-.8, 0., .4, jnp.pi / lattice_constant])
    ks = jnp.stack([momenta, jnp.zeros_like(momenta), jnp.zeros_like(momenta)])
    energies, _ = bulk.get_eigenbasis(ks)
    np.testing.assert_allclose(energies[:, 0], -2 * jnp.cos(momenta * lattice_constant), rtol=0, atol=1e-10)
    np.testing.assert_allclose(lattice @ bulk.reciprocal_basis, 2 * np.pi * np.eye(1), rtol=0, atol=1e-10)


def test_xyz_round_trip(tmp_path):
    """xyz: exporting and importing preserves atom labels and coordinates."""
    original = OrbitalList([
        Orbital(atom_name='C', position=[.1, -.2, .3]),
        Orbital(atom_name='H', position=[1.4, .5, -.6]),
        Orbital(atom_name='C', position=[-.7, .8, .9]),
    ])
    path = tmp_path / 'atoms.xyz'
    original.to_xyz(path)
    restored = OrbitalList.from_xyz(path)
    assert len(restored) == len(original)
    assert restored.atoms.keys() == original.atoms.keys()
    for atom in original.atoms:
        np.testing.assert_allclose(
            np.asarray(restored.atoms[atom], dtype=float),
            np.asarray(original.atoms[atom], dtype=float), rtol=0, atol=1e-10,
        )
