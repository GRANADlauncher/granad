import jax.numpy as jnp
import numpy as np
import pytest

from granad import Orbital, OrbitalList
from granad._numerics import _scf_loop


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
    np.testing.assert_allclose(system.initial_density_matrix, system.stationary_density_matrix, rtol=0, atol=1e-10)


@pytest.mark.parametrize('thermal', [False, True])
def test_scf_driver_preserves_coherences_and_custom_filling(thermal):
    bare = jnp.diag(jnp.array([-2., -1., 1., 2.])).astype(complex)
    bare = bare.at[0, 3].set(.3).at[3, 0].set(.3)
    bare = bare.at[1, 2].set(.4j).at[2, 1].set(-.4j)

    def potential(density):
        return bare + .2 * (density - .5 * jnp.eye(4))

    def build(vectors, energies):
        occupation = 1 / (1 + jnp.exp(energies - .3)) if thermal else (energies < 0)
        return (vectors * occupation) @ vectors.conj().T

    density, hamiltonian, energies, vectors = _scf_loop(
        potential, build, .5 * jnp.eye(4, dtype=complex), mix=.3, accuracy=1e-11, iterations=100,
    )
    np.testing.assert_allclose(hamiltonian, potential(density), rtol=0, atol=1e-11)
    np.testing.assert_allclose(hamiltonian @ vectors, vectors * energies, rtol=0, atol=1e-11)
    np.testing.assert_allclose(density, build(vectors, energies), rtol=0, atol=1e-11)
    assert abs(density[0, 3]) > .01 and abs(density[1, 2]) > .01
    if thermal:
        assert not np.isclose(np.trace(density), 2.)
    else:
        np.testing.assert_allclose(density @ density, density, rtol=0, atol=1e-11)


def hubbard_dimer(interleaved=False, hopping=-1.):
    labels = [(spin, site) for spin in (-1, 1) for site in (0, 1)]
    if interleaved:
        labels.sort(key=lambda label: label[1])
    system = OrbitalList([Orbital(spin=spin, position=[site, 0., 0.]) for spin, site in labels])
    for spin in (-1, 1):
        orbitals = [orb for orb in system if orb.spin == spin]
        system.set_hamiltonian_element(*orbitals, hopping)
    for site in (0, 1):
        orbitals = [orb for orb in system if orb.position[0] == site]
        system.set_coulomb_element(*orbitals, 4.)
    system.set_open_shell()
    system.set_electrons(2)
    seed = jnp.diag(jnp.array([.5 + .1 * spin * (1 - 2 * site) for spin, site in labels])).astype(complex)
    return system, seed


def dense_reference(hamiltonian, coulomb, electrons, seed):
    """Independent full-density SCF reference, without the library's iteration code."""
    density = np.asarray(seed)
    for _ in range(1000):
        fock = hamiltonian + np.diag(coulomb @ density.diagonal())
        energies, vectors = np.linalg.eigh(fock)
        target = vectors[:, :electrons] @ vectors[:, :electrons].conj().T
        if np.linalg.norm(target - density) < 1e-11:
            return fock, energies, target
        density = .3 * density + .7 * target
    raise AssertionError('Dense reference failed to converge')


@pytest.mark.parametrize('interleaved', [False, True])
@pytest.mark.parametrize('hopping', [-1., 1j])
def test_spin_blocks_match_dense_reference(interleaved, hopping, monkeypatch):
    system, seed = hubbard_dimer(interleaved, hopping)
    bare, coulomb = np.asarray(system.hamiltonian), np.asarray(system.coulomb)
    fock, energies, density = dense_reference(bare, coulomb, 2, seed)
    shapes = []
    original = jnp.linalg.eigh

    def record_shape(matrix, *args, **kwargs):
        shapes.append(matrix.shape)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(jnp.linalg, 'eigh', record_shape)
    system.set_mean_field(rho_0=seed, accuracy=1e-10)
    np.testing.assert_allclose(system.stationary_density_matrix * 2, density, rtol=0, atol=1e-8)
    np.testing.assert_allclose(system.energies, energies, rtol=0, atol=1e-8)
    np.testing.assert_allclose(system.hamiltonian, fock, rtol=0, atol=1e-8)
    assert shapes and set(shapes) == {(2, 2, 2)}
    np.testing.assert_allclose(system.eigenvectors.conj().T @ system.eigenvectors, np.eye(4), rtol=0, atol=1e-10)
    np.testing.assert_allclose(system.initial_density_matrix, system.stationary_density_matrix, rtol=0, atol=1e-10)


@pytest.mark.parametrize('fallback', ['spin_flip', 'unlabelled', 'unequal_sectors', 'callback'])
def test_dense_fallback_keeps_all_couplings(fallback, monkeypatch):
    system, seed = hubbard_dimer(hopping=1j)
    if fallback == 'spin_flip':
        system.set_hamiltonian_element(system[0], system[2], .2)
    elif fallback == 'unlabelled':
        # Equal labels cannot identify separate spin sectors.
        from dataclasses import replace
        system._list = [replace(orb, spin=None) for orb in system]
    elif fallback == 'unequal_sectors':
        del system[-1]
        system.set_electrons(2)
        seed = jnp.diag(jnp.array([.6, .7, .7])).astype(complex)
    bare, coulomb = np.asarray(system.hamiltonian), np.asarray(system.coulomb)
    _, _, expected = dense_reference(bare, coulomb, 2, seed)
    shapes = []
    original = jnp.linalg.eigh

    def record_shape(matrix, *args, **kwargs):
        shapes.append(matrix.shape)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(jnp.linalg, 'eigh', record_shape)
    kwargs = {}
    if fallback == 'callback':
        kwargs['f_mean_field'] = lambda rho, h: h + jnp.diag(jnp.asarray(coulomb) @ rho.diagonal())
    system.set_mean_field(rho_0=seed, accuracy=1e-10, **kwargs)
    np.testing.assert_allclose(system.stationary_density_matrix * 2, expected, rtol=0, atol=1e-8)
    expected_shape = (len(system), len(system))
    assert shapes and set(shapes) == {expected_shape}


@pytest.mark.parametrize('override', ['potential', 'filling', 'both'])
def test_mean_field_defaults_can_be_overridden_independently(override):
    system, seed = hubbard_dimer(hopping=1j)
    bare, coulomb = np.asarray(system.hamiltonian), np.asarray(system.coulomb)
    strength = .8
    fock, energies, density = dense_reference(bare, strength * coulomb, 2, seed)
    kwargs = {}
    if override in ('potential', 'both'):
        kwargs['f_mean_field'] = lambda rho, h: h + jnp.diag(strength * jnp.asarray(coulomb) @ rho.diagonal())
    if override in ('filling', 'both'):
        kwargs['f_build'] = lambda vectors, energies: vectors[:, :2] @ vectors[:, :2].conj().T
    system.set_mean_field(rho_0=seed, accuracy=1e-10, coulomb_strength=strength, **kwargs)
    np.testing.assert_allclose(system.hamiltonian, fock, rtol=0, atol=1e-8)
    np.testing.assert_allclose(system.energies, energies, rtol=0, atol=1e-8)
    np.testing.assert_allclose(system.stationary_density_matrix * 2, density, rtol=0, atol=1e-8)


def test_global_filling_allows_unequal_spin_populations():
    system, _ = hubbard_dimer()
    for orbital in system:
        system.set_hamiltonian_element(orbital, orbital, 3. * orbital.spin)
    system.set_mean_field(coulomb_strength=0.)
    occupations = np.diag(system.stationary_density_matrix).real * 2
    np.testing.assert_allclose([occupations[:2].sum(), occupations[2:].sum()], [2., 0.], rtol=0, atol=1e-10)


def test_magnetic_excitation_preserves_converged_reference():
    system, seed = hubbard_dimer()
    system.set_mean_field(rho_0=seed, accuracy=1e-10)
    stationary = np.asarray(system.stationary_density_matrix)
    system.set_excitation(0, 2, 1)
    np.testing.assert_allclose(system.stationary_density_matrix, stationary, rtol=0, atol=1e-10)
    np.testing.assert_allclose(system.initial_density_matrix_e.diagonal(), [.25] * 4, rtol=0, atol=1e-10)


@pytest.mark.parametrize('kwargs', [
    {'mix': 1.}, {'mix': -.1}, {'mix': np.nan},
    {'accuracy': 0.}, {'accuracy': np.inf},
    {'iterations': 0}, {'iterations': 1.5}, {'coulomb_strength': np.nan},
])
def test_invalid_iteration_controls(kwargs):
    system, _ = hubbard_dimer()
    system.set_mean_field(**kwargs)
    with pytest.raises(ValueError, match='Mean-field'):
        system.energies


@pytest.mark.parametrize('invalid', ['shape', 'nonfinite', 'nonhermitian', 'trace', 'lower_bound', 'upper_bound', 'spin_coherence'])
def test_invalid_density_seed(invalid):
    system, seed = hubbard_dimer()
    seed = np.asarray(seed).copy()
    if invalid == 'shape':
        seed = seed[:2, :2]
    elif invalid == 'nonfinite':
        seed[0, 0] = np.nan
    elif invalid == 'nonhermitian':
        seed[0, 1] = .1j
    elif invalid == 'trace':
        seed *= .5
    elif invalid == 'lower_bound':
        seed = np.diag([-.1, .7, .7, .7])
    elif invalid == 'upper_bound':
        seed = np.diag([1.1, .3, .3, .3])
    elif invalid == 'spin_coherence':
        seed[0, 2] = seed[2, 0] = .1
    system.set_mean_field(rho_0=seed)
    with pytest.raises(ValueError, match='rho_0'):
        system.energies


@pytest.mark.parametrize('callback', [False, True])
def test_large_mixing_does_not_hide_nonconvergence(callback):
    system, seed = hubbard_dimer()
    kwargs = {}
    if callback:
        coulomb = system.coulomb
        kwargs['f_mean_field'] = lambda rho, h: h + jnp.diag(coulomb @ rho.diagonal())
    system.set_mean_field(rho_0=seed, mix=1 - 1e-9, iterations=1, accuracy=1e-6, **kwargs)
    with pytest.raises(RuntimeError, match='did not converge.*residual'):
        system.energies
