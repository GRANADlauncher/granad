import numpy as np
import matplotlib.pyplot as plt
from granad import *

# TODO: tests for: default rhs (with all dissipators)

def test_rabi():
    d = 3
    e_field_strength = 0.5
    t = jnp.linspace(0, 10, 100)
    freq  = d * e_field_strength
    analytical = jnp.cos( freq/2 * t) ** 2
    
    lower_level = Orbital( )
    upper_level = Orbital( )
    adatom = OrbitalList( [lower_level, upper_level] )
    adatom.set_hamiltonian_element( lower_level, lower_level, 0 )
    adatom.set_hamiltonian_element( upper_level, upper_level, 2 )
    adatom.set_electrons(1)
    adatom.set_dipole_element( upper_level, lower_level, [d,0,0] )    
    wave = Wave(amplitudes = [e_field_strength * 1j / 2, 0, 0], frequency = max(adatom.energies) - min(adatom.energies) )        

    result = adatom.master_equation(
        start_time = t.min(),
        end_time = t.max(),
        grid = t,
        dt = 1e-5,
        illumination = wave,
        use_rwa = True,
        density_matrix = ["occ_e"]
    )
    occupations = result.output[0]    
    np.testing.assert_allclose( occupations[:,0], analytical, rtol = 1e-5 )
