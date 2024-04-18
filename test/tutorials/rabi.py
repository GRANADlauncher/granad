# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: base
# ---

# ## Rabi Oscillations
#
# We study Rabi oscillations and get more familiar with orbital lists.

### Building the adatom
#
# The adatom is modelled as a two-level system. Each level is an orbital, so let's combine them in a list.
#
# +
from granad import OrbitalList, Orbital
lower_level = Orbital( (0,0,0) )
upper_level = Orbital( (0,0,0) )
adatom = OrbitalList( [lower_level, upper_level] )
print(adatom)
# -

# We see that GRANAD assumes that every orbital is filled. But we want only the one of the levels filled. So, we set the electron number to 1.

# +
adatom.electrons = 1
print(adatom)
# -

# We now need to specify the Hamiltonian. We can do so by setting the elements corresponding to the orbitals.

# +
adatom.set_hamiltonian_element( upper_level, lower_level, 2.0 )
adatom.set_hamiltonian_element( upper_level, upper_level, 0.5 )
adatom.set_hamiltonian_element( lower_level, lower_level, -0.5 )
print(adatom)
print(adatom.hamiltonian)
# -

# Setting dipole transitions is similar.

# +
adatom.set_dipole_transition( upper_level, lower_level, [1,0,0])
print(adatom)
# -

# We set the initial excited state

# +
adatom.set_excitation( adatom.homo, adatom.homo+1, 1)
print(adatom)
# -

# We consider a continuous wave as an external illumination.

# +
from granad import Wave
wave = Wave(amplitudes = [0.05, 0, 0], frequency = 2)
# -

# We propagate the system in time.

# +
time_axis, density_matrices = adatom.get_density_matrix_time_domain( end_time = 10, relaxation_rate = 1, illumination = wave )
adatom.show_energy_occupations( time_axis, density_matrices )
# -



