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

# ## Rabi Oscillations in TLS
#
# We study Rabi oscillations in isolated two-level system.
#
# NOTE: this tutorial makes heavy use of the liberal way GRANAD lets you group orbitals. You might want to consult the tutorial on orbital lists first.

### Building a two-level system
#
# Consider an isolated atom, modelled as a two-level system. Each level is an orbital, so let's combine them in a list.

# +
from granad import Orbital, OrbitalList

lower_level = Orbital(tag="atom")
upper_level = Orbital(tag="atom")
atom = OrbitalList([lower_level, upper_level])
# -

# We have used a tag to signify that these the orbitals belong to the same atom. Let's see what we have done.

# +
print(atom)
# -

# We see that GRANAD assumes that every orbital is filled. But we want only the one of the levels filled. So, we set the electron number to 1.

# +
atom.simulation_params.electrons = 1
print(atom)
# -

# We now need to specify the Hamiltonian. We can do so by setting the elements corresponding to the orbitals.

# +
atom.set_hamiltonian_element(upper_level, lower_level, 2.0)
atom.set_hamiltonian_element(upper_level, upper_level, 0.5)
atom.set_hamiltonian_element(lower_level, lower_level, -0.5)
print(atom)
print(atom.hamiltonian)
# -

# Setting dipole transitions is similar. We want the lower and upper level to be connected by a dipole transition in z-direction.

# +
atom.set_dipole_transition(upper_level, lower_level, [1, 0, 0])
print(atom)
# -

# We set the initial excited state (in our point of view, this is a HOMO-LUMO transition).

# +
atom.set_excitation(atom.homo, atom.homo + 1, 1)
print(atom)
# -

# We consider a continuous wave as an external illumination.

# +
from granad import Wave

wave = Wave(amplitudes=[0.05, 0, 0], frequency=2)
# -

# We propagate the system in time.

# +
time, density_matrices = atom.get_density_matrix_time_domain(
    end_time=10, relaxation_rate=1, illumination=wave, use_rwa=True
)
atom.show_time_dependence(density_matrices, time=time)
# -
