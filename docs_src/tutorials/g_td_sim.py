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

# # Time-Domain simulations
#

# We introduce time-domain simulations, which are based on GRANAD's default master equation

# $$\dot{\rho}(t) = -i [H[t,\rho(t)], \rho] + \mathcal{D}[t, \rho]$$

# The non-linear Hamiltonian is given by

# $$H[t,\rho(t)] = h^{0} + \vec{E} \vec{P} + C(\rho(t) - \rho^{0})$$

# where 

# 1. $h^{0}$ is the unperturbed or bare Hamiltonian constructed from localized orbitals.
# 2. $\vec{P}$ is the dipole moment operator depending on space $\vec{r}$ and transition dipole moments $\vec{d}$, leading to an interatomic potential with unit charge $e$, given by $e \vec{E} \vec{r}$, and an intra-atomic potential $\vec{E} \vec{d}$.
# 3. $\rho^{0}$ is the stationary density matrix of the system's ground state, and $C$ is the Coulomb matrix (a subset of two-body matrix elements).

# This expression is the dipole-gauge expression for the Hamiltonian including an external electric field $\vec{E}$.

### Observables


# You can compute multiple observables in one simulation run

# +
from granad import MaterialCatalog, Hexagon, Pulse
flake = MaterialCatalog.get("graphene").cut_flake( Hexagon(10) )

pulse = Pulse(
    amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
)

operators = [flake.dipole_operator, flake.velocity_operator]

result = flake.master_equation(
    relaxation_rate = 1/10,
    illumination = pulse,
    expectation_values = operators,
    end_time = 40,
     )
# -

# The result object stores this info. Operators are concatenated in the order you passed them in.

# +
print(len(result.output))
print(result.output[0].shape)
# -

# The induced dipole moment at timestep 10 is given by

# +
print(result.output[0][10,:3])
# -

# Induced current at timestep 10

# +
print(result.output[0][10,3:])
# -


# We can access the Fourier transform as

# +
omega_min, omega_max = 0, 5
omegas, pulse_omega = result.ft_illumination( omega_min = omega_min, omega_max = omega_max )
output_omega = result.ft_output( omega_min = omega_min, omega_max = omega_max )[0]
# -

# From the classical continuity equation for the charge and current density $\rho(\vec{r},t), \vec{j}(\vec{r},t)$ and dipole moment density $\vec{p}(\vec{r},t)$, we have

# $$\dot{\rho}(\vec{r}, t) = -\nabla \vec{j}(\vec{r}, t) = \nabla \dot{\vec{p}}(\vec{r}, t) \implies \dot{\vec{p}}(\vec{r},t) = \vec{j}(\vec{r},t)$$

# After Fourier transformation, we obtain

# $$-i\omega \vec{p}(\vec{r}, \omega) = \vec{j}(\vec{r}, \omega)$$

# Its integrated form is given by

# $$\int d\vec{r} \dot{\vec{p}}(\vec{r},t) = \dot{\vec{p}}(t) = \int d\vec{r} \vec{j}(\vec{r},t) =  \vec{j}(t)$$ 

# $\vec{j}(t)$ and $\vec{p}(t)$ are the total current and dipole moment respectively. As classical quantities are given by ensemble averages of quantum operators, we can
# obtain these quantities directly from the time-domain simulations. The (integrated) continuity equation above can be verified as follows

# +
import matplotlib.pyplot as plt
p = -(omegas * output_omega[:,0]).imag
j = output_omega[:,3].real
plt.plot(omegas, p, label = r'$- \text{Im}[\omega p_x]$')
plt.plot(omegas, j, '--', label = r'$\text{Re}[j_x]$')
plt.xlabel(r'$\omega$ (eV)')
plt.ylabel(r'$j$')
plt.legend()
plt.show()
# -


# The field is also accessible

# +
print(result.td_illumination.shape)
# -

### Density matrices

# If we want to only get density matrices, we can  omit the operator list. The result object then contains a one-element list.

# +
result = flake.master_equation(
    relaxation_rate = 1/10,
    illumination = pulse,
    end_time = 40,
    density_matrix = ["full"], # this argument may be removed
     )
density_matrix = result.output[0]
print(density_matrix.shape)
# -

# We can convert them to energy basis

# +
density_matrix_e = flake.transform_to_energy_basis( density_matrix )
print(density_matrix_e.shape)
# -

### Occupations

# To reduce memory consumption, we can extract only the site occupations to avoid storing the entire stack of density matices in memory. 

# +
result = flake.master_equation(
    relaxation_rate = 1/10,
    illumination = pulse,
    density_matrix = ["occ_x"],
    end_time = 40,
     )
occ_x = result.output[0]
print(occ_x.shape)
# -

# Further reduction of the memory consumption can be controlled by the grid argument, which describes the density of the time domain sampling grid, explained in the API section on the `master_equation` method.

### Initially excited states

# GRANAD allows to set initially excited states by specifying the single-particle transition in the density matrix as follows

# +
flake.set_excitation( flake.homo, flake.homo + 1, 1)
flake.show_energies()
# -

# In case of degeneracies, GRANAD distributes electrons equally among all degenerate energy levels, as demonstrated in the figure above.

### Energy occupations

# Similarly to site occupations, only energy occupations can also be obtained from the time-domain simulations.

# *Warning*: this introduces additional cubic complexity 

# +
result = flake.master_equation(
    relaxation_rate = 1/10,
    density_matrix = ["occ_e"],
    end_time = 40,
     )
flake.show_res(result, plot_only = [flake.homo, flake.homo+1], plot_labels = ["homo", "lumo"], show_illumination = False )
# -

### Combinations

# We can also extract multiple quantities at the same time

# +
result = flake.master_equation(
    relaxation_rate = 1/10,
    density_matrix = ["full", "occ_x"],
    expectation_values = [flake.dipole_operator],
    end_time = 40,
    illumination = pulse,
)
# -

# The output will now contain three arrays: induced dipole moments, site occupations and full density matrices

# +
print(len(result.output))
print(result.output[0].shape) # by default, operators come first
print(result.output[1].shape) # we specified ["full", "occ_x"] => full density matrices
print(result.output[2].shape) # we specified ["full", "occ_x"] => site occupations
# -

### EPI

# The energy-based plasmonicity index is a quantity to characterize the steady-state plasmonicity. We use plane wave illumination to drive the system into a steady state and then compute the EPI

# +
from granad import Wave

omega = 2.5 # we pick the approximate resonance at 2.5 
res = flake.master_equation(
    relaxation_rate = 1/10,
    illumination = Wave(frequency = omega / 2 * jnp.pi, amplitudes = [1e-5, 0, 0]),
    end_time = 40)

# numerical broadening parameter of 0.05 eV
epsilon = 0.05

# print and compute the EPI
print(flake.get_epi(res.final_density_matrix, omega = omega, epsilon = epsilon))
# -
