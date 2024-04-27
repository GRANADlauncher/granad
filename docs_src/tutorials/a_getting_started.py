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

# # Getting started
#
# We introduce the basics of GRANAD and do a quick simulation.

#
### Orbitals
#
# At its core, GRANAD is all about orbitals. Let's create one at the origin and inspect it.
#

# +
from granad import Orbital

my_first_orbital = Orbital(
    position = (0, 0, 0),
    tag = "a tag contains arbitrary information",
)
print(my_first_orbital)
# -

# The group_id, unsurprisingly, groups orbitals. For example: if you create a two-level adatom, you need two orbitals that share the same group_id. In the same way, all orbitals in a graphene sheet share the same group_id.

# +

my_second_orbital = Orbital(
    position = (1, 1, 1),
    tag = "a new, shifted orbital",
)
print(my_second_orbital)

# -

# This is all there is to know about orbitals!

#
### Materials
#
# Materials are stuff you can cut orbitals from. You can define one yourself or import a prebuilt one. We will use the latter option.

# +
from granad import MaterialCatalog
MaterialCatalog.available()
# -

# Let's inspect a material.

# +
MaterialCatalog.describe("graphene")
# -

# There are parameters regarding the geometry, the type of the involved orbitals (the built-in graphene model contains only a single spin polarized pz orbital) , the position of orbitals in the unit cell and the interactions (currently, GRANAD supports hoppings and Coulomb interactions). Let's pick a concrete material.

# +
graphene = MaterialCatalog.get("graphene")
# -

### OrbitalLists
#
# OrbitalLists are the last class you need to know. Unsurprisingly, an OrbitalList is a list of orbitals. You can create one yourself from the two orbitals you created above

# +
from granad import OrbitalList

my_first_orbital_list = OrbitalList([my_first_orbital, my_second_orbital])
print(my_first_orbital_list)
# -

# Alternatively, you get orbital lists if you cut a flake from a material. You do this by specifying the shape of the flake.
# You can specify any shape you want, but this is covered in a separate tutorial.
# For now, we will use a built-in shape: an equilateral triangle with a side length of 15 Angström.

# +
from granad import Triangle
import jax.numpy as jnp
triangle = Triangle(15, armchair = True)
# -

# Now, our shape is ready and we can start cutting. To make sure that we are satisfied with what we get, we plot the flake. By default, GRANAD cuts any "dangling" atoms.

# +
my_first_flake = graphene.cut_flake(triangle, plot = False)
print(my_first_flake)
# -

# For more information on cutting, including different edge types and how to keep dangling atoms, have a look at the corresponding tutorial.

### A first simulation

# To get a feeling of the setup, we first inspect the energies of the flake

# +
my_first_flake.show_energies()
# -

# Physical observables are expectation values of Hermitian operators. GRANAD offers access to the time-resolved density matrix $\rho(t)$ of a system by integrating a nonlinear master equation. As a result, it is possible to track the evolution of the physical observable associated with a Hermitian operator $A$ by computing $a(t) = Tr[\rho(t) A]$. Optical properties in particular are largely determined by the polarization or dipole operator $\hat{P}$ and they are usually expressed in frequency domain. To this end, GRANAD offers a way to compute the Fourier transform $a(\omega)$ directly after time propagation. We will look at an example tracking the time evolution of the dipole operator below, where the computation proceeds in two steps:

#
# 1. Excite the flake with an electric field.
# 2. Compute its dipole moment $p(\omega)$ from the expectation value of its dipole operator.

# We first do step 1. To obtain a broad frequency spectrum, we must pick a narrow pulse in time-domain

# +
from granad import Pulse

my_first_illumination = Pulse(
    amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
) 
# -

# For step 2, a few parameters have to be chosen
#
# 1. Simulation duration: we go from 0 to 40 in 1e5 steps.
# 2. Relaxation rate: this is $r$ in the dissipation term $D[\rho] = r \cdot(\rho - \rho_0)$ in the master equation.
# 3. Frequency domain limits: we choose the interval [0, 16].
# 4. Density matrix sampling rate: producing 1e5 density matrices can quickly exhaust RAM ressources. So we only save every 100th density matrix, such that we get 1000 density matrices.

# A simulation is just passing all of these parameters to the corresponding method of our flake.

# +
omegas, dipole_omega, pulse_omega =  my_first_flake.get_expectation_value_frequency_domain(
        operator=my_first_flake.dipole_operator, 
        end_time=40,
        steps_time=1e5,
        relaxation_rate=1 / 10,
        illumination=my_first_illumination,
        omega_min=0,
        omega_max=10,
        skip=100,
    )
# -

# We see that three variables are returned: the omega axis we have specified, the dipole moment and the pulse in freqeuency domain. There is no way to control the number of points in the omega axis, because it is the result of a Fourier transform.

# We now plot the dipole moment and the pulse in frequency domain.

# +
import matplotlib.pyplot as plt

plt.plot(omegas, dipole_omega)
plt.plot(omegas, pulse_omega, "--")
plt.show()
# -
