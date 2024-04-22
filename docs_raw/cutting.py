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

# ## Cutting Tutorial
#
# We introduce the basics of GRANAD and do a quick simulation.

### Materials
#
# At its core, GRANAD is all about orbitals. Let's create one at the origin and inspect it.
#
# +

from granad import Orbital

my_first_orbital = Orbital( (0,0,0), "this is some information about my first orbital I might need to use later" )
my_first_orbital

# -

# The group_id, unsurprisingly, groups orbitals. For example: if you create a two-level adatom, you need two orbitals that share the same group_id. In the same way, all orbitals in a graphene sheet share the same group_id. For now, a quick warning is in order: Orbitals are immutable. This crashes

# +

#my_first_orbital.position = (1,1,1)

# -

# If you need a new orbital at (1,1,1), just create a new one and forget about the old

# +

my_second_orbital = Orbital( (1, 1, 1) )
my_second_orbital

# -

# This is all there is to know about orbitals!

#
# ### Materials
#
# Materials are stuff you can cut orbitals from. Let's see how this works

# +
from granad import Material2D

Material2D.available()

# -

# So we see that we have a few materials available. Let's pick one. As usual, just showing what graphene is is better than long-form explanations

# +

graphene = Material2D.get( "graphene" )
graphene

# -

# ### OrbitalLists
#
# OrbitalLists are the last class you need to know. Unsurprisingly, an OrbitalList is a list of orbitals. You can create one yourself

# +
from granad import OrbitalList

my_first_orbital_list = OrbitalList( [my_first_orbital, my_second_orbital] )
my_first_orbital_list
# -

# Alternatively, you get orbital lists if you cut a flake from a material. You do this by specifying the shape of the flake, e.g. a triangle, hexagon, circle, ... .
# You can specify any shape you want, but this is covered in a separate tutorial. For now, we will use a built-in shape: an equilateral triangle with a side length of 10 Angström. TODO: reference units, currently 

# +
from granad import Triangle
triangle = Triangle(15)
# -

# Now, our shape is ready and we can start cutting. To make sure that we are satisfied with what we get, we plot the flake.

# +
my_first_flake = graphene.cut_orbitals( triangle, plot = True )
# -

# Let's say that this is not what we want. For example, we might have gotten the wrong edge type. The solution to this is simple: we just have to rotate the triangle by 90 degrees to get the correct edge type.
# Luckily, rotated shapes are already built in by passing armchair = True in the "constructor" of any shape "object".

# +
triangle_ac = Triangle(17, armchair = True) 
my_second_flake = graphene.cut_orbitals( triangle_ac, plot = True )
1/0
# -

# Now we are satisified and can start simulating.

# ### A quick simulation

# To get a feeling of the setup, we first inspect the energies of the flake

# +
# my_second_flake.show_energies()
# -

# Let's say we want to compute the absorption spectrum of the flake. One way of doing this is based on integrating a modified von-Neumann equation in time-domain. You can do this in two steps: # TODO: ref
# 
# 1. Excite the flake with an electric field.
# 2. Compute its dipole moment $p(\omega)$ from the expectation value of its dipole operator to obtain the absorption spectrum from $Im[p(\omega)]$.

# We first do step 1. To make sure that we get a meaningful spectrum, we must pick an electric field with many frequencies, i.e. a narrow pulse in time-domain

# +
from granad import Pulse
my_first_illumination = Pulse( amplitudes = [1e-5, 0, 0], frequency = 1, peak = 1.0, fwhm = 0.1 ) # TODO: units
# -

# For step 2, we have to think of a few parameters:
#
# 1. TD simulation duration: we go from 0 to 40 in 1e5 steps.
# 2. relaxation rate: this is $r$ in the dissipation term $D[\rho] = r \cdot(\rho - \rho_0)$.
# 3. frequency domain limits: we choose the interval [0, 16].
# 4. density matrix sampling rate: producing 1e5 density matrices can quickly tire our RAM. so we only save every 100th density matrix, such that we get 1000 density matrices.

# A simulation is just passing all of these parameters to the corresponding method of our flake

# +
omegas, dipole_omega, pulse_omega = flake.get_expectation_value_frequency_domain(
    operator = flake.dipole_operator, # the dipole moment is the expectation value of the dipole operator
    end_time = 40,
    steps_time = 1e5,
    relaxation_rate = 1/10,
    illumination = my_first_illumination,
    omega_min = 0,
    omega_max = 60,
    skip = 100,
)
# -

# We now have the dipole moment and the electric field in frequency domain. To compute the absorption, we need the imaginary part

# +
import matplotlib.pyplot as plt
spectrum = -(dipole_omega / pulse_omega).imag * omegas[:,None]
plt.plot(omegas, jnp.abs(spectrum) / jnp.max(jnp.abs(spectrum)) )
plt.show()
# -

