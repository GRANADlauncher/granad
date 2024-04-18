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

# ## Getting started
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
from granad import Material

Material.available()

# -

# So we see that we have a few materials available. Let's pick one. As usual, just showing what graphene is is better than long-form explanations

# +

graphene = Material.get( "graphene" )
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
# You can do this manually, but this is covered in a separate tutorial. For now, we will use a built-in shape.

# +
from granad import Shapes

triangle = Shapes.triangle
triangle

# -

# So, a shape is just an array. To be more precise, it's a [polygon](https://en.wikipedia.org/wiki/Polygon#Naming). By default, the built-in shapes have a side-length of 1 Angström. TODO: reference units
# Since they are arrays, we can scale them to any size we want!

# + 
ten_angstroem_wide_triangle = 10 * triangle
# -

# Now, our shape is ready and we can start cutting. To make sure that we are satisfied with what we get, we plot the flake.

# +
my_first_flake = graphene.cut_orbitals( ten_angstroem_wide_triangle, plot = False )
# -

# Let's say that this is not what we want. For example, we might have gotten the wrong edge type. The solution to this is simple: we just have to rotate the triangle by 90 degrees to get the correct edge type. Luckily, rotated shapes are already built in.

# +
triangle_rotated = Shapes.triangle_rotated
ten_angstroem_wide_rotated_triangle = 10 * triangle_rotated
my_second_flake = graphene.cut_orbitals( ten_angstroem_wide_rotated_triangle, plot = False )
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
# 2. Compute its dipole moment $p(\omega)$  to obtain the absorption spectrum from $Im[p(\omega)]$.

# We first do step 1. To make sure that we get a meaningful spectrum, we must pick an electric field with many frequencies, i.e. a narrow pulse in time-domain

# +
from granad import Pulse
my_first_illumination = Pulse( amplitudes = [1e-5, 1e-5, 0], frequency = 1, peak = 1.0, fwhm = 0.1 ) # TODO: units
# -

# Step 2 looks like this # TODO: explain better

# import jax.numpy as jnp; (my_first_illumination(0.1) + jnp.arange(4*3).reshape(4,3)).shape, (jnp.arange(4*3).reshape(4,3) + my_first_illumination(0.1)  ).shape

# +
import matplotlib.pyplot as plt
omegas, polarizability = my_second_flake.get_polarizability_time_domain( end_time = 0.1, relaxation_rate = 0.1, illumination = my_first_illumination )
plt.plot( omegas, polarizability[:,0].imag )
plt.show()
# -

orbs.expectation_value( orbs.operator, orbs.rho )

# This computation does not involve hidden state. It's just offered by the orbital instance for convenience. This here is equally valid

OrbitalList.expectation_value( orbs.operator, orbs.rho )


transition_dipole_moments = orbs.dipole_operator.to_energy_basis()
