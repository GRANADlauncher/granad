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

### Introduction

# GRANAD lets you simulate structures from a few orbitals up to finite bulks. The process of specifying a structure is designed to be easy and interactive: You can add, shift, combine and manipulate parts of your structure at varying levels of detail involving large groups of orbitals or just single orbitals. To this end, GRANAD offers three fundamental datatypes:

# 1. *Orbitals* are the fundamental building block of a structure.
# 2. *OrbitalLists* represent a concrete structure. In essence, you can handle these as normal Python lists with additional information regarding Orbital coupling.
# 3. *Materials* are a stand-in for infinite bulks. You can cut finite pieces from these bulks, which are again just lists of orbitals.

# We will use each of them below to set up a small graphene nanoantenna coupling to an external field.

# Once this is done, we will use GRANAD's main feature and simulate the induced dynamics in the nanoantenna directly in time domain.
#
### Orbitals
#
# At its core, GRANAD is all about orbitals. Here we create an orbital and print its properties to understand its structure.
#

# +
from granad import Orbital

my_first_orbital = Orbital(
    tag = "a tag contains arbitrary information",
)
print(my_first_orbital)
# -

# The group_id, unsurprisingly, groups orbitals. It shouldn't concern us too much now. From the output above, we see that orbitals are placed at the origin by default. We can change this by passing an explicit position.

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

# There are parameters regarding the geometry, the type of the involved orbitals (the built-in graphene model contains only a single pz orbital per atom), the position of orbitals in the unit cell and the interactions (currently, GRANAD supports hoppings and Coulomb interactions). Let's pick a concrete material.

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
# For now, we will use a built-in shape: an hexagon with a base length of 10 Angström.

# +
from granad import Triangle
import jax.numpy as jnp
triangle = Triangle(18)
# -

# Now, our shape is ready and we can start cutting. To make sure that we are satisfied with what we get, we plot the flake. By default, GRANAD cuts any "dangling" atoms.

# +
my_first_flake = graphene.cut_flake(triangle, plot = True)
print(my_first_flake)
# -

# There is an extra option for dealing with graphene-like lattices we can pass to the built-in shape, which is the armchair boolean. It just rotates the shape to get the correct edge type. The optional "shift" argument lets you shift the shape in the plane.

# +
triangle_ac = Triangle(18, armchair = True, shift = [10,10])
my_first_flake = graphene.cut_flake(triangle_ac, plot = True)
# -

# For more information on cutting, including different edge types and how to keep dangling atoms, have a look at the corresponding tutorial.

### A first simulation

# Physical observables are expectation values of Hermitian operators. GRANAD offers access to the time-resolved density matrix $\rho(t)$ of a system by integrating a nonlinear master equation. Once the time dependent density matrix is known, dynamical expectation values can be computed. Say you have a Hermitian operator epresented by a matrix $A$ and the solution of the master equation $\rho(t)$. The expectation value is then just $a(t) = \text{Tr][\rho(t) A]$. We will illustrate this at the example of the dipole moment in the small graphene flake we created above.

# But before we dive into exploring the dynamics of the flake, we first inspect its energies

# +
my_first_flake.show_energies()
# -

# GRANAD offers similar built-in functions to visualize (both static and dynamic) properties of a flake. For more information, please consult the corresponding tutorial.

# Now that we are ready, we can study the induced dipole moment. In particular, we will:

# 1. Excite the flake with an electric field.
# 2. Compute its dipole moment $p(t) = \text{Tr}[\rho(t) P]$

# The electric field is given as a function mapping the time (a single float) to a vector representing the field components like this `field : t -> vec`.
 
# You can specify custom functions, but GRANAD comes with a few built-ins. We will just use the built-in Pulse.

# +
from granad import Pulse

my_first_illumination = Pulse(
    amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
)
print( my_first_illumination( 0.0) ) # initial value of the field
# -


# Now we come to the actual simulation. For any time-domain simulation, we have to decide on a few additional parameters:
#
# 1. Simulation duration: we go from 0 to 40 to make sure everything has equilibrated.
# 2. Relaxation rate: here, we pick a single number characterizing the rate of energy dissipation across all channels.
# 3. The operators whose expectation values we want to compute. They are given as a simple list.

# We want to calculate the induced polarization from the dipole operator. This operator can be represented as a 3xNxN matrix, where N is the number of orbitals and 3 corresponds to Cartesian components x,y,z and we can compute it directly

# +
print(my_first_flake.dipole_operator.shape)
# -

# We want to compute its expectation value, so we have to wrap it in a list and pass it to the TD simulation, called `td_run`

# +
result = my_first_flake.td_run(
    end_time=40, # the start is set to 0 by default 
    relaxation_rate=1 / 10,
    illumination=my_first_illumination,
    expectation_values = [my_first_flake.dipole_operator] # you can also omit the list for a single operator, but this is bad practice
    )
# -

# If you want to compute expecation values of more operators, you can simply add them to the list. The result variable is a container for 

# 1. the last density matrix in the simulation (this is important if you want to continue the time evolution).
# 2. the time axis, which is an array of samples [t_1, t_2, ... t_n].
# 3. an "output", which is a list of arrays, corresponding to the operators we passed in. Each array contains the time-dependent expectation value like [p_1, p_2, ..., p_n].

# We have only specified one operator to compute the expectation value of, such the list only contains one element.

# +
print(len(result.output))
# -

# This list contains an array of shape Tx3. So, if we want the dipole moment at the 10-th timestep in x-direction, we would do

# +
dipole_moments = result.output[0]
print(dipole_moments[10, 0])
# -

# Now that we understand how a time domain simulation works, we can visualize the result. GRANAD offers a dedicated function for this

# +
my_first_flake.show_res( result, plot_labels = ["p_x", "p_y", "p_z"], show_illumination = False ) 
# -
