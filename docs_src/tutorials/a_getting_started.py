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
# We introduce the basics of GRANAD and perform an example simulation.

### Introduction

# GRANAD lets you simulate structures from a few orbitals up to finite bulks. The process of specifying a structure is designed to be easy and interactive: You can add, shift, combine and manipulate parts of your structure at varying levels of detail involving large groups of orbitals or  single orbitals. To this end, GRANAD offers three fundamental datatypes:

# 1. *Orbitals* are the fundamental building block of a structure.
# 2. *Materials* are a stand-in for infinite bulks. You can cut finite pieces from these bulks, which correspond to lists of orbitals with specified positions and properties.
# 3. *OrbitalLists* represent a concrete structure. In essence, you can handle these as normal Python lists with additional information regarding Orbital coupling.

# We will use each of them below to set up a small graphene nanoflake coupling to an external electric field.

# Once this is done, we will use GRANAD's main feature and simulate the induced dynamics in the nanoflake directly in time domain.
#
### Orbitals
#
# Here, we create an orbital and print its properties to understand its structure. By default, each orbital is considered to be occupied by 1 electron.
#

# +
from granad import Orbital

my_first_orbital = Orbital(
    tag = "a tag contains arbitrary information",
)
print(my_first_orbital)
# -

# From the output above, we see that orbitals are characterized by their position and a user-defined tag. They are placed at the origin by default. We can change this by passing an explicit position.

# +

my_second_orbital = Orbital(
    position = (1, 1, 1),
    tag = "a new, shifted orbital",
)
print(my_second_orbital)

# -

#
### Materials
#
# Materials can be used to create orbital lists. Users can define custom materials or choose among the prebuilt options. Prebuilt materials correspond to fixed models common in the literature. In modelling electronic interactions, GRANAD supports hoppings and Coulomb interactions. The built-in graphene model contains only a single pz orbital per atom. Materials are obtained through the MaterialCatalog

# +
from granad import MaterialCatalog
MaterialCatalog.available()
# -

# Let's inspect a material.

# +
MaterialCatalog.describe("graphene")
# -

# This function prints a description of the material, i.e., its parameters, geometry, the type of the involved orbitals, the position of orbitals in the unit cell and the interactions. Let's pick a concrete material.

# +
graphene = MaterialCatalog.get("graphene")
# -

# NOTE: Materials offered by the Catalog are populated with default values. They can be changed by defining custom materials or via convenience wrappers, e.g. `get_ssh` as follows:

# +
my_ssh_model = get_ssh(delta, displacement)
my_ssh_chain = my_ssh_model.cut_flake(10)  # 10 unit cells, as usual
# -

### OrbitalLists
#
# An OrbitalList is a list of orbitals. They can be created directly by combining user-defined orbitals

# +
from granad import OrbitalList

my_first_orbital_list = OrbitalList([my_first_orbital, my_second_orbital])
print(my_first_orbital_list)
# -

# This displays information about the number of orbitals, electrons and single-particle excitations involved in constructing the initial density matrix.

# Alternatively, orbital lists can represent flakes cut from a material. You do this by specifying the shape of the flake.
# You can specify an arbitrary shape, but this is covered in a separate tutorial.
# For now, we will use a built-in shape: an triangle with a base length of 18 Angström.

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

# Note that the `plot=True` option displays an "infinite" graphene sheet (blue points), from which only the flake (orange points) will be kept in the final structure. If you want to visualize the flake in isolation, use `show_2d`, as explained in more detail in the plotting tutorial.

# There is an extra option for dealing with graphene-like lattices we can pass to the built-in shape, which is the armchair boolean. It rotates the shape to get the correct edge type. The optional "shift" argument lets you shift the shape in the plane.

# +
triangle_ac = Triangle(18, armchair = True)
my_first_flake = graphene.cut_flake(triangle_ac, plot = True)
# -

# For more information on cutting, including different edge types and how to keep dangling atoms, have a look at the corresponding tutorial.

### A first simulation

# GRANAD offers access to the time-resolved density matrix $\rho(t)$ of a system by integrating a nonlinear master equation. The exact mathematical details are covered in the tutorial on time-domain simulations. Once the time dependent density matrix is known, dynamical expectation values can be computed. For a Hermitian operator $A$, the expectation value is then  $a(t) = \text{Tr}[\rho(t) A]$. We will illustrate this with the example of the dipole moment in the small graphene flake we created above.

# The density matrix is normalized to allow two electrons per single-particle energy eigenstate and populated according to the Aufbau principle. The energy landscape together with the initial state occupation can be visualized as follows

# +
my_first_flake.show_energies()
# -

# GRANAD offers similar built-in functions to visualize static and dynamic properties of a flake. For more information, please consult the corresponding tutorial.

# Now that we are ready, we can study the induced dipole moment. In particular, we will:

# 1. Excite the flake with an electric field.
# 2. Compute its dipole moment $\bf{p}(t) = \text{Tr}[\rho(t) \bf{P}]$, where $\bf{P}$ is the dipole operator.

# The electric field is given as a function mapping the time (a single float) to a vector representing the field components like this `field : t -> vec`.
 
# In addition to GRANAD's built-in functions, custom functions for illumination can be specified. We will use the built-in Pulse.

# +
from granad import Pulse

my_first_illumination = Pulse(
    amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
)
print( my_first_illumination( 0.0) ) # initial value of the field
# -

# The pulse corresponds to a plane wave with a Gaussian temporal profile. To visualize it, please consult the electric fields tutorial.

# Now we come to the actual simulation. For any time-domain simulation, we have to decide on a few additional parameters:
#
# 1. Simulation duration: here, we set the duration from 0 to 40 units.
# 2. Relaxation rate: here, we pick a single number characterizing the rate of energy dissipation across all channels.
# 3. The operators whose expectation values we want to compute. They are given as a simple list.

# We want to calculate the induced polarization from the dipole operator. This operator can be represented as a 3xNxN matrix, where N is the number of orbitals and 3 corresponds to Cartesian components x,y,z and we can compute it directly

# +
print(my_first_flake.dipole_operator.shape)
# -

# We want to compute its expectation value, so we have to wrap it in a list and pass it to the time-domain simulation, called `master_equation`

# +
result = my_first_flake.master_equation(
    end_time=40, # the start is set to 0 by default 
    relaxation_rate=1 / 10,
    illumination=my_first_illumination,
    expectation_values = [my_first_flake.dipole_operator] # you can also omit the list for a single operator, but this is bad practice
    )
# -

# If you want to compute expecation values of more operators, you can add them to the list.

# The result variable is a container for 

# 1. the last density matrix in the simulation, which is important if you want to continue the time evolution.
# 2. the time axis, which is an array of samples [t_1, t_2, ... t_n].
# 3. an "output", which is a list of arrays, corresponding to the operators we passed in. Each array contains the time-dependent expectation value like [p_1, p_2, ..., p_n].

# We have specified one operator to compute the expectation value of, such that the list only contains one element.

# +
print(len(result.output))
# -

# This list contains an array of shape Tx3, where T is the number of saved time steps. So,  the dipole moment at the 10-th timestep in x-direction is obtained by

# +
dipole_moments = result.output[0]
print(dipole_moments[10, 0])
# -

# Now that we understand how a time domain simulation works, we can visualize the result. GRANAD offers a dedicated function for this

# +
my_first_flake.show_res( result, plot_labels = ["p_x", "p_y", "p_z"], show_illumination = False ) 
# -
