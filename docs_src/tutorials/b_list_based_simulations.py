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

# # List-based Simulations 
#
# We detail the main datatype of GRANAD.

### Orbitals: A Recap
# 
# As already explained in the "Getting Started" Tutorial, Orbitals are the basic building blocks of orbital lists. Let's inspect the class

# +
from granad import *
print(Orbital.__doc__)
# -

### Orbital lists

# To group orbitals, we put them in a list. 

# +
print(OrbitalList.__doc__)
# -

# So the docstring tells us that orbital lists

# 1. allow to couple orbitals 
# 2. store simulation parameters in a dataclass
# 3. compute physical observables lazily (it also talks about bases, but see the seperate tutorial).
# 4. let us simulate things

# We will look at these remaining points below.

### Coupling orbitals

# Say we want to create a stack of two graphene flakes.

# +
flake = MaterialCatalog.get("graphene").cut_flake( Rectangle(10, 10) )
# -

# How do we create the second flake and stack it on top? First of all, we duplicate the existing flake

# +
flake_shifted = MaterialCatalog.get("graphene").cut_flake( Rectangle(10, 10) )
# -

# We then inspect its orbitals.

# +
print(flake_shifted)
# -

# When the orbitals in a flake are instantiated, they are automatically assigned a group id that has not been in use before. This flake contains only one group for all pz orbitals in the flake. We can check this

# +
print(flake_shifted.get_group_ids()) # the i-th entry is the group of the i-th orbital
print(flake_shifted.get_unique_group_ids()) # this is just jnp.unique on the previous array
# -

# So, we know that all orbitals in the flake have the same group_id. As such, we can group them and shift them together.

# +
group_id_upper = flake_shifted.get_unique_group_ids()

# this shift will be applied to all orbitals with the same group => the flake is lifted in z-direction
flake_shifted.shift_by_vector( group_id_upper, [0,0,1] )
# -

# Creating the stack is easy now

# +
stack = flake + flake_shifted
stack.show_3d()
# -

# Okay, we have the geometry right, but what about the coupling? If we inspect the energies

# +
stack.show_energies()
# -

# We see that we get every point twice: the interlayer coupling is zero by default, so we have a two-fold degenerate spectrum. To lift this degeneracy, we need to couple the layers. We do this via a function depending on distance. Say, you want to couple only nearest neighbors in a layer with a strength of -2.66. Interlayer nearest neighbors are separated by a distance of 1 Angström. So, one way to express the coupling as a function is by a narrow gaussian around 1.0

# +
def interlayer_hopping( distance ):
    return jnp.exp( -100*(distance - 1.0)**2 )
# -

# If the distance is (sufficiently close to) 1.0, we couple with -2.66, otherwise we don't couple. So, we want to couple two groups: the lower group of pz orbitals (the flake in the xy-plane) and the upper group. We do this like this:

# +
lower_id = flake.get_unique_group_ids()[0]
upper_id = flake_shifted.get_unique_group_ids()[0]
stack.set_groups_hopping( lower_id, upper_id, interlayer_hopping )
stack.show_energies()
# -

# The degeneracy is lifted! Consider now an adatom

# +
lower_level = Orbital(tag="atom")
upper_level = Orbital(tag="atom")
atom = OrbitalList([lower_level, upper_level])
print(atom)
# -

# For now, we have two electrons. We learn how to change this below. Each orbital has its own group_id. Don't change these, GRANAD handles these by default. We want to set energies of this adatom, i.e. its hamiltonian. If we just want a TLS with energies $\pm 0.5$, we have a $2x2$ matrix, where H[0,0] = -0.5, and H[1,1] = 0.5. We can set the hamiltonian elements directly

# +
atom.set_hamiltonian_element( 0, 0, -0.5)
atom.set_hamiltonian_element( 1, 1, 0.5) 
print(atom.hamiltonian)
# -

# We can also set the elements by directly referencing the orbitals

# +
atom.set_hamiltonian_element( upper_level, upper_level, 0.8)
print(atom.hamiltonian)
# -

# You can do the same thing for any element of the graphene flake btw. Talking about it, let's couple the atom to it. First, we combine the lists

# +
stack_with_atom = stack + atom
# -

# Now, we move the atom somewhere in between the two flakes

# +
stack_with_atom.show_3d( show_index = True )
# -

# We pick two indices we like and put the atom in between. To move all orbitals on the atom, we use the tag we just defined.

# +
new_atom_position = stack_with_atom[0].position + jnp.array( [0,0,0.5] )
stack_with_atom.set_position("atom", position = new_atom_position)
stack_with_atom.show_3d()
# -

# Now, we look at the energies

# +
stack_with_atom.show_energies()
# -

# Not much of a change, but the reason is that we forgot to couple the atom to the flakes. So let's do that. We couple it just to its nearest neighbors.

# +
stack_with_atom.set_hamiltonian_element( lower_level, 0, 0.3 ) # we can even mix orbitals and indices
stack_with_atom.set_hamiltonian_element( upper_level, 0, 0.3 )

stack_with_atom.set_hamiltonian_element( upper_level, 64, 0.3 )
stack_with_atom.set_hamiltonian_element( upper_level, 64, 0.3 )
stack.show_energies()
# -

# Setting coulomb elements works analogously. To wrap up, we can couple by 

# 1. setting matrix elements indexing via orbitals or their list indices
# 2. setting coupling functions via group ids (you can also pass orbitals to these functions, but the behavior is a bit weirder).


### Simulation Parameters

# Let's revisit the adatom we just built.

# +
print(atom)
# -

# We have two electrons, but a traditional TLS should only have one. We can do this like that

# +
atom.set_electrons( atom.electrons - 1 )
print(atom)
# -

# This looks better. Let's excite the transition (in our lingo, this is HOMO-LUMO)

# +
atom.set_excitation( 0, 1, 1)
atom.show_energies()
# -

# This works. Let's now combine it with another TLS

# +
a = Orbital(tag="atom2")
b = Orbital(tag="atom2")
atom2 = OrbitalList([a, b])

atoms = atom + atom2
print(atoms.energies)
print(atoms)
# -

# WAIT! We are back to 2 + 2 = 4 electrons, i.e. one per orbital? Why is this? The reason is that addition for orbital lists is defined as, schematically

# 1. orb1 + orb2 = [orb1, orb2]
# 2. coupling1 + coupling2 = [coupling1, coupling2]
# 3. param1 + param2  = default_params

# This might change in the future allowing you to define your own addition for all orbital list attributes. For now, we need to reset to the correct orbital number manually

# +
atoms.set_electrons( atoms.electrons - 1)
# -

# Another peculiarity of orbital lists is: they can contain each orbital only one time. So, if you try:

# +
# updated_stack = stack_with_atom + atoms
# -

# This will fail, because stack_with_atom already contains the atom contained in atoms. Admittedly, this is weird.

# If you are interested in the simulation parameters

# +
print(SimulationParams.__doc__)
# -

# So, they encapsulate the state of the simulation. Just remember to set them directly before you simulate.

### Lazy computation

# This is quickly explained: we have used it all the time!

# +
atom.set_hamiltonian_element(0,0,0.0) # doesnt compute anything
atom.set_hamiltonian_element(0,0,1.0) # still, nothing computed
atom.energies # now, we need to compute
# -


### Simulations

# Simulations don't change anything about the orbitals, so we can do the same time propagation with two different relaxation rates etc. The only methods that change anything about the list explicitly tell you so by starting with "set".
