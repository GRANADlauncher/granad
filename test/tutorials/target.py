# ## Getting started
#
# We introduce the basics of GRANAD and do a quick simulation.

### Materials
#
# At it's core, GRANAD is all about orbitals. Let's create one.
#
# +

from granad import Orbital

my_first_orbital = Orbital("model orbital", position = (0, 0, 0) )
my_first_orbital

# -

# The special id is probably the only mysterious bit about it. It is a way to group orbitals that we will demonstrate shortly. For now, a quick warning is in order: Orbitals are immutable. So, this will not work:

# +

my_first_orbital.position = (1,1,1)

# -

# If you need a new orbital named "model orbital" at (1,1,1), just create a new one and forget about the old

# +

my_second_orbital = Orbital("model orbital", position = (1, 1, 1) )
my_second_orbital

# -

# This is all there is to know about orbitals!

#
# ### Materials
#
# Materials are stuff that you can cut orbitals from. GRANAD offers you a few in a MaterialDatabase. Let's look at a graphene example. To make this a bit nicer, we will look at everyting with Python's pretty printing module "pprint"

# +
from granad import MaterialDatabase
from pprint import pprint

pprint(MaterialDatabase.graphene)
# -

# We see that graphene is a dictionary with quite a lot of information. Let's cover the entries, 

# 1. There are couplings, coulomb and hopping, given in the following form: all orbitals with the quantum numbers n,l,m,s = 0,0,1,0 sitting on carbon ("C") atoms 
# 2. 

# load graphene
graphene = Material( graphene )

# Let's inspect graphene more closely. It has an attribute, "orbitals", which is just a Python dict.

print(graphene.orbitals)

# Here you see that the premade version of graphene only contains pz orbitals at specific positions. Since graphene is a bulk material we want to cut finite pieces from, it obviously has an attribute defining its lattice basis.

print(graphene.lattice_basis)

# Now, we come to the couplings between the orbitals. GRANAD works with one-electron operators. This means they can be represented by matrix elements <a|O|b>, which tells us how strongly the operator O couples two electrons. For the Hamiltonian, these matrix entries are called hopping rates. We can inspect how GRANAD's empirical model for graphene handles them

print(graphene.hopping)

# You see that this is just a dictionary with a single item: a list containing two elements. These correspond to hoppings of the form [onsite, nearest neighbor]. We now come to the Coulomb matrix, which is basically the same. In a mean-field approach, two body operators like the Coulomb interaction are turned into one-body operators (typically, this happens self-consistently in the ground state). 

print(graphene.coulomb)

# You see that the onsite interaction is XXX and the interaction with the third nearest neighbor is XXX etc.

# You may wonder how additional degrees of freedom like spin, angular momentum or relative orientation can be handled that way. GRANAD allows you to set all parameters including these degrees of freedom in a finite structure via the Slater-Koster procedure. This is covered in another tutorial, but for the moment, we shall be content with the simple empirical TB model discussed above.
#
# ### Orbitals 
#
# The easiest way to understand an orbital is to create it ourselves.
# 
my_orbital = Orbital( position = [1.0, 0.0, 0.0], orbital_id = 'A' )
print( my_orbital)
#
# So, an orbital is just a container for a position and an id. In addition, there are two attributes entirely for book keeping: occupation (helps GRANAD tell how many electrons are in the structure) and sublattice id (only relevant for cuts from bulk material).
#
# ### Lists of Orbitals
#
# Speaking of cuts, we can

# define a triangle by its corners
triangle_corners = ...

# get an orbital list
orbitals = cut_orbitals( graphene, triangle_spec )

# as the name suggests, an orbital list can be handled pretty much the same way as a usual Python list

# it can be indexed
print( orbitals[0] )

# it can be sliced and iterated over etc
for orb in orbitals[:10]:
    print( orb.position )

# shift all orbitals up in the z-direction
orbitals.shift( )

# add another layer of graphene
orbitals += cut_orbitals( graphene, triangle_spec )

# inspect the layers
orbitals.layers()

# TODO: set the coupling in the first layer from the Slater-Koster procedure
orbitals.couple_slater_koster( layer = 0 )
# TODO: perform structure relaxation
orbitals.relax()

# show the orbitals
orbitals.show()

# TODO: include a
orbitals.set_coulomb(  )


# get the absorption in the RPA
omegas, absorption = orbitals.rpa_absorption()
plt.plot( omegas, absorption )

# get the absorption via the TD simulation
# TODO: dont return updated stack, just solution, supply initial state as optional argument, supply flag whether to get full solution along the way
omegas, absorption = orbitals.td_absorption()
plt.plot( omegas, absorption )

