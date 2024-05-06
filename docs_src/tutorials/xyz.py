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

# # Exporting and Importing
#

# GRANAD lets you save orbitals to xyz files. It also offers rudimentary support for loading orbitals from xyz files. For demonstration, we will create a triangular flake, print its xyz representation, save it to file and reload it.

# +
from granad import MaterialCatalog, Triangle, OrbitalList
graphene = MaterialCatalog.get( "graphene" )
flake = graphene.cut_flake( Triangle(15)  )
# -

# We can look at only the atoms in this file. They are given as a dictionary. The type of atom is the key. The value is all positions where atoms of this type are.

# +
print(flake.atoms)
# -

# If you supply no name to the to_xyz method, it will just return the string

# + 
print(flake.to_xyz())
# -

# If you want to save this to a file, do

# +
flake.to_xyz('flake.xyz')
# -

# You can also reload, although this is limited: all atoms in the xyz file get the same group id by default.

# +
new_flake = OrbitalList.from_xyz('flake.xyz')
assert new_flake.atoms == flake.atoms
# -
