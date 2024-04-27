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

# # Defining Materials
#
# We talk about how to modify built-in materials and define custom ones.

### Defining a custom material
#
# Let's first look at the Material class itself

# +
from granad import MaterialCatalog, Material, Hexagon
print(Material.__doc__)
# -

# So, the Material class essentially defines a small language we can use to specify a material. The Hubbard model would look like this

# +
t = 1. # nearest-neighbor hopping
U = 0.1 # onsite coulomb repulsion
hubbard = (
    Material("Hubbard")
    .lattice_constant(1.0)
    .lattice_basis([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    .add_orbital_species("up", s = -1)
    .add_orbital_species("down", s = 1)
    .add_orbital(position=(0, 0, 0), species = "up")
    .add_orbital(position=(0, 0, 0), species = "down")   
    .add_interaction(
        "hopping",
        participants=("up", "up"),
        parameters=[0.0, t],
    )
    .add_interaction(
        "hopping",
        participants=("down", "down"),
        parameters=[0.0, t],
    )
    .add_interaction(
        "coulomb",
        participants=("up", "down"),
        parameters=[U]
        )
)
# -

# To see how cutting finite flakes from this 3D material works, let's inspect

# +
help(hubbard.cut_flake)
# -

# So, the cut_flake method is automatically determined. We have specified a 3D material, so let's look at the function that is applied

# +
from granad.materials import cut_flake_generic
help(cut_flake_generic)
# -

# Let's see this in action

# +
flake = hubbard.cut_flake( [(0,3), (0,3), (0,3)] ) # 3*3*3 = 27 unit cells, 3 in every direction, each hosting spin up and spin down atom
flake.show_3d()
# -

### Modifying existing materials

# You can just copy the material you want to modify and change/override its attributes. As an example, we will turn our ordinary graphene model into a variant of the Haldane model by introducing complex nnn hoppings

# +
from copy import deepcopy
graphene  = MaterialCatalog.get("graphene")
graphene_haldane = deepcopy(graphene)
graphene_haldane.add_interaction("hopping", participants = ('pz', 'pz'), parameters = [0, 1.0, 1j*0.1])
print(graphene_haldane)
# -

# The Haldane model breaks inversion symmetry explicity by a staggered onsite potential. There is no (nice) way to achieve this with a few modifications from the normal graphene model, so we simply use the versatile properties of the orbital list datatype when we cut finite flakes

# +
hexagon =  Hexagon(30, armchair = False)
flake_topological = graphene_haldane.cut_flake(hexagon, plot = True )
delta = 0.3
for orb_1 in [orb for orb in flake_topological if orb.tag == 'sublattice_1']:
    flake_topological.set_hamiltonian_element(orb_1, orb_1, delta)    
# -

# We now display the edge state

# +
import jax.numpy as jnp
idx = jnp.argwhere(jnp.abs(flake_topological.energies) < 1e-2)[0].item()
flake_topological.show_2d( display = flake_topological.eigenvectors[:, idx]  )
# -


### Handling non-periodic dimensions

# Say you want to have a custom ssh chain with 2 atoms in the unit cell, but displace one of the atoms in y-direction. You do this by including a second basis vector in y-direction and give the atom you want to displace a non-vanishing y-coordinate. To not get a 2D lattice, you have to specify explicitly which direction(s) you want to be periodically repeated.

# +
_ssh_modified = (
    Material("ssh")
    .lattice_constant(2.46)
    .lattice_basis([
        [1, 0, 0],
        [0, 1, 0],
    ],
    periodic = [0]) # THIS IS THE LINE    
    .add_orbital_species("pz", l=1, atom='C')
    .add_orbital(position=(0,0), tag="sublattice_1", species="pz")
    .add_orbital(position=(0.8,0.1), tag="sublattice_2", species="pz")
    .add_interaction(
        "hopping",
        participants=("pz", "pz"),
        parameters=[0.0, 1 + 0.2, 1 - 0.2],
    )
    .add_interaction(
        "coulomb",
        participants=("pz", "pz"),
        parameters=[16.522, 8.64, 5.333],
        expression=lambda d: 14.399 / d
    )
)
flake = _ssh_modified.cut_flake( unit_cells = 10)
flake.show_2d()
flake.show_energies()
# -


