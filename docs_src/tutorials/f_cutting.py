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

# # Cutting
#

### Cutting
#
# GRANAD offers different materials. They can be defined in different dimensions. The process by which a finite flake is cut from an "infinite" bulk differs by dimensionality. This process is run by calling the `cut_flake` method. To see why, let's inspect

# +
from granad import MaterialCatalog
ssh = MaterialCatalog.get("ssh")
help(ssh.cut_flake)
# -

# So, the cut_flake method is automatically determined. Let's look at the 1D case

# +
from granad.materials import cut_flake_1d
help(cut_flake_1d)
# -

# We need to specify the number of unit cells. 

# +
flake = ssh.cut_flake( unit_cells = 40, plot = False)
# -

# You may notice this is the configuration without edge states in the band gap

# +
flake.show_energies()
# -

# If you don't want this, delete the edges by removing the first and the last orbital in the list

# +
del flake[0]
del flake[-1]
flake.show_2d()
flake.show_energies()
# -

# We now cover cutting in 2D.

# +
from granad.materials import cut_flake_2d
help(cut_flake_2d)
# -

# This is more complex. We can give an arbitrary polygon to the cutting function, so let's do this by approximating a potato.

# +
import jax.numpy as jnp
potato = 10 * jnp.array( [
  (3, 1),    # Bottom center (widest point)
  (2, 2),    # Lower left bulge
  (1, 3),    # Mid left indent
  (2, 4),    # Upper left bulge
  (3, 5),    # Top center
  (4, 4),    # Upper right bulge
  (5, 3),    # Mid right indent
  (4, 2),    # Lower right bulge
  (3, 1)     # Connect back to the bottom center
])
# -

# Now, we cut a flake

# +
graphene = MaterialCatalog.get("graphene")
flake = graphene.cut_flake( potato, plot = True )
# -

#
### Shapes
#
# Built-in shapes cover Triangles, Hexagons, Rectangles, (approximate) Circles and Parallelograms. Specialized to hexagonal lattices, they can be cut with zigzag, armchair or bearded edges. They are implemented as functions returning a set of vertices. All parameters you pass to them are in Angström.

# +
from granad import Rectangle
help(Rectangle)
# -

# Note: The "extent" of the shape refers to its minimum dimensions. For instance, when working with graphene, which has a bond length of approximately 1.42 Å, you should specify dimensions that avoid cutting into very small fragments, such as a single benzene ring (~2.46 Å in diameter). For meaningful simulations, ensure that the shape dimensions exceeds this scale.

### Bearded configurations

# Cutting removes "dangling" atoms by default. Dangling atoms are defined by their neighbor number: if they have only one neighbor, they are removed. If you want to deactivate this to keep "bearded" configurations, do

# +
flake = graphene.cut_flake( Rectangle(10,12), plot = True, minimum_neighbor_number = 0 )
# -
