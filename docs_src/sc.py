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

# ## Self-consistency procedure
#
# Density matrices at zero temperature are constructed according to the aufbau principle: if there are $2N$ energy levels and $2N$ electrons, then the lowest $N$ energy levels will get occupied by two electrons each by default. If electrons are added / removed, the corresponding real space charge distribution will not be at equilibrium anymore. To correct for this effect, GRANAD offers a self-consistency procedure we will detail in this tutorial.
#

### Self-Consistency
#
# We set up a flake as usual and inspect its charge distribution

# +
from granad import Material2D, Hexagon
flake = Material2D.get("graphene").cut_orbitals( Hexagon(20) )
flake.show_2d()
# -

# We now add two electrons to the flake

# +
flake.electrons += 2
flake.show_2d()
# -

# We make the flake self-consistent, if we specify parameters for the SC procedure

# +
flake.make_self_consistent( {"iterations" : 100, "mix" : 0.05, accuracy : 0.05} )
flake.show_2d()
# -
