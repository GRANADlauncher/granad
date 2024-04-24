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

# # RPA simulation options
#

# We will illustrate the RPA simulation options by executing the same simulation for all parameters.

# First, we set up the RPA simulation. We will consider a small triangle.

# +
import time
import jax.numpy as jnp
from granad import MaterialCatalog, Triangle

# get material
graphene = MaterialCatalog.get( "graphene" )
flake = graphene.cut_flake( Triangle(10)  ) 
print(flake)
# -

# We now come to the simulation. 

# +
def time_option( hungry ):
    omegas_rpa = jnp.linspace( 0, 5, 10 )
    start = time.time()
    polarizability = flake.get_polarizability_rpa(
        omegas_rpa,
        relaxation_rate = 1/10,
        polarization = 0, 
        hungry = hungry )
    return time.time() - start
# -

# +
for hungry in range(3):
    print( f'{hungry} {time_option(hungry)}' )
# -
