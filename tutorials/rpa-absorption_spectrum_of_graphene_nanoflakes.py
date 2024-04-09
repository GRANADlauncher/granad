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

# ## RPA-absorption spectrum of graphene nanoflakes
#
# This example demonstrates an advanced simulation. We will initialize a triangular graphene nanoflake and compute its TB absorption spectrum within the RPA. This is a frequency-domain simulation, so no need for time propagation.
#
# ### Set up the Stack
#
# The setup is analogous to the first tutorial. We build the Stack using the StackBuilder, which needs to know the material (a triangular armchair graphene nanoflake of 7.4 Å) and the coupling between the pz-orbitals (hopping rates and Coulomb interaction).

# +
import granad
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import matplotlib.pyplot as plt

sb = granad.StackBuilder()

# geometry
triangle = granad.Triangle(7.4) 
graphene = granad.Lattice(
    shape=triangle,
    lattice_type=granad.LatticeType.HONEYCOMB,
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_constant=2.46,
)
sb.add("pz", graphene)

# couplings
hopping_graphene = granad.LatticeCoupling(
    orbital_id1="pz", orbital_id2="pz", lattice=graphene, couplings=[0, -2.66] # list of hopping amplitudes like [onsite, nn, ...]
)
sb.set_hopping(hopping_graphene)
coulomb_graphene = granad.LatticeCoupling(
    orbital_id1="pz",
    orbital_id2="pz",
    lattice=graphene,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
sb.set_coulomb(coulomb_graphene)

stack = sb.get_stack()
# -

# We now compute the RPA absorption for an x-polarized uniform electric field. We do so by first computing the polarizability and then aking its imaginary part. Similar to the previous function, we have to tell GRANAD some parameters: a phenomennological broadening, coulomb strength scaling factor. In addition, we make it fast, but take a lot of memory with hungry = True. 

# +
omegas = jnp.linspace(0, 20, 200)
polarization = 0
tau = 0.1
coulomb_strength = 1.0
alpha = granad.rpa_polarizability_function( stack = stack, tau = tau, polarization = polarization, coulomb_strength=coulomb_strength, hungry = True )
absorption = jax.lax.map( alpha, omegas ).imag * 4 * jnp.pi * omegas
plt.plot( omegas, absorption )
plt.show()
# -
