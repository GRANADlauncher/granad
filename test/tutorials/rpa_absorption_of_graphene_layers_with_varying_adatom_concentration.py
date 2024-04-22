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

# ## RPA absorption of graphene layers with varying adatom concentration
#
# In this example, we will investigate the RPA absorption of rhomboid graphene when exposed to increasing levels of adsorbed adatoms.
#
# ### Set up the Stack
#
# We define a function to add a single layer to the adatom

# +

import os

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import granad

PLOTTING = False


def add_single_layer(sb, x, y, orb_id, shift=jnp.zeros(3)):

    # geometry
    rhomboid = granad.Rhomboid(x, y)
    graphene = granad.Lattice(
        shape=rhomboid,
        lattice_type=granad.LatticeType.HONEYCOMB,
        lattice_edge=granad.LatticeEdge.ARMCHAIR,
        lattice_constant=2.46,
        shift=shift,
    )
    sb.add(orb_id, graphene)

    # couplings
    hopping_graphene = granad.LatticeCoupling(
        orbital_id1=orb_id,
        orbital_id2=orb_id,
        lattice=graphene,
        couplings=[0j, 2.66],  # list of hopping amplitudes like [onsite, nn, ...]
    )
    sb.set_hopping(hopping_graphene)
    coulomb_graphene = granad.LatticeCoupling(
        orbital_id1=orb_id,
        orbital_id2=orb_id,
        lattice=graphene,
        couplings=[16.522, 8.64, 5.333],
        coupling_function=lambda d: 14.399 / d + 0j,
    )
    sb.set_coulomb(coulomb_graphene)


def add_adatoms(sb, threshold, orb_id, graphene_orb_id, key, hoppings):
    positions = sb.get_positions(graphene_orb_id)
    bounds_up, bounds_low = positions.max(axis=0), positions.min(axis=0)

    check = (
        jax.random.uniform(key, (len(positions),), minval=0.0, maxval=1.0) < threshold
    )
    keys = jax.random.split(key, num=check.size)

    orbitals = []

    # [top, bridge, hollow], for gaussian coupling, the distances should be very dissimilar!
    shifts = jnp.array(
        [[0.0, 0.0, 1.0], [1.42028166 / 2, 0, 1.0], [1.42028166 / 2, 2.46 / 2, 0.5]]
    )
    distances = jnp.linalg.norm(shifts, axis=1)

    for i, pos in enumerate(positions):
        if check[i]:
            adatom_pos = pos + jax.random.choice(keys[i], shifts)

            # skip if out of bounds
            # if jnp.any( jnp.logical_or( adatom_pos > bounds_up , adatom_pos < bounds_low  ) ):
            #     continue

            orbitals.append(granad.Orbital(orb_id, position=adatom_pos))

    sb.orbitals += orbitals

    # onsite hopping
    sb.set_hopping(
        granad.SpotCoupling(orbital_id1=orb_id, orbital_id2=orb_id, coupling=0j)
    )

    # onsite coulomb
    sb.set_coulomb(
        granad.SpotCoupling(
            orbital_id1=orb_id, orbital_id2=orb_id, coupling=16.522 + 0j
        )
    )

    # couplings
    hopping_nt = granad.DistanceCoupling(
        orbital_id1=orb_id,
        orbital_id2=graphene_orb_id,
        coupling_function=granad.gaussian_coupling(1e-2, distances, hoppings),
    )
    sb.set_hopping(hopping_nt)

    coulomb = granad.DistanceCoupling(
        orbital_id1=orb_id,
        orbital_id2=graphene_orb_id,
        coupling_function=lambda d: 16.522 / (jnp.sqrt(d**2 + 1) + 0j),
    )
    sb.set_coulomb(coulomb)
    print(
        "adatoms: ",
        len(sb.get_positions(orb_id)),
        "carbon atoms: ",
        len(sb.get_positions(graphene_orb_id)),
    )


# -

# We now define a function we want to loop over taking in the size of the flake and the adatom concentration.

# +


def rpa_sim(x, y, concentration, t, n_sims=10):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num=n_sims)

    data = []

    for key in keys:
        sb = granad.StackBuilder()
        add_single_layer(sb, x, y, "pz")
        add_adatoms(sb, concentration, "A", "pz", key, [t, t, t])
        stack = sb.get_stack()

        if PLOTTING:
            granad.show_eigenstate2D(stack, name="rhomboid.pdf")

        omegas = jnp.linspace(0, 15, 200)
        polarization = 0
        tau = 5
        coulomb_strength = 1.0
        alpha = granad.rpa_polarizability_function(
            stack=stack,
            tau=tau,
            polarization=polarization,
            coulomb_strength=coulomb_strength,
            hungry=False,
        )
        absorption = jax.lax.map(alpha, omegas).imag * 4 * jnp.pi * omegas
        data.append(absorption)

    avg_absorption = sum(data) / len(data)
    plt.plot(omegas, avg_absorption / jnp.max(avg_absorption), label=f"{concentration}")


# x,y, concentration = 4,4, 0.25 generates one adatom on an 18 atom flake
params = [(20, 20, 0.05), (20, 20, 0.1), (20, 20, 0.2), (20, 20, 0.5)]
for t in jnp.linspace(0, 2, 10) * 2.66:
    for p in params:
        rpa_sim(*p, t)
        plt.legend()
        plt.savefig(f"{t}_adatom_concentrations.pdf")

# -
