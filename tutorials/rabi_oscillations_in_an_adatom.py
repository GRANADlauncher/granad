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

# ## Rabi Oscillations in an Adatom
#
# This example demonstrates the simulation of a single adatom involving a dipole transition between its energy levels.
#
# ### Set up the StackBuilder
#
# The setup is analogous to the previous tutorials. We need two adatom levels, called "A" and "B" at the same spot. Both orbitals should togehter host a single electron, so we need to explicitly mention a vanishing occupation in one of them.

import jax.numpy as jnp
import matplotlib.pyplot as plt

# +
import granad

sb = granad.StackBuilder()
spot = granad.Spot(position=[0.0, 0.0, 0.0])
sb.add("A", spot)
sb.add("B", spot, occupation=0)
# -

# We now include energies and coulomb interactions. We want the orbitals to be connected by a dipole transition moment. NOTE THAT THIS HAS CHANGED: the transitions are now an attribute of the stack.

# +

# onsite hopping
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_hopping(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=2))

# onsite coulomb
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=1))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=1))

# -

# Now, we visualize the initial state.

stack = sb.get_stack(from_state=0, to_state=1, transitions={("A", "B"): [1.0, 0, 0.0]})
# granad.show_energies(stack)

# Now, we simulate its dynamics. We want to model an external x-polarized field.

# +
amplitudes = [1, 0, 0]
frequency = max(stack.energies) - min(stack.energies)
field_func = granad.electric_field(amplitudes, frequency, stack.positions[0, :])

# -

# The time evolution happens as usual (we again compare the old and new version).

# +

# propagate in time
time_axis = jnp.linspace(0, 10, 10**4)

# we can also omit the saveat argument to save the entire time axis
new_stack, sol = granad.evolution(
    stack,
    time_axis,
    field_func,
)

density_matrices_new = jnp.einsum(
    "ijk,kl,mj->iml", sol.ys, stack.eigenvectors, stack.eigenvectors.conj()
)
energy_occupations_new = jnp.diagonal(density_matrices_new, axis1=1, axis2=2)

# propagate in time
time_axis = jnp.linspace(0, 10, 10**4)
_, energy_occupations_old = granad.evolution_old(
    stack,
    time_axis,
    field_func,
    postprocess=lambda r: granad.to_energy_basis(stack, r).diagonal(),
)

plt.plot(time_axis, energy_occupations_new.real, "--", label="new")
plt.plot(time_axis, energy_occupations_old.real, label="old")
plt.legend()
plt.show()

# -

# We see that the results dont quite line up, but why? We have to remember that the old solver is RK first order, so not too precise.
# Incresing the step size fixes the problem.

# +

# propagate in time
time_axis_high_res = jnp.linspace(0, 10, 10**5)
_, energy_occupations_old = granad.evolution_old(
    stack,
    time_axis_high_res,
    field_func,
    postprocess=lambda r: granad.to_energy_basis(stack, r).diagonal(),
)

plt.plot(time_axis, energy_occupations_new.real, "--", label="new")
plt.plot(time_axis_high_res, energy_occupations_old.real, label="old")
plt.legend()
plt.show()
