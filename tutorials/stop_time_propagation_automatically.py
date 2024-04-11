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

# ## Stop time propagation automatically
#
# In the previous examples, we used the time propagation in a by presciently picking the correct end point of the time evolution. Here we cover how to do this automatically.
#
# ### Set up the simulation
#
# We set up the simulation as usual: we take a graphene flake and specify its couplings.

import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# +
import granad

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
    orbital_id1="pz",
    orbital_id2="pz",
    lattice=graphene,
    couplings=[0, -2.66],  # list of hopping amplitudes like [onsite, nn, ...]
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

amplitudes = [1e-5, 0, 0]
frequency = 1
field_func = granad.electric_field(amplitudes, frequency, stack.positions[0, :])

amplitudes = [1e-5, 0, 0]
frequency = 1
ramp_duration = 2
time_ramp = 0.0
field_func = granad.electric_field_with_ramp_up(
    amplitudes, frequency, stack.positions[0, :], ramp_duration, time_ramp
)

loss_function = granad.relaxation(5)

# -

# We are now done with the usual house keeping. Now we start the time propagation. In contrast to the previous simulation, we deliberately pick a too short end point

# +

time_axis = jnp.linspace(0, 4, int(1e5))
saveat = time_axis[::100]
stack, sol = granad.evolution(
    stack,
    time_axis,
    field_func,
    loss_function,
    saveat=saveat,
    stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10),
)
occupations = jnp.diagonal(sol.ys, axis1=1, axis2=2)
dipole_moment = granad.induced_dipole_moment(stack, occupations)

# -

# We inspect the induced dipole moments in time domain

# +

plt.plot(saveat, dipole_moment)
plt.show()

# -

# This does not look complete: we clearly see that the dipole moments have not reached steady states yet. One way to overcome this problem is to resume the simulation by simply propagating the new stack in time.

# +

time_axis_old = time_axis
saveat_old = saveat
dipole_moment_old = dipole_moment

time_axis = jnp.linspace(4, 8, int(1e5))
saveat = time_axis[::100]
stack, sol = granad.evolution(
    stack,
    time_axis,
    field_func,
    loss_function,
    saveat=saveat,
    stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10),
)
occupations = jnp.diagonal(sol.ys, axis1=1, axis2=2)
dipole_moment = granad.induced_dipole_moment(stack, occupations)

saveat = jnp.concatenate([saveat_old, saveat])
dipole_moment = jnp.concatenate([dipole_moment_old, dipole_moment])
plt.plot(saveat, dipole_moment)
plt.show()


# -

# However, we have still not reached a steady state. We can in principle wrap the code above in a loop and execute it until we visually see the oscillations stabilize. GRANAD provides an automatic way to do this with a specialized function that takes a signal (say, the x-components of the dipole moment) and crudely estimates the percentage of it that is periodic

# +

print(granad.fraction_periodic(dipole_moment[:, 0]))

# -

# GRANAD tells us that approximately 7% of our signal is periodic. If you don't think this matches up, you can play with the additional "threshold" parameter. We will simply automate the simulation as follows: as long as the periodic fraction is below 70% and a certain maximum time is not surpassed, we will continue the simulation

# +

time_max = 8
while granad.fraction_periodic(dipole_moment[:, 0]) < 0.7 and time_axis.max() < 400:

    time_axis_old = time_axis
    saveat_old = saveat
    dipole_moment_old = dipole_moment

    time_min = time_max
    time_max += 20

    time_axis = jnp.linspace(time_min, time_max, int(1e5))
    saveat = time_axis[::100]
    stack, sol = granad.evolution(
        stack,
        time_axis,
        field_func,
        loss_function,
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10),
    )
    occupations = jnp.diagonal(sol.ys, axis1=1, axis2=2)
    dipole_moment = granad.induced_dipole_moment(stack, occupations)

    saveat = jnp.concatenate([saveat_old, saveat])
    dipole_moment = jnp.concatenate([dipole_moment_old, dipole_moment])

print(
    f"Finished simulation at {time_max} with {granad.fraction_periodic( dipole_moment[:,0]  ) }% periodic!"
)

# -

# We check to see whether this has worked. Indeed, the dipole moments have stabilized!

# +

plt.plot(saveat, dipole_moment)
plt.show()

# -
