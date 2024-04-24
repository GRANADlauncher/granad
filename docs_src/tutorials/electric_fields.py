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

# # Electric Fields
#
# We present how to handle electric fields

#
# The built-in electric fields are just callables, dependent on time.
#

# +
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from granad import Wave
help(Wave)
# -

# So, calling "Wave" gives back a function we can evaluate at single points in time

# +
wave = Wave( [1, 0, 0], 1  )
print(wave(0))
# -

# A quick way to visualize them is to plot their real and imaginary part. JAX offers the vmap function that vectorizes the application.

# +
time = jnp.linspace(0, 2 * 2 * jnp.pi, 100)
e_field = jax.vmap( wave ) (time)
print(e_field.shape)
# -

# +
plt.plot(time, e_field.real)
plt.plot(time, e_field.imag, '--')
plt.show()
# -
