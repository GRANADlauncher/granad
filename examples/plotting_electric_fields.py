import granad

import jax.numpy as jnp

electric_field = granad.electric_field(
    [0, 1, 0],
    1,
    jnp.array([1, 0, 0]),
)

granad.show_electric_field_time(jnp.linspace(0, 10, 100), electric_field)
