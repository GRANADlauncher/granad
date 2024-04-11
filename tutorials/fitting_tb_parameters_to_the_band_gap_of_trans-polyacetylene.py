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

# ## Fitting TB parameters to the band gap of trans-Polyacetylene
#
# In this example, we will optimize the  TB parameters of polyacetylene by a gradient descent fit to the trans-polyacetylene band gap, reported as 1.8eV in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.41.12845
#
# ### Define functions
#
# First, we will do the necessary imports and define a function to create a polyacetylene chain from a set of parameters. These parameters are contained in a JAX array and will be optimized.
# They look like [alpha * equilibrium_displacement, t_0] = [geometry parameter, nn hopping without dimerization]. The hoppings are computed according to t_0 \pm 2 * alpha * K as in the original SSH paper. This is a cheep way to incorporate "geometry optimization". A more full-fledged fit would involve, e.g. defining spring constants and minimizing the total energy E_tot = E_el + E_springs, as also done in the original SSH paper.

# +

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import granad


def stack_poly(params, n=180):
    """constructs a polyacetylene chain with specified parameters"""

    # this constructs a topologically trivial dimerized "grid" to identify neighbors. The actual "geometry" is contained in the parameters.
    displacement = 0.2
    positions = jnp.arange(n) - displacement * (jnp.arange(n) % 2) + 3
    distances = [1.0 - displacement, 1.0 + displacement]

    geom, t_0 = params

    sb = granad.StackBuilder()
    sb.orbitals += [
        granad.Orbital(orbital_id="pi", position=[p, 0, 0]) for p in positions
    ]

    hoppings = [t_0 + 2 * geom, t_0 - 2 * geom]
    cf = granad.gaussian_coupling(1e-1, distances, hoppings)

    # couplings
    hopping_nt = granad.DistanceCoupling(
        orbital_id1="pi", orbital_id2="pi", coupling_function=cf
    )
    sb.set_hopping(hopping_nt)
    stack = sb.get_stack(energies_only=True, pingback=False)
    return stack


# -

# We define a function to compute the "band gap" of trans-polyacetylene as the HOMO-LUMO gap

# +


def gap(stack):
    occ = jnp.argwhere((stack.rho_0.real.diagonal() > 0))[-1][0]
    return stack.energies[occ + 1] - stack.energies[occ]


# -

# For optimization, we will:
#
# 1. initialize a stack with random hoppings
# 2. compute its ground state energy
# 3. take the gradient for the hoppings
# 4. adjust the hoppings according to gradient
# 5. build a new stack and go to 2.

# +

trans_band_gap = 1.8


def target(params):
    return (gap(stack_poly(params)) - trans_band_gap) ** 2


grad = jax.grad(target)

# Initialize a random seed
key = jax.random.PRNGKey(3)

# Generate random initial values
params = jax.random.uniform(key, (2,), minval=1.0, maxval=4.0)

# initial gradient
grad_val = grad(params)
print(grad_val)

# gradient descent for 100 steps with a rate of 1e-2
for i in range(100):
    if target(params) < 1e-4:
        print(
            f"Converged to alpha*u, t_0 = {params} with gap energy {gap(stack_poly(params))}"
        )
        break
    grad_val = grad(params)
    params -= 1e-2 * grad_val
    print(params)

# -

# After convergence, we can inspect the energy spectrum and compare it to the one obtained from typical parameter choices

# +

granad.show_energies(stack_poly(params))
granad.show_energies(stack_poly([0.2, 2.5]))

# -

# We see that we retain deviations, owing to the fact that we fitted to a single number. Ideally, one would fit to an energy spectrum.
