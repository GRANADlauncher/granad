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

# ## Fitting TB-hopping rates to data
#
# In this example, we will show how to fit TB-hoppping rates to reproduce external data.
#
# ### Define functions
#
# The scenario we are considering is the following: we will initialize a stack with random hoppings and get its ground state energy. Then, we try to construct another stack to reproduce the correct ground state energy by adjusting its hoppings via gradient descent.
#
# First, we will do the necessary imports and create the random stack.

# +

import granad
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

    
# define the target stack

def stack_chain( hopping ):
    """constructs a chain of 10 atom with the specified nearest neighbour hopping"""
    positions = [ [float(i), 0, 0]  for i in range(10) ]
    sb = granad.StackBuilder()
    sb.orbitals += [ granad.Orbital( orbital_id = 'orb', position = p) for p in positions ]

    # couplings
    hopping_nt = granad.DistanceCoupling(
        orbital_id1="orb", orbital_id2="orb", coupling_function = granad.gaussian_coupling( 1e-1, [1.0], [hopping]  )
        )                        
    sb.set_hopping(hopping_nt)                        
    stack = sb.get_stack( energies_only = True, pingback = False )
    return stack

target_stack = stack_chain( 2.0 )

# -

# We define a function to compute the ground state energy of a stack

# +


def ground_state_energy( stack ):
    return stack.energies @ stack.rho_0.diagonal()                        

# -

# For optimization, we will:
# 1. initialize a stack with random hoppings
# 2. compute its ground state energy
# 3. take the gradient for the hoppings
# 4. adjust the hoppings according to gradient
# 5. build a new stack and go to 2.

# +
                                                
def target( hopping ):    
    return (ground_state_energy( stack_chain(hopping) ) - ground_state_energy( target_stack ))**2

grad = jax.grad(target)

# Initialize a random seed
key = jax.random.PRNGKey(0)
# Generate a random number
hopping = jax.random.uniform(key, minval=0.0, maxval=1.0)

# compute and print gradient
grad_val = grad(hopping)
print( grad_val )

# gradient descent for 100 steps with a rate of 1e-1
for i in range( 100 ):
    if abs(grad_val) < 0.1:
        print(f'Converged to {hopping}')
        break
    grad_val = grad( hopping )
    hopping -= 1e-1 * grad_val
    print(hopping)
    
# -

# After convergence, we can now inspect the energy spectra

# +

granad.show_energies( stack_chain(hopping) )
granad.show_energies( target_stack )

# -
