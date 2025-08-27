from granad import *
import inspect

from granad import *

def myfun(a, b):
    d = jnp.linalg.norm(a-b)
    return jax.lax.cond(jnp.abs(d - 1.42028166) < 1e-5, lambda x : -2.66, lambda x : 0., d)
    
h1 = get_graphene().cut_flake(Triangle(18)).hamiltonian
flake = get_graphene().cut_flake(Triangle(18))
flake.set_hamiltonian_groups(flake[0], flake[0], myfun)

assert jnp.linalg.norm(h1 - flake.hamiltonian) == 0., "Failed"
