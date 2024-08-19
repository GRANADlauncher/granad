import jax.numpy as jnp

from granad import *

def test_ip_greens_function():
    flake = MaterialCatalog.get("graphene").cut_flake(Triangle(10))
    dip_x = flake.dipole_operator_e[0]
    omegas = jnp.linspace(0, 4, 20)
    ip = flake.get_ip_green_function(dip_x, dip_x, omegas)
    ref = flake.get_polarizability_rpa(omegas, 0, 0., 0.1)
    ip = jnp.abs(ip)
    ref = jnp.abs(ref)
    assert jnp.allclose( ip / ip.max(), ref / ref.max(), rtol = 0.1)

def test_ip_greens_function_masked():    
    flake = MaterialCatalog.get("ssh").cut_flake( unit_cells = 40, plot = False)
    dip_x = flake.dipole_operator_e[0]
    omegas = jnp.linspace(0, 4, 20)
    trivial = jnp.abs(flake.energies) > 1e-1
    mask = jnp.logical_and(trivial[:, None], trivial)

    # only topological ip response
    ip_topo = flake.get_ip_green_function(dip_x, dip_x, omegas, mask = mask)
    assert jnp.all(jnp.abs(ip_topo) != 0)

    # topologically trivial ssh model has no states at zero energies
    del flake[0]
    del flake[-1]
    dip_x = flake.dipole_operator_e[0]
    trivial = jnp.abs(flake.energies) > 1e-1
    mask = jnp.logical_and(trivial[:, None], trivial)
    ip = flake.get_ip_green_function(dip_x, dip_x, omegas, mask = mask)
    assert jnp.all(jnp.abs(ip) == 0)
