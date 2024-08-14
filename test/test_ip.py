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
