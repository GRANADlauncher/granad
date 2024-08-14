from granad import *

def test_oam():
    flake = MaterialCatalog.get("graphene").cut_flake(Triangle(10))
    # planar structure
    assert jnp.all(0 == flake.oam_operator[:2])
    assert jnp.all(flake.oam_operator[2] == flake.dipole_operator[0] @ flake.velocity_operator[1] - flake.dipole_operator[1] @ flake.velocity_operator[0])
