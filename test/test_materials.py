from granad import *
from granad._numerics import *
import jax.numpy as jnp

def test_all():
    for m in Material._materials.keys():
        mat = Material.get(m)
        orbs = mat.cut_orbitals(9*Shapes.triangle_rotated, plot = False)
        assert len(orbs) > 0
