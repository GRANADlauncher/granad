import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from granad.materials import get_graphene
from granad import MaterialCatalog
from granad import Pulse
from granad import Triangle
import diffrax

triangle = Triangle(40,armchair=True)
flake=get_graphene(hopping=-2.66).cut_flake(triangle,plot=True)
print(flake)
flake.show_energies()

pulse= Pulse(
    amplitudes=[1e-5, 0, 0], frequency=2, peak=2, fwhm=0.5)

operators = [flake.dipole_operator]
result = flake.master_equation(
    dt=1e-5,
    relaxation_rate = 0.05,
    illumination = pulse,
    expectation_values = operators,
    grid = 10,
    end_time = 150,
    solver=diffrax.Dopri8(),
    stepsize_controller = diffrax.PIDController(rtol=1e-10, atol=1e-10),
    coulomb_strength=1,
    max_mem_gb=32
) 