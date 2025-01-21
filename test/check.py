import jax.numpy as jnp
from granad import MaterialCatalog, Triangle
from granad import Pulse
import matplotlib.pyplot as plt

def rpa_vs_td(flake, omegas_rpa, name):
    
    # resulting arrays
    res_rpa, res_td = [], []

    # coulomb parameters
    cs =  jnp.linspace(0, 1, 20)
    
    for c in cs:

        ## RPA
        polarizability = flake.get_polarizability_rpa(
            omegas_rpa,
            relaxation_rate = 1/10,
            polarization = 0,
            coulomb_strength = c,
            hungry = 2 )

        absorption_rpa = jnp.abs( polarizability.imag * 4 * jnp.pi * omegas_rpa )
        res_rpa.append(absorption_rpa)

        ## TD
        pulse_complex = Pulse(
            amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
        )
        # make sure illumination is real
        pulse = lambda t : pulse_complex(t).real
        
        result = flake.master_equation(
            expectation_values = [ flake.dipole_operator ],
            end_time=40,
            relaxation_rate=1/10,
            coulomb_strength = c,
            illumination=pulse,
        )

        omega_max = omegas_rpa.max()
        omega_min = omegas_rpa.min()
        p_omega = result.ft_output( omega_max, omega_min )[0]
        omegas_td, pulse_omega = result.ft_illumination( omega_max, omega_min )
        absorption_td = jnp.abs( -omegas_td * jnp.imag( p_omega[:,0] / pulse_omega[:,0] ) )
        res_td.append(absorption_td)

    ## plot comparison
    res_td = jnp.array(res_td).T
    res_rpa = jnp.array(res_rpa).T

    interpolation = None #'sinc'
    plt.imshow( (res_rpa / res_rpa.max())**(1/4) , extent=[cs.min(), cs.max(), omegas_rpa.min(), omegas_rpa.max()],
               origin='lower', aspect='auto', cmap='viridis', interpolation=interpolation)
    
    plt.savefig(f'{name}_rpa.pdf')
    plt.imshow( (res_td / res_td.max())**(1/4), extent=[cs.min(), cs.max(), omegas_td.min(), omegas_td.max()],
               origin='lower', aspect='auto', cmap='viridis', interpolation=interpolation)
    plt.savefig(f'{name}_td.pdf')

if __name__ == '__main__':
    doping = 10
    graphene = MaterialCatalog.get( "graphene" )
    flake = graphene.cut_flake( Triangle(30), plot = False  )
    flake.set_electrons(flake.electrons + doping)
    flake.show_2d(name = 'graphene.pdf')
    omegas_rpa = jnp.linspace(0, 4, 40)
    rpa_vs_td(flake, omegas_rpa, "graphene")

    doping = 10
    flake = MaterialCatalog.get("metal_1d").cut_flake( 70 )
    flake.set_electrons(flake.electrons + doping)
    flake.show_2d(name = 'chain.pdf')
    omegas_rpa = jnp.linspace(0, 4, 40)    
    rpa_vs_td(flake, omegas_rpa, "chain")
