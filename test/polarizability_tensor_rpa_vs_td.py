from granad import *

def get_polarizability_td(flake,
                       amplitude,
                       frequency,
                       peak,
                       fwhm,
                       omega_max,
                       omega_min,
                        rtol,
                        atol,
                       **kwargs
                       ):
    
    amplitudes = [ [amplitude, 0, 0], [0, amplitude, 0] ]
    fields, pulses, ret = [], [], []
    
    for i, a in enumerate(amplitudes):
        
        result = flake.master_equation(
            expectation_values = [flake.dipole_operator],
            end_time=40,
            illumination=Pulse(amplitudes = a, frequency = frequency, peak = peak, fwhm = fwhm),
            stepsize_controller = diffrax.PIDController(rtol=rtol,atol=atol),
            **kwargs
        )


        p_omega = result.ft_output(omega_max, omega_min)[0]
        omegas, pulse_omega = result.ft_illumination(omega_max, omega_min)

        # issue: errors can accumulate in wrong dimension
        p_omega = p_omega * (jnp.abs(p_omega).max(axis=0) > atol * 1e2)

        fields.append(p_omega[:, :2])
        pulses.append(pulse_omega[:, i])
        ret.append(p_omega[:, :2] / pulse_omega[:, i][:, None])

    return omegas, jnp.stack(ret, axis = 1), jnp.stack(fields, axis = 1), jnp.stack(pulses, axis = 1)
    

# get material
graphene = MaterialCatalog.get( "graphene" )

# cut a 15 Angström wide triangle from the lattice (can also be an arbitrary polygon)
flake = graphene.cut_flake(Triangle(15))

## TD SIM
omegas, pol_td, fields, pulses = get_polarizability_td(flake,
                                                       1e-5,
                                                       frequency = 1,
                                                       peak = 5,
                                                       fwhm = 2,
                                                       omega_max = 6,
                                                       omega_min = 0,
                                                       relaxation_rate = 1/10,
                                                       rtol = 1e-12,
                                                       atol = 1e-12,
                                                       solver = diffrax.Dopri5()
                                                       )
## COMPARE TO RPA
sus_rpa = flake.get_susceptibility_rpa(
    omegas,
    relaxation_rate = 1/10,
    hungry = 1,
    coulomb_strength = 1.
)
pol_rpa = jnp.einsum("ni, mj, wnm -> wij", flake.positions, flake.positions, sus_rpa)

## Plotting
cutoff, i, j = 0, 0, 0
fig, axs = plt.subplots(2,1)
axs[0].plot(omegas[cutoff:], pol_rpa[cutoff:, i, j].imag)
axs[0].plot(omegas[cutoff:], pol_td[cutoff:, i, j].imag, '--')
axs[1].plot(omegas[cutoff:], pol_rpa[cutoff:, i, j].real)
axs[1].plot(omegas[cutoff:], -pol_td[cutoff:, i, j].real, '--') # real part is flipped => likely FFT issue
plt.show()
