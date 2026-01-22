from granad import *

def get_td(flake, sigma, dipole_moment, source_location, omega_max, omega_min):

    # dipole potential
    DP_func=DipolePulse(
        dipole_moment=dipole_moment,
        source_location=source_location,
        omega=1,
        sigma=sigma
    )
    hamiltonian_DP_field=flake.get_hamiltonian()
    hamiltonian_DP_field["dipole_kick"]=DP_func

    # integrate
    result = flake.master_equation(
        end_time = 40,
        hamiltonian = hamiltonian_DP_field,
        expectation_values = [flake.dipole_operator],
        # density_matrix=['diag_x'],
        relaxation_rate = 1/10,
        solver = "RK45",
        max_mem_gb=50
    )

    p_omega = result.ft_output( omega_max, omega_min )[0]
    omegas_td, pulse_omega = result.ft_illumination( omega_max, omega_min )
    # alpha = p_omega / pulse_omega[:,0]    

    return p_omega, omegas_td, result


def potential(pos, w, sigma, dipole_moment, source_location):
    def f(w):
        return 1 / jnp.sqrt(2 * sigma**2) * (jnp.exp(-0.25 * sigma**2 * (w - 1)**2) + jnp.exp(-0.25 * sigma**2 * (w + 1)**2)  )

    loc = jnp.array( source_location )[:,None]
    dip = jnp.array( dipole_moment )

    distances = pos - loc
    r_term = 14.39*(dip @ distances) / jnp.linalg.norm( distances, axis = 0 )**3

    return jnp.nan_to_num(r_term) * f(w)[:, None]

## SETUP

# get material
graphene = MaterialCatalog.get( "graphene" )

# cut a 15 Angström wide triangle from the lattice (can also be an arbitrary polygon)
flake = graphene.cut_flake(Triangle(15))

source_location = flake.positions[0, :] + jnp.array([0, 0, 0.1])
sigma, dipole_moment = 0.1, [1, 0, 0]

# td
pol_td, omegas, result = get_td(flake, sigma, dipole_moment, source_location, 4, 0)

# rpa
pot = lambda w : potential(flake.positions.T, w, sigma, dipole_moment, source_location)

sus_rpa = flake.get_susceptibility_rpa(
    omegas,
    relaxation_rate = 1/10,
    hungry = 1,
    coulomb_strength = 1.
)
pol_rpa = jnp.einsum("ni, wm, wnm -> wi", flake.positions, pot(omegas), sus_rpa)

## Plotting
def norm(s):
    s = jnp.abs(s)
    return s / s.max()

cutoff, i = 0, 1
fig, axs = plt.subplots(2,1)
axs[0].plot(omegas[cutoff:], norm(pol_rpa[cutoff:, i].imag))
axs[0].plot(omegas[cutoff:], norm(pol_td[cutoff:, i].imag), '--')
axs[1].plot(omegas[cutoff:], norm(pol_rpa[cutoff:, i].real))
axs[1].plot(omegas[cutoff:], norm(pol_td[cutoff:, i].real), '--') 
plt.show()
