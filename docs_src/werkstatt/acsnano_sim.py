import granad
import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import time
import os

DIR = "/scratch/local/ddams/acsnano/"

def get_fourier_transform(time_series, signal):
    signal_freq = jnp.fft.fft(signal) / len(time_series)
    freq_axis = 2 * jnp.pi * len(time_series) / jnp.max(time_series) * jnp.fft.fftfreq(signal_freq.shape[-1])
    return freq_axis, signal_freq

def load_reference_data():
    data = jnp.array(np.genfromtxt('acsnano.csv', delimiter=';'))
    h = 6.6261e-34
    c = 2.9979e8
    frequencies = h * c / (data[:, 0] * 1e-9) / (1.6e-19)
    valid_indices = jnp.argwhere(data[:, 0] < 800)
    return frequencies[valid_indices][:, 0], data[valid_indices, 1] / jnp.max(data[valid_indices, 1])

def plot_fourier_transformed_data(filename, interpolate=False):
    data = jnp.load(f'{DIR}{filename}.npz')
    dipole_moment, electric_field, saveat = data['dipole_moment'], data['electric_field'], data['saveat']
    omegas_ref, ref_data = load_reference_data()
    
    for component in range(dipole_moment.shape[1]):
        dip = jnp.interp(jnp.linspace(saveat.min(), saveat.max(), 2 * saveat.size), saveat, dipole_moment[:, component]) if interpolate else dipole_moment[:, component]
        omega_axis, dipole_omega = get_fourier_transform(saveat, dip)
        _, field_omega = get_fourier_transform(saveat, electric_field[:, component])
        idxs = jnp.argwhere((omegas_ref.min() < omega_axis) & (omega_axis < omegas_ref.max()))
        omega_axis, dipole_omega, field_omega = omega_axis[idxs], dipole_omega[idxs], field_omega[idxs]
        
        polarizability = dipole_omega / field_omega
        spectrum = -omega_axis * jnp.imag(polarizability)
        plt.plot(omega_axis, spectrum / jnp.max(jnp.abs(spectrum)), '-', label='TD')
        plt.plot(omegas_ref, ref_data, '--', label='Reference')
        plt.xlabel(r"$E$ [eV]", fontsize=20)
        plt.ylabel(r"$\sigma$", fontsize=25)
        plt.legend()
        plt.savefig(f"absorption_{filename}_{component}.pdf")
        plt.close()

def plot_time_domain_data(filename):
    data = jnp.load(f'{DIR}{filename}.npz')
    dipole_moment, saveat = data['dipole_moment'], data['saveat']
    
    for component in range(dipole_moment.shape[1]):
        plt.plot(saveat, dipole_moment[:, component], '--')
        plt.xlabel(r"$t$", fontsize=20)
        plt.ylabel(r"$d$", fontsize=25)
        plt.savefig(f"dip_{filename}_{component}.pdf")
        plt.close()

def simulate_random_phase_approximation(stack, tau):
    omegas_ref, ref_data = load_reference_data()
    polarization = 0
    coulomb_strength = 1.0

    start_time = time.time()
    alpha = granad.rpa_polarizability_function(stack=stack, tau=tau, polarization=polarization, coulomb_strength=coulomb_strength, hungry=False)
    absorption = jax.vmap(alpha)(omegas_ref).imag * 4 * jnp.pi * omegas_ref
    print(f"RPA Simulation time: {time.time() - start_time}")

    plt.plot(omegas_ref, absorption / jnp.max(jnp.abs(absorption)), '-', label='RPA')
    plt.plot(omegas_ref, ref_data, '--', label='Reference')
    plt.legend()
    plt.savefig('acsnano_rpa.pdf')
    plt.close()        

def simulate_time_dependent(stack, tau):
    time_axis = jnp.linspace(0, 100, int(1e5))
    saveat = time_axis[::10]
    amplitudes, frequency, peak, fwhm = [1e-5, 0, 0], 2.3, 2, 0.5
    field_func = granad.electric_field_pulse(amplitudes, frequency, stack.positions[0, :], peak, fwhm)
    
    start_time = time.time()
    stack, solution = granad.evolution(
        stack, time_axis, field_func, dissipation=granad.relaxation(tau),
        saveat=saveat, stepsize_controller=diffrax.PIDController(atol=1e-10, rtol=1e-10)
    )
    print(f"Simulation time: {time.time() - start_time}")

    dipole_moment = granad.induced_dipole_moment(stack, jnp.diagonal(solution.ys, axis1=1, axis2=2))
    electric_field = field_func(saveat)
    jnp.savez(f'{DIR}{stack.electrons}.npz', dipole_moment=dipole_moment, electric_field=electric_field, saveat=saveat, positions=stack.positions)
    plot_fourier_transformed_data(stack.electrons)
    plot_time_domain_data(stack.electrons)

def run_simulation(length):
    sb = granad.StackBuilder()
    graphene = granad.Lattice(shape=granad.Rhomboid(length, 8), lattice_type=granad.LatticeType.HONEYCOMB, lattice_edge=granad.LatticeEdge.ARMCHAIR, lattice_constant=2.46)
    sb.add("pz", graphene)
    sb.set_hopping(granad.LatticeCoupling("pz", "pz", graphene, [0, -2.66]))
    sb.set_coulomb(granad.LatticeCoupling("pz", "pz", graphene, [16.522, 8.64, 5.333], coupling_function=lambda d: 14.399 / d + 0j))
    
    stack = sb.get_stack(doping=0)
    for tau in [5]:
        simulate_time_dependent(stack, tau)
        simulate_random_phase_approximation(stack, tau)
        
if __name__ == '__main__':
    if not os.path.isdir(DIR):
        os.mkdir(DIR)

    for length in [150]:
        run_simulation(length)
