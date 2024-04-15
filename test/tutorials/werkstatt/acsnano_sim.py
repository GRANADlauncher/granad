# simulation following https://pubs.acs.org/doi/full/10.1021/acsnano.3c05246
import time

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import granad

DIR = "/users/tfp/ddams/"


def get_fourier_transform(t_linspace, function_of_time):
    function_of_omega = jnp.fft.fft(function_of_time) / len(t_linspace)
    omega_axis = (
        2
        * jnp.pi
        * len(t_linspace)
        / jnp.max(t_linspace)
        * jnp.fft.fftfreq(function_of_omega.shape[-1])
    )
    return omega_axis, function_of_omega


def find_closest_index(omegas, omega_max):
    # Calculate the absolute differences from omega_max
    differences = jnp.abs(omegas - omega_max)

    # Find the index of the smallest difference
    closest_index = jnp.argmin(differences)

    return closest_index


def eV_to_nm(eV_array):
    # Constants
    h = 4.135667696e-15  # Planck's constant in eV·s
    c = 299792458  # Speed of light in m/s

    # Convert eV to nm
    wavelength_nm = (h * c * 1e9) / eV_array

    return wavelength_nm


def plot_results(electrons, interpolate=True):

    data = jnp.load(f"{DIR}{electrons}.npz")
    dipole_moment = data["dipole_moment"]
    electric_field = data["electric_field"]
    saveat = data["saveat"]
    positions = data["positions"]

    #  perform fourier transform
    for component in range(1):
        if interpolate:
            dip = jnp.interp(
                jnp.linspace(saveat.min(), saveat.max(), 2 * saveat.size),
                saveat,
                dipole_moment[:, component],
            )
        else:
            dip = dipole_moment[:, component]
        omega_axis, dipole_omega = get_fourier_transform(saveat, dip)
        _, field_omega = get_fourier_transform(saveat, electric_field[component])

        # omega_max, omega_min = omega_axis.max(),omega_axis.min() # wavelength range is something
        # omega_max, omega_min = find_closest_index(omega_axis, omega_max), find_closest_index(omega_axis, omega_min)
        # 350-900nm, 3.5-1.3

        lower, upper = 0.8, 4
        idxs = jnp.argwhere(jnp.logical_and(lower < omega_axis, omega_axis < upper))
        # import pdb; pdb.set_trace()
        omega_axis, dipole_omega, field_omega = (
            omega_axis[idxs],
            dipole_omega[idxs],
            field_omega[idxs],
        )

        # plot result
        polarizability = dipole_omega / field_omega
        spectrum = -omega_axis * jnp.imag(polarizability)
        plt.plot(omega_axis, spectrum / jnp.max(jnp.abs(spectrum)), "-", label="TD")
        plt.plot(*load_ref(), "--", label="ref")
        # plt.gca().invert_xaxis()  # Inverts the x-axis
        # plt.xlabel(r"$\lambda$ [nm]", fontsize=20)
        plt.xlabel(r"$E$ [eV]", fontsize=20)
        plt.ylabel(r"$\sigma$", fontsize=25)
        plt.legend()
        # plt.show()
        plt.savefig(f"{DIR}absorption_{electrons}_{component}.pdf")
        plt.close()


def plot_results_real_space(electrons):

    data = jnp.load(f"{DIR}{electrons}.npz")
    dipole_moment = data["dipole_moment"]
    electric_field = data["electric_field"]
    saveat = data["saveat"]
    positions = data["positions"]

    #  perform fourier transform
    for component in range(1):
        plt.plot(saveat, dipole_moment[:, component], "--")
        plt.xlabel(r"$t$", fontsize=20)
        plt.ylabel(r"$d$", fontsize=25)
        plt.savefig(f"{DIR}dip_{electrons}_{component}.pdf")
        plt.close()


def load_ref():
    acsnano = jnp.array(np.genfromtxt("acsnano.csv", delimiter=";"))
    h = 6.6261 * 1e-34
    c = 2.9979 * 1e8
    omegas_ref = h * c / (acsnano[:, 0] * 1e-9) / (1.6 * 1e-19)
    idxs = jnp.argwhere(acsnano[:, 0] < 800)
    return (omegas_ref[idxs][:, 0], acsnano[idxs, 1] / jnp.max(acsnano[idxs, 1]))


def sim_rpa(stack):
    omegas = load_ref()[0]
    polarization = 0
    tau = 5
    coulomb_strength = 1.0
    start = time.time()
    alpha = granad.rpa_polarizability_function(
        stack=stack,
        tau=tau,
        polarization=polarization,
        coulomb_strength=coulomb_strength,
        hungry=False,
    )
    absorption = jax.vmap(alpha)(omegas).imag * 4 * jnp.pi * omegas
    print(time.time() - start)
    plt.plot(omegas, absorption / jnp.max(jnp.abs(absorption)), "-", label="RPA")
    plt.plot(*load_ref(), "--", label="ref")
    plt.legend()
    plt.savefig("acsnano_rpa.pdf")
    plt.close()


def sim_td(stack):
    # electric field
    amplitudes = [1e-5, 0, 0]
    frequency = 2.3

    # simulation duration
    time_axis = jnp.linspace(0, int(1e2), int(1e5))
    saveat = time_axis[::10]
    print(saveat.size)

    field_func = granad.electric_field(amplitudes, frequency, stack.positions[0, :])

    start = time.time()
    # run the simulation and extract occupations with a suitable postprocessing function
    stack, sol = granad.evolution(
        stack,
        time_axis,
        field_func,
        dissipation=lambda x, y: 0.0,  # granad.relaxation(0.2),
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(atol=1e-10, rtol=1e-10),
    )
    print(time.time() - start)
    print(sol.result == diffrax.RESULTS.successful)

    # calculate dipole moment
    dipole_moment = granad.induced_dipole_moment(
        stack, jnp.diagonal(sol.ys, axis1=1, axis2=2)
    )

    electric_field = granad.electric_field(
        amplitudes, frequency, stack.positions[0, :]
    )(saveat)

    # Save the arrays to an .npz file
    jnp.savez(
        f"{DIR}{stack.electrons}.npz",
        dipole_moment=dipole_moment,
        electric_field=electric_field,
        saveat=saveat,
        positions=stack.positions,
    )
    plot_results(stack.electrons)


def run_sim(length):
    # build stack
    sb = granad.StackBuilder()

    # ribbon geometry
    width = 8  # 9 atoms wide

    # make lattice
    graphene = granad.Lattice(
        shape=granad.Rhomboid(length, width),
        lattice_type=granad.LatticeType.HONEYCOMB,
        lattice_edge=granad.LatticeEdge.ARMCHAIR,
        lattice_constant=2.46,
    )
    sb.add("pz", graphene)

    # set couplings
    hopping_graphene = granad.LatticeCoupling(
        orbital_id1="pz", orbital_id2="pz", lattice=graphene, couplings=[0, -2.66]
    )
    sb.set_hopping(hopping_graphene)
    coulomb_graphene = granad.LatticeCoupling(
        orbital_id1="pz",
        orbital_id2="pz",
        lattice=graphene,
        couplings=[16.522, 8.64, 5.333],
        coupling_function=lambda d: 14.399 / d + 0j,
    )
    sb.set_coulomb(coulomb_graphene)

    # plotting takes a long time
    # sb.show2D()

    # get stack object
    stack = sb.get_stack(doping=0)

    sim_td(stack)
    sim_rpa(stack)


if __name__ == "__main__":
    lengths = [150]  # minimum length of 15 nm, maxmimum length is 25 nm
    for length in lengths:
        run_sim(length)
