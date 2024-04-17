import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import diffrax

# TODO: write somewhere that the einsum convention used is: site indices are small, electric field component indices are big
# TODO basis conversion is duplicated at least three times here :///
# TODO: decouple rpa from the OrbitalList by just passing (a lot of) arguments
# TODO: think about public / private

def fraction_periodic(signal, threshold=1e-2):

    # signal is periodic => abs(signal) is periodic
    cum_sum = jnp.abs(signal).cumsum()

    # cumulative mean of periodic signal is constant
    cum_mean = cum_sum / jnp.arange(1, len(signal) + 1)

    # if cumulative mean doesn't move anymore, we have a lot of periodic signal
    med = jnp.median(cum_mean)
    deviation = jnp.abs(med - cum_mean) / med

    # approximate admixture of periodic signal
    return (deviation < threshold).sum() / len(signal)

# TODO: does this work as intended?
def get_fourier_transform(t_linspace, function_of_time, return_omega_axis = True):
    function_of_omega = jnp.fft.fft(function_of_time) / len(t_linspace)
    omega_axis = (
        2
        * jnp.pi
        * len(t_linspace)
        / jnp.max(t_linspace)
        * jnp.fft.fftfreq(function_of_omega.shape[-1])
    )
    return omega_axis, function_of_omega if return_omega_axis else function_of_omega

def fermi(e, beta, mu):
    return 1 / (jnp.exp(beta * (e - mu)) + 1)

def _density_matrix(
    energies,
    electrons,
    spin_degeneracy,
    eps,
    from_state,
    to_state,
    excited_electrons,
    beta,
):
    """Calculates the normalized spin-traced 1RDM. For zero temperature, accordning to the Aufbau principle.
    At finite temperature, the chemical potential is determined for the thermal density matrix.

    - `energies`: IP energies of the nanoflake
    """

    if beta == jnp.inf:
        return _density_aufbau(
            energies,
            electrons,
            spin_degeneracy,
            eps,
            from_state,
            to_state,
            excited_electrons,
        )
    return _density_thermo(energies, electrons, spin_degeneracy, eps, beta)


def _density_thermo(
    energies, electrons, spin_degeneracy, eps, beta, learning_rate=0.1, max_iter=10
):

    def loss(mu):
        return (spin_degeneracy * fermi(energies, beta, mu).sum() - electrons) ** 2

    # Compute the gradient of the loss function
    grad_loss = jax.grad(loss)

    # Define the gradient descent update step
    def update_step(mu):
        gradient = grad_loss(mu)
        new_mu = mu - learning_rate * gradient
        return new_mu

    # Define the condition for the while loop to continue
    # Here we simply run for a fixed number of iterations
    def cond_fun(val):
        mu, i = val
        return jnp.logical_or(loss(mu) > 1e-6, i < max_iter)

    # Define the body of the while loop
    def body_fun(val):
        mu, i = val
        new_mu = update_step(mu)
        return new_mu, i + 1

    # Initialize mu and iteration counter
    mu_init = 0.0  # initial value
    i_init = 0  # initial iteration

    # Run the gradient descent
    final_mu, final_iter = jax.lax.while_loop(cond_fun, body_fun, (mu_init, i_init))

    if final_iter == max_iter:
        raise ValueError("Thermal density matrix construction did not converge.")

    return  jnp.diag(spin_degeneracy * fermi(energies, beta, final_mu)) / electrons,


def _density_aufbau(
    energies, electrons, spin_degeneracy, eps, from_state, to_state, excited_electrons
):

    def _occupation(flags, spin_degeneracy, fraction):
        return jax.vmap(
            lambda flag: jax.lax.switch(
                flag,
                [lambda x: spin_degeneracy, lambda x: fraction, lambda x: 0.0],
                flag,
            )
        )(flags)

    def _flags(index):
        """
        returns a tuple
        first element : array, labels all energies relative to a level given by energies[index] with
        0 => is_below_level, 1 => is_level, 2 => is_above_level
        second element : degeneracy of level, identified by the number of energies e fullfilling |e - level| < eps
        """
        is_level = jnp.abs(energies[index] - energies) < eps
        is_larger = energies - energies[index] > eps
        degeneracy = is_level.sum()
        return is_level + 2 * is_larger, degeneracy

    def _excitation(take_from, put_to, number):
        flags_from, deg_from = _flags(take_from)
        flags_to, deg_to = _flags(put_to)

        # transform to flag array, where 0 => else, 1 => from, 2 => to
        flags = (flags_from == 1) + (flags_to == 1) * 2
        return jax.vmap(
            lambda flag: jax.lax.switch(
                flag,
                [
                    lambda x: 0.0,
                    lambda x: -number / deg_from,
                    lambda x: number / deg_to,
                ],
                flag,
            )
        )(flags)

    def _body_fun(index, occupation):
        return jax.lax.cond(
            -from_state[index] == to_state[index],
            lambda x: x,
            lambda x: x
            + _excitation(
                homo - from_state[index],
                homo + to_state[index],
                excited_electrons[index],
            ),
            occupation,
        )

    homo = jnp.array(jnp.ceil(electrons / spin_degeneracy), int) - 1
    # determine where and how often this energy occurs
    flags, degeneracy = _flags(homo)
    # compute electrons at the fermi level
    below = (energies < energies[homo] - eps).sum()
    homo = jnp.nonzero(flags, size=1)[0]
    remaining_electrons = electrons - spin_degeneracy * below
    # compute ground state occupations by distributing the remaining electrons across all degenerate levels
    occupation = _occupation(flags, spin_degeneracy, remaining_electrons / degeneracy)
    # excited states
    occupation = jax.lax.fori_loop(0, to_state.size, _body_fun, occupation)

    return jnp.diag(occupation) / electrons
        

# TODO: this needs more clarity, and basis trafo is essentially duplicated here
def _get_self_consistent(hamiltonian, coulomb, positions, spin_degeneracy, electrons, eps, eigenvectors, rho_stat, iterations, mix, accuracy ):

    def _to_site_basis(ev, mat):
        return ev @ mat @ ev.conj().T

    def _phi(rho):
        return coulomb @ jnp.diag(rho - rho_uniform)

    def _stop(args):
        return jnp.logical_and(
            jnp.linalg.norm(_phi(args[0]) - _phi(args[1])) > accuracy,
            args[2] < iterations,
        )

    def _loop(args):
        rho, rho_old, idx = args
        ham_new = hamiltonian + _phi(rho) * mix + _phi(rho_old) * (1 - mix)

        # diagonalize
        energies, eigenvectors = jnp.linalg.eigh(ham_new)

        # new density matrix
        rho_energy, _ = _density_aufbau(
            energies,
            electrons,
            spin_degeneracy,
            eps,
            eq_arr,
            eq_arr,
            eq_arr,
        )

        return _to_site_basis(eigenvectors, rho_energy), rho, idx + 1

    # construct a uniform density matrix in real space
    _, arr, counts = jnp.unique(
        jnp.round(positions, 8), return_inverse=True, return_counts=True, axis=0
    )
    normalization = 2.0 if int(spin_degeneracy) == 1 else 1.0
    rho_uniform = jnp.diag((1 / counts)[arr[:, 0]]) / (
        energies.size * normalization
    )
    eq_arr = jnp.array([0])

    # first induced potential
    rho_old = jnp.zeros_like(hamiltonian)
    rho = _to_site_basis(eigenvectors, rho_stat)

    # sc loop
    rho, rho_old, idx = jax.lax.while_loop(_stop, _loop, (rho, rho_old, 0))
    if idx == iterations - 1:
        raise ValueError("Self-consistent procedure did not converge!!")

    # new hamiltonian and initial state
    ham_new = hamiltonian + _phi(rho) * mix + _phi(rho_old) * (1 - mix)
    energies, eigenvectors = jnp.linalg.eigh(ham_new)
    rho_0, homo = _density_aufbau(
        energies,
        electrons,
        spin_degeneracy,
        eps,
        from_state,
        to_state,
        excited_electrons,
    )
    
    return ham_new, rho_0, eigenvectors.conj().T @ rho @ eigenvectors, energies, eigenvectors

def relaxation_time_approximation(relaxation_time, stationary_density_matrix):
    """Function for modelling dissipation according to the relaxation approximation.

        - `relaxation_time`: relaxation time

    **Returns:**

    -compiled closure that is needed for computing the dissipative part of the lindblad equation
    """
    return lambda r: -(r - stationary_density_matrix) / (2 * relaxation_time)


def lindblad_saturation_functional(eigenvectors, gamma, saturation):
    """Function for modelling dissipation according to the saturated lindblad equation as detailed in https://link.aps.org/doi/10.1103/PhysRevA.109.022237.

        - `stack`: object representing the state of the system
        - `gamma`: symmetric (or lower triangular) NxN matrix. The element gamma[i,j] corresponds to the transition rate from state i to state j
        - `saturation`: a saturation functional to apply, defaults to a sharp turn-off

    **Returns:**

    -compiled closure that is needed for computing the dissipative part of the lindblad equation
    """

    commutator_diag = jnp.diag(gamma)
    gamma_matrix = gamma.astype(complex)
    saturation_vmapped = jax.vmap(saturation, 0, 0)

    def inner(r):
        # convert rho to energy basis
        r = eigenvectors.conj().T @ r @ eigenvectors

        # extract occupations
        diag = jnp.diag(r) * stack.electrons

        # apply the saturation functional to turn off elements in the gamma matrix
        gamma = gamma_matrix * saturation_vmapped(diag)[None, :]

        a = jnp.diag(gamma.T @ jnp.diag(r))
        mat = jnp.diag(jnp.sum(gamma, axis=1))
        b = -1 / 2 * (mat @ r + r @ mat)
        val = a + b

        return eigenvectors @ val @ eigenvectors.conj().T

    return inner

def get_coulomb_field_to_from(source_positions, target_positions):
    """
    Calculate the contributions of point charges located at `source_positions`
    on points at `target_positions`.

    **Args:**
    - source_positions (array): An (n_source, 3) array of source positions.
    - target_positions (array): An (n_target, 3) array of target positions.

    **Returns:**
    - array: An (n_source, n_target, 3) array where each element is the contribution
          of a source at a target position.
    """
    # Calculate vector differences between each pair of source and target positions
    distance_vector = target_positions[:,None,:] - source_positions
    # Compute the norm of these vectors
    norms = jnp.linalg.norm(distance_vector, axis=-1)
    # Safe division by the cube of the norm
    one_over_distance_cubed = jnp.where(norms > 0, 1 / norms**3, 0)
    # Calculate and return the contributions
    coulomb_field_to_from = distance_vector * one_over_distance_cubed[:, :, None]
    return coulomb_field_to_from


def get_induced_field_contribution(
        positions,
        compute_only_at : None,
        
):
    """Takes into account dipole transitions.

        - `stack`:
        - `add_induced`: add induced field to the field acting on the dipole

    **Returns:**

     function as additional input for evolution function
    """

    contributions = get_point_charge_contributions(positions, positions)    
    if compute_only_at is not None:
        selected_contributions = jnp.zeros_like( contributions )
        selected_contributions.at[compute_only_at].set( contributions[compute_only_at] )
        return selected_contributions
    return contributions


# TODO: this is slightly awkard, change to faster axis
def get_induced_electric_field( coulomb_field_to_from, charge ):
    # sum up up all charges weighted like \sum_r q_r r/|r-r'|
    # read: field to i from j
    return jnp.einsum( 'ijK,j->iK', coulomb_field_to_from, charge)    

# TODO: units
def density_matrix_vector_potential(
        hamiltonian,
        coulomb,
        dipole_operator,
        electrons,
        initial_density_matrix,
        stationary_density_matrix,
        time_axis,
        vector_potential_function,
        velocity_operator,        
        dissipation_function,
        coulomb_field_from_to,
        no_induced_contribution,
        coulomb_strength,
        solver,
        stepsize_controller,
):
    # TODO: uff, also: should this be here?
    def _get_induced_field_potential( charge ):
        if no_induced_contribution:
            return 0.0
        induced_electric_field = get_induced_electric_field( coulomb_field_to_from, charge )
        return jnp.einsum( 'ijK,jK->i', dipole_operator, induced_electric_field )        

    def rhs(time, density_matrix, args):
        # inhomogeneous external field => Nx3-dim array of real
        vector_potential = vector_potential_function(time)
        
        # net induced charge => N-dim array of complex (although diagonal elements are mostly real)
        charge = -jnp.diag( density_matrix - stationary_density_matrix ) * electrons

        # coulomb term is just \sum_{i} C_{ij} \rho_{ii}
        coulomb_potential = coulomb @ charge
        induced_field_potential = _get_induced_field_potential( charge )

        # ~ A p 
        paramagnetic = - q * jnp.einsum("ijr, ir -> ij", velocity_operator, vector_potential)

        # ~ A^2
        diamagnetic = q**2 / m * 0.5 * jnp.sum(vector_potential**2, axis=1) 
        
        h_total = hamiltonian + paramagnetic + jnp.diag( diamagnetic + induced_field_potential + coulomb_potential )
        
        return -1j * (h_total @ rho - rho @ h_total) + dissipation_function(rho)

    # atomic units
    q, m = 1, 1
        
    term = diffrax.ODETerm(rhs)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=time_axis[0],
        t1=time_axis[-1],
        dt0=time_axis[1] - time_axis[0],
        y0=rho_init,
        saveat=time_axis,
        stepsize_controller=stepsize_controller,
    )

def density_matrix_electric_field(
        hamiltonian,
        coulomb,
        dipole_operator,
        electrons,
        initial_density_matrix,
        stationary_density_matrix,
        time_axis,
        electric_field_function,
        dissipation_function,
        coulomb_field_to_from,
        no_induced_contribution,
        coulomb_strength,
        solver,
        stepsize_controller
):
    # TODO: uff
    def _induced_electric_field_func( charge ):
        if no_induced_contribution:
            return 0.0
        return get_induced_electric_field(coulomb_field_to_from, charge)
    
    def rhs(time, density_matrix, args):
        
        # homogeneous external field => 3-dim array of real
        external_electric_field = electric_field_function(time)

        # net induced charge => N-dim array of complex (although diagonal elements are mostly real)
        charge = -jnp.diag(density_matrix - stationary_density_matrix) * electrons
        
        # coulomb term is just \sum_{i} C_{ij} \rho_{ii}
        coulomb_potential = coulomb @ charge

        induced_electric_field = _induced_electric_field_func( charge )

        # total electric field
        electric_field = external_electric_field + induced_electric_field

        # TODO: order of site indices does not matter for real transition dipole moments, but what about complex?
        # the dipole operator contains the dipole moments and the position matrix elements
        total_field_potential = jnp.einsum( 'ijK,jK->i', dipole_operator, electric_field )

        # collect terms in hamiltonian
        h_total = hamiltonian + jnp.diag( total_field_potential + coulomb_potential )
        
        return -1j * (h_total @ rho - rho @ h_total) + dissipation_function(rho)


    term = diffrax.ODETerm(rhs)    
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=time_axis[0],
        t1=time_axis[-1],
        dt0=time_axis[1] - time_axis[0],
        y0=initial_density_matrix,
        saveat=time_axis,
        stepsize_controller=stepsize_controller,
    )

# TODO: this is duplication!!!
def density_matrix_old(
        hamiltonian,
        coulomb,
        dipole_operator,
        electrons,
        initial_density_matrix,
        stationary_density_matrix,
        time_axis,
        electric_field_function,
        dissipation_function,
        coulomb_field_to_from,
        no_induced_contribution,
        coulomb_strength,
        solver,
        stepsize_controller
):
    def _induced_electric_field_func( charge ):
        if no_induced_contribution:
            return 0.0
        return get_induced_electric_field(coulomb_field_to_from, charge)

    def runge_kutta_first_order(density_matrix, time):
        
        # homogeneous external field => 3-dim array of real
        external_electric_field = electric_field_function(time)

        # net induced charge => N-dim array of complex (although diagonal elements are mostly real)
        charge = -jnp.diag(density_matrix - stationary_density_matrix) * electrons
        
        # coulomb term is just \sum_{i} C_{ij} \rho_{ii}
        coulomb_potential = coulomb @ charge

        induced_electric_field = _induced_electric_field_func( charge )

        # total electric field
        electric_field = external_electric_field + induced_electric_field

        # TODO: order of site indices does not matter for real transition dipole moments, but what about complex?
        # the dipole operator contains the dipole moments and the position matrix elements
        total_field_potential = jnp.einsum( 'ijK,jK->i', dipole_operator, electric_field )

        # collect terms in hamiltonian
        h_total = hamiltonian + jnp.diag( total_field_potential + coulomb_potential )
        
        return density_matrix - 1j * dt * (h_total @ rho - rho @ h_total) + dt * dissipation_function(density_matrix)    

    dt = time[1] - time[0]
    rho, rhos = jax.lax.scan( runge_kutta_first_order, initial_density_matrix, time )

    return rhos

def rpa_polarizability_function(
    orbs, relaxation_time, polarization, coulomb_strength, phi_ext=None, hungry=True
):
    def _polarizability(omega):
        ro = sus(omega) @ phi_ext
        return -pos @ ro

    pos = orbs.positions[:, polarization]
    phi_ext = pos if phi_ext is None else phi_ext
    sus = rpa_susceptibility_function(orbs, relaxation_time, coulomb_strength, hungry)
    return _polarizability


def rpa_susceptibility_function(orbs, relaxation_time, coulomb_strength, hungry=True):
    def _rpa_susceptibility(omega):
        x = sus(omega)
        return x @ jnp.linalg.inv(one - c @ x)

    sus = bare_susceptibility_function(orbs, relaxation_time, hungry)
    c = orbs.coulomb * coulomb_strength
    one = jnp.identity(orbs.hamiltonian.shape[0])

    return _rpa_susceptibility


def bare_susceptibility_function(orbs, relaxation_time, hungry=True):

    def _sum_subarrays(arr):
        """Sums subarrays in 1-dim array arr. Subarrays are defined by n x 2 array indices as [ [start1, end1], [start2, end2], ... ]"""
        arr = jnp.r_[0, arr.cumsum()][indices]
        return arr[:, 1] - arr[:, 0]

    def _susceptibility(omega):

        def susceptibility_element(site_1, site_2):
            # calculate per-energy contributions
            prefac = eigenvectors[site_1, :] * eigenvectors[site_2, :] / delta_omega
            f_lower = prefac * (energies - omega_low) * occupation
            g_lower = prefac * (energies - omega_low) * (1 - occupation)
            f_upper = prefac * (-energies + omega_up) * occupation
            g_upper = prefac * (-energies + omega_up) * (1 - occupation)
            f = (
                jnp.r_[0, _sum_subarrays(f_lower)][mask]
                + jnp.r_[0, _sum_subarrays(f_upper)][mask2]
            )
            g = (
                jnp.r_[0, _sum_subarrays(g_lower)][mask]
                + jnp.r_[0, _sum_subarrays(g_upper)][mask2]
            )
            b = jnp.fft.ihfft(f, n=2 * f.size, norm="ortho") * jnp.fft.ihfft(
                g[::-1], n=2 * f.size, norm="ortho"
            )
            Sf1 = jnp.fft.hfft(b)[:-1]
            Sf = -Sf1[::-1] + Sf1
            eq = 2.0 * Sf / (omega - omega_grid_extended + 1j / (2.0 * relaxation_time))
            return -jnp.sum(eq)

        if hungry:
            return jax.vmap(
                jax.vmap(susceptibility_element, (0, None), 0), (None, 0), 0
            )(sites, sites)
        return jax.lax.map(
            lambda i: jax.lax.map(lambda j: susceptibility_element(i, j), sites), sites
        )

    # unpacking
    energies = orbs.energies.real
    eigenvectors = orbs.eigenvectors.real
    occupation = jnp.diag(orbs.rho_0).real * orbs.electrons / orbs.spin_degeneracy
    sites = jnp.arange(energies.size)
    freq_number = 2**12
    omega_max = jnp.real(max(orbs.energies[-1], -orbs.energies[0])) + 0.1
    omega_grid = jnp.linspace(-omega_max, omega_max, freq_number)

    # build two arrays sandwiching the energy values: omega_low contains all frequencies bounding energies below, omega_up bounds above
    upper_indices = jnp.argmax(omega_grid > energies[:, None], axis=1)
    omega_low = omega_grid[upper_indices - 1]
    omega_up = omega_grid[upper_indices]
    delta_omega = omega_up[0] - omega_low[0]

    omega_dummy = jnp.linspace(-2 * omega_grid[-1], 2 * omega_grid[-1], 2 * freq_number)
    omega_3 = omega_dummy[1:-1]
    omega_grid_extended = jnp.insert(omega_3, int(len(omega_dummy) / 2 - 1), 0)

    # indices for grouping energies into contributions to frequency points.
    # e.g. energies like [1,1,2,3] on a frequency grid [0.5, 1.5, 2.5, 3.5]
    # the contribution array will look like [ f(1, eigenvector), f(1, eigenvector'), f(2, eigenvector_2), f(3, eigenvector_3) ]
    # we will have to sum the first two elements
    # we do this by building an array "indices" of the form: [ [0, 2], [2,3], [3, 4] ]
    omega_low_unique, indices = jnp.unique(omega_low, return_index=True)
    indices = jnp.r_[jnp.repeat(indices, 2)[1:], indices[-1] + 1].reshape(
        omega_low_unique.size, 2
    )

    # mask for inflating the contribution array to the full size given by omega_grid.
    comparison_matrix = omega_grid[:, None] == omega_low_unique[None, :]
    mask = (jnp.argmax(comparison_matrix, axis=1) + 1) * comparison_matrix.any(axis=1)
    comparison_matrix = omega_grid[:, None] == jnp.unique(omega_up)[None, :]
    mask2 = (jnp.argmax(comparison_matrix, axis=1) + 1) * comparison_matrix.any(axis=1)
    return _susceptibility
