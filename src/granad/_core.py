import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import diffrax

def fermi(e, beta, mu):
    return 1 / (jnp.exp(beta * (e - mu)) + 1)

def density_matrix(
    energies,
    electrons,
    spin_degeneracy,
    eps,
    from_state,
    to_state,
    excited_electrons,
    beta,
) -> tuple[jax.Array, int]:
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


# TODO: notify if gradient descent does not converge
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

    return (
        jnp.diag(spin_degeneracy * fermi(energies, beta, final_mu)) / electrons,
        jnp.nan,
    )


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

    return jnp.diag(occupation) / electrons, homo


## DIPOLE TRANSITIONS
def dipole_transitions(
    stack: Stack,
    add_induced: bool = False,
) -> Callable:
    """Takes into account dipole transitions.

        - `stack`:
        - `add_induced`: add induced field to the field acting on the dipole

    **Returns:**

     function as additional input for evolution function
    """

    def inner(charge, H, E):
        def element(i, j):
            return jax.lax.cond(
                # check if position index combination corresponds to an adatom orbital
                jnp.any(jnp.all(indices == jnp.array([i, j]), axis=2)),
                lambda x: jax.lax.switch(
                    2 * (i == j) + (i < j),
                    [
                        lambda x: x
                        + 0.5 * moments[ind(i), :] @ (E[:, i] + induced_E[ind(i), :]),
                        lambda x: x
                        + (
                            0.5 * moments[ind(i), :] @ (E[:, i] + induced_E[ind(i), :])
                        ).conj(),
                        lambda x: x + stack.positions[i, :] @ induced_E[ind(i), :].real,
                    ],
                    x,
                ),
                lambda x: x,
                H[i, j],
            )

        # array of shape orbitals x 3, such that induced_E[i, :] corresponds to the induced field at the position of the adatom associated with the i-th transition
        induced_E = (
            14.39
            * jnp.tensordot(
                r_point_charge, charge.real, axes=[0, 0]
            )  # computes \sum_i r_i/|r_i|^3 * Q_i
            * add_induced
        )
        return jax.vmap(jax.vmap(jax.jit(element), (0, None), 0), (None, 0), 0)(
            idxs, idxs
        )

    if stack.transitions is None:
        return jax.jit(lambda c, h, e: h)

    # map position index to transition index
    ind = jax.jit(lambda i: jnp.argmin(jnp.abs(i - indices[:, 0, :])))
    idxs = jnp.arange(stack.positions.shape[0])

    indices, moments = [], []
    for (orb1, orb2), moment in stack.transitions.items():
        i1, i2 = (
            jnp.where(stack.ids == stack.unique_ids.index(orb1))[0],
            jnp.where(stack.ids == stack.unique_ids.index(orb2))[0],
        )
        assert (
            i1.size == i2.size == 1
        ), "Dipole transitions are allowed only between orbitals with unique names in the entire stack (name each dipole orbital differently, e.g. d1,d2 for 1 dipole or d11, d12 and d21, d22 for two dipoles)"
        i1, i2 = i1[0], i2[0]
        assert jnp.allclose(
            stack.positions[i1], stack.positions[i2]
        ), "Dipole transitions must happen between orbitals at the same location"
        indices.append([[i1, i2], [i2, i1], [i1, i1], [i2, i2]])
        moments += [moment, moment]

    indices, moments = jnp.array(indices), jnp.array(moments)

    # array of shape positions x orbitals x 3, with entries vec_r[i, o, :] = r_i - r_o
    vec_r = (
        jnp.repeat(
            stack.positions[:, jnp.newaxis, :], 2 * len(stack.transitions), axis=1
        )
        - stack.positions[indices[:, 0, :].flatten(), :]
    )
p
    # array of shape positions x orbitals x 3, with entries r_point_charge[i, o, :] = (r_o - r_i)/|r_o - r_i|^3
    r_point_charge = jnp.nan_to_num(
        vec_r / jnp.expand_dims(jnp.linalg.norm(vec_r, axis=2) ** 3, 2),
        posinf=0.0,
        neginf=0.0,
    )

    return jax.jit(inner)

def relaxation(tau: float) -> Callable:
    """Function for modelling dissipation according to the relaxation approximation.

        - `tau`: relaxation time

    **Returns:**

    -compiled closure that is needed for computing the dissipative part of the lindblad equation
    """
    return jax.jit(lambda r, rs: -(r - rs) / (2 * tau))


def lindblad(stack, gamma, saturation):
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

    def inner(r, rs):
        # convert rho to energy basis
        r = stack.eigenvectors.conj().T @ r @ stack.eigenvectors

        # extract occupations
        diag = jnp.diag(r) * stack.electrons

        # apply the saturation functional to turn off elements in the gamma matrix
        gamma = gamma_matrix * saturation_vmapped(diag)[None, :]

        a = jnp.diag(gamma.T @ jnp.diag(r))
        mat = jnp.diag(jnp.sum(gamma, axis=1))
        b = -1 / 2 * (mat @ r + r @ mat)
        val = a + b

        return stack.eigenvectors @ val @ stack.eigenvectors.conj().T

    return inner

def get_self_consistent(
    stack: Stack, iterations: int = 50, mix: float = 0.05, accuracy: float = 1e-6
) -> Stack:
    """Get a stack with a self-consistent IP Hamiltonian.

        - `stack`: a stack object
        - `iterations`:
        - `mix`:
        - `accuracy`:

    **Returns:**

     stack object
    """

    def _to_site_basis(ev, mat):
        return ev @ mat @ ev.conj().T

    def _phi(rho):
        return stack.coulomb @ jnp.diag(rho - rho_uniform)

    def _stop(args):
        return jnp.logical_and(
            jnp.linalg.norm(_phi(args[0]) - _phi(args[1])) > accuracy,
            args[2] < iterations,
        )

    def _loop(args):
        rho, rho_old, idx = args
        ham_new = stack.hamiltonian + _phi(rho) * mix + _phi(rho_old) * (1 - mix)

        # diagonalize
        energies, eigenvectors = jnp.linalg.eigh(ham_new)

        # new density matrix
        rho_energy, _ = _density_matrix(
            energies,
            stack.electrons,
            stack.spin_degeneracy,
            stack.eps,
            eq_arr,
            eq_arr,
            eq_arr,
            stack.beta,
        )

        return _to_site_basis(eigenvectors, rho_energy), rho, idx + 1

    # construct a uniform density matrix in real space
    _, arr, counts = jnp.unique(
        jnp.round(stack.positions, 8), return_inverse=True, return_counts=True, axis=0
    )
    normalization = 2.0 if int(stack.spin_degeneracy) == 1 else 1.0
    rho_uniform = jnp.diag((1 / counts)[arr[:, 0]]) / (
        stack.energies.size * normalization
    )
    eq_arr = jnp.array([0])

    # first induced potential
    rho_old = jnp.zeros_like(stack.hamiltonian)
    rho = _to_site_basis(stack.eigenvectors, stack.rho_stat)

    # sc loop
    rho, rho_old, idx = jax.lax.while_loop(_stop, _loop, (rho, rho_old, 0))
    if idx == iterations - 1:
        raise Exception("Self-consistent procedure did not converge!!")

    # new hamiltonian and initial state
    ham_new = stack.hamiltonian + _phi(rho) * mix + _phi(rho_old) * (1 - mix)
    energies, eigenvectors = jnp.linalg.eigh(ham_new)
    rho_0, homo = _density_matrix(
        energies,
        stack.electrons,
        stack.spin_degeneracy,
        stack.eps,
        stack.from_state,
        stack.to_state,
        stack.excited_electrons,
        stack.beta,
    )

    return stack.replace(
        hamiltonian=ham_new,
        rho_0=rho_0,
        rho_stat=eigenvectors.conj().T @ rho @ eigenvectors,
        energies=energies,
        eigenvectors=eigenvectors,
        homo=homo,
    )

# TODO: units
def evolution(
    stack: Stack,
    time: jax.Array,
    field: FieldFunc,
    dissipation: DissipationFunc = lambda x, y: 0.0,
    coulomb_strength: float = 1.0,
    saveat=None,
    solver=diffrax.Dopri5(),
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    spatial=False,
    add_induced=False,
    custom = False
):

    def integrate_custom(rho, time):
        e_field, delta_rho = field(time), rho - rho_stat
        charge = -jnp.diag(delta_rho) * stack.electrons
        p_ext = jnp.sum(stack.positions * e_field.real.T, axis=1)
        p_ind = coulomb @ charge
        h_total = transition(charge, stack.hamiltonian, e_field) + jnp.diag(
            p_ext - p_ind
        )
        if dissipation:
            return (
                rho
                - 1j * dt * (h_total @ rho - rho @ h_total)
                + dt * dissipation(rho, rho_stat),
                postprocess(rho) if postprocess else rho,
            )
        else:
            return (
                rho - 1j * dt * (h_total @ rho - rho @ h_total),
                postprocess(rho) if postprocess else rho,
            )

    def rhs_uniform(time, rho, args):
        e_field, delta_rho = field(time), rho - rho_stat
        charge = -jnp.diag(delta_rho) * stack.electrons
        p_ext = jnp.sum(stack.positions * e_field.real.T, axis=1)
        p_ind = coulomb @ charge
        h_total = transition(charge, stack.hamiltonian, e_field) + jnp.diag(
            p_ext - p_ind
        )
        return -1j * (h_total @ rho - rho @ h_total) + dissipation(rho, rho_stat)

    def rhs_spatial(time, rho, args):
        vector_potential, delta_rho = field(time), rho - rho_stat
        charge = -jnp.diag(delta_rho) * stack.electrons
        p_ind = coulomb @ charge

        # we need to go from velocity to momentum under minimal coupling, i.e. p -> p - qA; H -> H - q A v + q^2/(2m) A^2
        h_total = (
            stack.hamiltonian
            - q * jnp.einsum("ijr, ir -> ij", v, vector_potential)
            + jnp.diag(q**2 / m * 0.5 * jnp.sum(vector_potential**2, axis=1) - p_ind)
        )
        return -1j * (h_total @ rho - rho @ h_total) + dissipation(rho, rho_stat)

    coulomb = stack.coulomb * coulomb_strength
    rho_stat = stack.eigenvectors @ stack.rho_stat @ stack.eigenvectors.conj().T

    if spatial:
        q = 1
        m = 1
        v = velocity_operator(stack)
        rhs = rhs_spatial
    else:
        rhs = rhs_uniform
        transition = dipole_transitions(stack, add_induced)

    if custom:

        dt = time[1] - time[0]
        coulomb = stack.coulomb * coulomb_strength
        rho_stat = stack.eigenvectors @ stack.rho_stat @ stack.eigenvectors.conj().T
        transition = dipole_transitions(stack, add_induced=add_induced)
        rho, rhos = jax.lax.scan(
            integrate_custom, stack.eigenvectors @ stack.rho_0 @ stack.eigenvectors.conj().T, time
        )

        return rhos

    term = diffrax.ODETerm(rhs)
    rho_init = stack.eigenvectors @ stack.rho_0 @ stack.eigenvectors.conj().T
    saveat = diffrax.SaveAt(ts=time if saveat is None else saveat)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=time[0],
        t1=time[-1],
        dt0=time[1] - time[0],
        y0=rho_init,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )

def rpa_polarizability_function(
    stack, tau, polarization, coulomb_strength, phi_ext=None, hungry=True
):
    def _polarizability(omega):
        ro = sus(omega) @ phi_ext
        return -pos @ ro

    pos = stack.positions[:, polarization]
    phi_ext = pos if phi_ext is None else phi_ext
    sus = rpa_susceptibility_function(stack, tau, coulomb_strength, hungry)
    return _polarizability


def rpa_susceptibility_function(stack, tau, coulomb_strength, hungry=True):
    def _rpa_susceptibility(omega):
        x = sus(omega)
        return x @ jnp.linalg.inv(one - c @ x)

    sus = bare_susceptibility_function(stack, tau, hungry)
    c = stack.coulomb * coulomb_strength
    one = jnp.identity(stack.hamiltonian.shape[0])

    return _rpa_susceptibility


def bare_susceptibility_function(stack, tau, hungry=True):

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
            eq = 2.0 * Sf / (omega - omega_grid_extended + 1j / (2.0 * tau))
            return -jnp.sum(eq)

        if hungry:
            return jax.vmap(
                jax.vmap(susceptibility_element, (0, None), 0), (None, 0), 0
            )(sites, sites)
        return jax.lax.map(
            lambda i: jax.lax.map(lambda j: susceptibility_element(i, j), sites), sites
        )

    # unpacking
    energies = stack.energies.real
    eigenvectors = stack.eigenvectors.real
    occupation = jnp.diag(stack.rho_0).real * stack.electrons / stack.spin_degeneracy
    sites = jnp.arange(energies.size)
    freq_number = 2**12
    omega_max = jnp.real(max(stack.energies[-1], -stack.energies[0])) + 0.1
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


# TODO: think of smarter of doing this
def to_site_basis(stack: Stack, matrix: jax.Array) -> jax.Array:
    """Transforms an arbitrary matrix from energy to site basis.

        - `stack`: stack object
        - `matrix`: square array in energy basis

    **Returns:**

     array in site basis
    """
    return stack.eigenvectors @ matrix @ stack.eigenvectors.conj().T


def to_energy_basis(stack: Stack, matrix: jax.Array) -> jax.Array:
    """Transforms an arbitrary matrix from site to energy basis.

        - `stack`: stack object
        - `matrix`: square array in energy basis

    **Returns:**

     array in energy basis
    """
    return stack.eigenvectors.conj().T @ matrix @ stack.eigenvectors
