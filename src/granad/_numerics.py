import jax
import jax.numpy as jnp
import itertools
import functools

jax.config.update("jax_enable_x64", True)
import diffrax

def fraction_periodic(signal, threshold=1e-2):
    """
    Estimates the fraction of a periodic component in a given signal by analyzing the deviation of the cumulative mean from its median value. The periodicity is inferred based on the constancy of the cumulative mean of the absolute value of the signal.

    Parameters:
        signal (jax.Array): A 1D array representing the signal of interest.
        threshold (float, optional): A threshold value to determine the significance level of deviation from periodicity. Defaults to 0.01.

    Returns:
        float: A ratio representing the fraction of the signal that is considered periodic, based on the specified threshold.

    """
    
    # signal is periodic => abs(signal) is periodic
    cum_sum = jnp.abs(signal).cumsum()

    # cumulative mean of periodic signal is constant
    cum_mean = cum_sum / jnp.arange(1, len(signal) + 1)

    # if cumulative mean doesn't move anymore, we have a lot of periodic signal
    med = jnp.median(cum_mean)
    deviation = jnp.abs(med - cum_mean) / med

    # approximate admixture of periodic signal
    return (deviation < threshold).sum().item() / len(signal)

def get_fourier_transform(t_linspace, function_of_time, omega_max = jnp.inf, omega_min = -jnp.inf, return_omega_axis=True):
    # Calculate the frequency axis
    delta_t = t_linspace[1] - t_linspace[0]  # assuming uniform spacing

    # Compute the FFT along the first axis    
    function_of_omega = jnp.fft.fft(function_of_time, axis=0) * delta_t

    N = function_of_time.shape[0]  # number of points in t_linspace
    omega_axis = 2 * jnp.pi * jnp.fft.fftfreq(N, d=delta_t)

    mask = (omega_axis >= omega_min) & (omega_axis <= omega_max)
    
    if return_omega_axis:
        return omega_axis[mask], function_of_omega[mask]
    else:
        return function_of_omega[mask]


def fermi(e, beta, mu):
    return 1 / (jnp.exp(beta * (e - mu)) + 1)


def _density_matrix(
    energies,
    electrons,
    spin_degeneracy,
    eps,
    excitation,
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
            excitation
        )

    density = fermi(energies, beta, 0)
    return jnp.diag(density / density.sum())
    # return _density_thermo(energies, electrons, spin_degeneracy, eps, beta)


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

    return (jnp.diag(spin_degeneracy * fermi(energies, beta, final_mu)) / electrons,)


def _density_aufbau(
    energies, electrons, spin_degeneracy, eps, excitation
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
                from_state[index],
                to_state[index],
                excited_electrons[index],
            ),
            occupation,
        )

    from_state, to_state, excited_electrons = excitation
    
    homo = jnp.array(jnp.ceil(electrons / spin_degeneracy), int) - 1

    # determine where and how often this energy occurs
    flags, degeneracy = _flags(homo)
    # compute electrons at the fermi level
    below = (energies < energies[homo] - eps).sum()
    remaining_electrons = electrons - spin_degeneracy * below
    # compute ground state occupations by distributing the remaining electrons across all degenerate levels
    occupation = _occupation(flags, spin_degeneracy, remaining_electrons / degeneracy)
    # excited states
    occupation = jax.lax.fori_loop(0, to_state.size, _body_fun, occupation)
    
    return jnp.diag(occupation) / electrons

def _mf_loop(hamiltonian,
             coulomb,
             excitation,
             spin_degeneracy,
             electrons,
             eps,
             mix,
             accuracy,
             coulomb_strength,
             iterations,
             f_mean_field,
             f_build,
             rho_0
             ):
    
    def update(arg):
        """scf update"""
        
        rho_old, step, error = arg

        # initial effective hamiltonian
        ham_eff = f_mean_field(rho_old, hamiltonian)
        
        # diagonalize
        vals, vecs = jnp.linalg.eigh(ham_eff)
        
        # build new density matrix (EXTREMELY INEFFICIENTLY)
        rho = spin_degeneracy * (1 - mix) * f_build(vecs, vals) + mix * rho_old

        # update breaks
        error = jnp.linalg.norm(rho - rho_old)
        
        step = jax.lax.cond(error <= accuracy, lambda x: step, lambda x: step + 1, step)

        return rho, step, error
    
    def step(idx, res):
        """single SCF update step"""
        return jax.lax.cond(res[-1] <= accuracy, lambda x: res, update, res)

    if f_mean_field is None:
        f_mean_field = lambda r, hamiltonian : jnp.diag(coulomb_strength * coulomb @ r.diagonal()) + hamiltonian

    if f_build is None:
        f_build = lambda vecs, energies : vecs[:, :electrons] @ vecs[:, :electrons].T        
        
    # initial guess for the density matrix
    if rho_0 is None:
        rho_0 = jnp.zeros_like(hamiltonian)
    rho_old = rho_0

    # scf loop
    rho, steps, error = jax.lax.fori_loop(0, iterations, step, (rho_old, 0, jnp.inf))

    # new hamiltonian
    ham_eff = f_mean_field(rho, hamiltonian)    
    energies, eigenvectors = jnp.linalg.eigh(ham_eff)

    # initial density matrix may be excited
    rho_0 = _density_aufbau(
        energies,
        electrons,
        spin_degeneracy,
        eps,
        excitation
    )

    print(f"Mean field finished with accuracy {accuracy}, iterations {steps}, yielding final error {error}")
    
    # orbital list builder expects density matrices in energy basis
    return (
        ham_eff,
        rho_0, # already in energy basis
        (eigenvectors.conj().T @ rho @ eigenvectors) / electrons, # convert
        energies,
        eigenvectors,
    )

def _get_self_consistent(
    hamiltonian,
    coulomb,
    positions,
    excitation,
    spin_degeneracy,
    electrons,
    eps,
    eigenvectors,
    rho_stat,
    iterations,
    mix,
    accuracy,
    coulomb_strength,    
):

    def _to_site_basis(ev, mat):
        return ev @ mat @ ev.conj().T

    def _phi(rho):
        return coulomb_strength * (coulomb @ jnp.diag(rho - rho_uniform))

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
        rho_energy = _density_aufbau(
            energies,
            electrons,
            spin_degeneracy,
            eps,
            [0*el for el in excitation],
        )

        return _to_site_basis(eigenvectors, rho_energy), rho, idx + 1

    # construct a uniform density matrix in real space
    _, arr, counts = jnp.unique(
        jnp.round(positions, 8), return_inverse=True, return_counts=True, axis=0
    )
    normalization = 2.0 if int(spin_degeneracy) == 1 else 1.0
    rho_uniform = jnp.diag((1 / counts)[arr[:, 0]]) / (hamiltonian.diagonal().size * normalization)
    eq_arr = jnp.array([0])

    # first induced potential
    rho_old = jnp.zeros_like(hamiltonian)
    rho = _to_site_basis(eigenvectors, rho_stat)

    # sc loop
    rho, rho_old, idx = jax.lax.while_loop(_stop, _loop, (rho, rho_old, 0))
    if idx == iterations - 1:
        raise ValueError("Self-consistent procedure did not converge!!")
    print(f"SC finished: {idx} / {iterations}")

    # new hamiltonian and initial state
    ham_new = hamiltonian + _phi(rho) * mix + _phi(rho_old) * (1 - mix)
    energies, eigenvectors = jnp.linalg.eigh(ham_new)
    rho_0 = _density_aufbau(
        energies,
        electrons,
        spin_degeneracy,
        eps,
        excitation
    )

    return (
        ham_new,
        rho_0,
        eigenvectors.conj().T @ rho @ eigenvectors,
        energies,
        eigenvectors,
    )


def get_coulomb_field_to_from(source_positions, target_positions, compute_at=None):
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
    if compute_at is None:
        return None

    # Calculate vector differences between each pair of source and target positions
    distance_vector = target_positions[:, None, :] - source_positions
    # Compute the norm of these vectors
    norms = jnp.linalg.norm(distance_vector, axis=-1)
    # Safe division by the cube of the norm
    one_over_distance_cubed = jnp.where(norms > 0, 1 / norms**3, 0)
    # Calculate the contributions
    coulomb_field_to_from = distance_vector * one_over_distance_cubed[:, :, None]
    # final array
    coulomb_field_to_from_final = jnp.zeros_like(coulomb_field_to_from)        
    # include only contributions where desired
    return coulomb_field_to_from_final.at[compute_at].set(
        coulomb_field_to_from[compute_at]
    )

def get_time_axis( mat_size, grid, start_time, end_time, max_mem_gb, dt ):

    # if grid is an array, sample these times, else subsample time axis
    time_axis = grid
    if not isinstance(grid, jax.Array):
        steps_time = jnp.ceil( (end_time - start_time) / dt ).astype(int)
        time_axis = jnp.linspace(start_time, end_time, steps_time)[::grid]

    # number of rhos in a single RAM batch 
    matrices_per_batch = jnp.floor( max_mem_gb / mat_size  ).astype(int).item()
    assert matrices_per_batch > 0, "Density matrix exceeds allowed max memory."

    # batch time axis accordingly, stretch to make array
    splits = jnp.ceil(time_axis.size /  matrices_per_batch ).astype(int).item()
    tmp = jnp.array_split(time_axis, [matrices_per_batch * i for i in range(1, splits)] )        
    if len(tmp[0]) != len(tmp[-1]):
        tmp[-1] = tmp[-2][-1] + (time_axis[1] - time_axis[0]) + tmp[0]

    return jnp.array( tmp )

def setup_rhs( hamiltonian_func, dissipator_func ):
    @jax.jit
    def rhs( time, density_matrix, args ):
        print("RHS compiled")
        h_total = sum(f(time, density_matrix, args) for f in hamiltonian_func)
        h_times_d = h_total @ density_matrix
        hermitian_term = -1j * (h_times_d  - h_times_d.conj().T)        
        return hermitian_term + sum(f(time, density_matrix, args) for f in dissipator_func)

    return rhs

def get_integrator( hamiltonian, dissipator, postprocesses, solver, stepsize_controller, dt):
    rhs = setup_rhs( hamiltonian, dissipator )
    term = diffrax.ODETerm(rhs)

    if not isinstance(solver, str):
        @jax.jit
        def integrator( d_ini, ts, args ):
            dms = diffrax.diffeqsolve(term,
                                      solver,
                                      t0=ts.min(),
                                      t1=ts.max(),
                                      dt0=dt,
                                      y0=d_ini,
                                      saveat=diffrax.SaveAt(ts=ts),
                                      stepsize_controller=stepsize_controller,
                                      args = args,
                                    ).ys
            
            return dms[-1], [ p(dms, args) for p in postprocesses ]
    
    elif solver == "Euler" :
        def integrator( d_ini, ts, args ):
            return  Euler_solver(rhs, ts, d_ini, args, postprocesses)
        
    elif solver == "RK45" :
        def integrator( d_ini, ts, args ):
            return RK45_solver(rhs, ts, d_ini, args, postprocesses)
        
    elif solver == "DP54" :
        def integrator( d_ini, ts, args ):
            return DP54_solver(rhs, ts, d_ini, args, postprocesses)
    elif solver == "adaptive_DP54":
        def integrator ( d_ini, ts, args ):
            return DP54_adaptive_solver2(rhs, ts, d_ini, args, postprocesses)

    else : raise ValueError(f"Invalid Solver {solver}. Available solvers are homemade 'RK45', 'DP54', 'Euler' and all diffrax solvers. Default solver diffrax.Dopri5() ")
    
    return integrator

def td_run(d_ini, integrator, time_axis, args):
    shapes_known = False
    for ts in time_axis:
        d_ini, res = integrator( d_ini, ts, args )
        if shapes_known == False:
            result = res
            shapes_known = True
        else:
            for i, res_part in enumerate(res):
                result[i] = jnp.concatenate( (result[i], res_part) )            
        print(f"{ts[-1] / jnp.max(time_axis) * 100} %")
    return d_ini, result 
    

def rpa_polarizability_function(
        args, polarization, hungry, phi_ext=None
):
    def _polarizability(omega):
        ro = sus(omega) @ phi_ext
        return -pos @ ro

    pos = args.positions[:, polarization]
    phi_ext = pos if phi_ext is None else phi_ext
    sus = rpa_susceptibility_function(args, hungry)
    return _polarizability


def rpa_susceptibility_function(args, hungry):
    def _rpa_susceptibility(omega):
        x = sus(omega)
        return x @ jnp.linalg.inv(one - args.coulomb_scaled @ x)

    sus = bare_susceptibility_function(args, hungry)        
    one = jnp.identity(args.hamiltonian.shape[0])

    return _rpa_susceptibility

def bare_susceptibility_function(args, hungry):

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
                        
            eq = spin_degeneracy * Sf / (omega - omega_grid_extended + 1j * relaxation_rate )
            return -jnp.sum(eq)
                
        if hungry == 2:
            return jax.vmap(
                jax.vmap(susceptibility_element, (0, None), 0), (None, 0), 0
            )(sites, sites)        
        elif hungry == 1:
            return jax.lax.map(
                lambda i: jax.vmap(lambda j: susceptibility_element(i, j), (0,), 0)(sites), 
                sites
            )
        else:
            return jax.lax.map(
                lambda i: jax.lax.map(lambda j: susceptibility_element(i, j), sites), 
                sites
            )
    
    # unpacking
    energies = args.energies.real
    eigenvectors = args.eigenvectors.real
    relaxation_rate = args.relaxation_rate
    occupation = jnp.diag(args.eigenvectors.conj().T @ args.initial_density_matrix @ args.eigenvectors).real * args.electrons / args.spin_degeneracy
    spin_degeneracy = args.spin_degeneracy
    sites = jnp.arange(energies.size)
    freq_number = 2**12
    omega_max = jnp.real(max(args.energies[-1], -args.energies[0])) + 0.1
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

def nest_result(result_list, shape):
    """
    Recursively reshapes a flat list into a nested list structure matching the Cartesian product shape.
    
    Parameters:
        result_list (list): The flat list of computed results.
        shape (list): A list representing the shape of the Cartesian product.
    
    Returns:
        list: A nested list following the Cartesian product shape.
    """
    if len(shape) == 1:
        return result_list  # Base case: last dimension

    chunk_size = int(len(result_list) / shape[0])  # Compute chunk size dynamically
    return [
        nest_result(result_list[i * chunk_size:(i + 1) * chunk_size], shape[1:]) 
        for i in range(shape[0])
    ]

def iterate(func):
    """
    A decorator that allows a function to iterate over list inputs.
    
    Functionality:
    1. If one or more of the function’s input arguments is a list, the function is executed for every combination of elements.
    2. If multiple list inputs are present, the computation follows a Cartesian product pattern:
       - For the 1st element of list A, iterate over all elements of list B.
       - For the 2nd element of list A, iterate over all elements of list B.
       - And so on.
    3. The results are reshaped into a nested structure matching the input lists.
    4. If the function returns multiple values (a tuple), each return value is separately structured into its own nested list.
    
    Parameters:
        func (callable): The function to be decorated.
    
    Returns:
        callable: The decorated function that processes list inputs correctly.
    """
    @functools.wraps(func)
    def inner(*args, **kwargs):
        result = []
        
        # Extract parameters that are lists
        dict_params = {key: values for key, values in kwargs.items() if isinstance(values, list)}
        shape = [len(values) for values in dict_params.values()]
        
        if dict_params:  # If any parameter is a list, perform Cartesian product iteration
            for combination in itertools.product(*dict_params.values()):
                new_kwargs = kwargs | dict(zip(dict_params.keys(), combination))
                result.append(func(*args, **new_kwargs))  # Call the original function
            
            # If function returns a tuple, separate results for each return value
            if isinstance(result[0], tuple):
                transposed_results = list(zip(*result))  # Split values across multiple lists
                return tuple(nest_result(list(r), shape) for r in transposed_results)
            else:
                return nest_result(result, shape)  # Single return value case
        
        return func(*args, **kwargs)  # Direct function call if no list parameters
    
    return inner

#================================= HOMEMADE SOLVERS ========================================================#

def Euler_solver(rhs_func, ts, d_ini, args, postprocesses):
    """Solves an ODE using the explicit Euler method with JAX acceleration.

    Args:
        rhs_func (callable): Function defining the ODE's right-hand side (dy/dt = rhs_func(t, y, args)).
        ts (array): Array of time points for the solution.
        d_ini (array): Initial condition(s) of the dependent variable(s).
        args (tuple): Additional arguments to pass to rhs_func.
        postprocesses (list): List of callable post-processing functions to apply to each step's solution.

    Returns:
        tuple: Solution array and list of post-processed results from jax.lax.scan.

    Notes:
        Implements the explicit Euler method: y[n+1] = y[n] + dt * rhs_func(t[n], y[n], args).
        Uses JAX's JIT compilation and scan for efficient computation.
    """
    dt = ts[1] - ts[0]
    #import pdb; pdb.set_trace()
    def rho_nxt_Euler(rho, t, rhs_func, args):
        rho_nxt = rho + rhs_func(t, rho, args) * dt
        return rho_nxt, [p(rho_nxt, args) for p in postprocesses]
    
    jitted_rho_nxt_Euler = jax.jit(rho_nxt_Euler, static_argnums=(2,))

    @jax.jit
    def jax_scan_compatible_rho_nxt_Euler(rho, t):
        return rho_nxt_Euler(rho, t, rhs_func=rhs_func, args=args)
        
    return jax.lax.scan(jax_scan_compatible_rho_nxt_Euler, d_ini, ts)

def RK45_solver(rhs_func, ts, d_ini, args, postprocesses):
    """Solves an ODE using the 4th-order Runge-Kutta (RK4) method with JAX acceleration.

    Args:
        rhs_func (callable): Function defining the ODE's right-hand side (dy/dt = rhs_func(t, y, args)).
        ts (array): Array of time points for the solution.
        d_ini (array): Initial condition(s) of the dependent variable(s).
        args (tuple): Additional arguments to pass to rhs_func.
        postprocesses (list): List of callable post-processing functions to apply to each step's solution.

    Returns:
        tuple: Solution array and list of post-processed results from jax.lax.scan.

    Notes:
        Implements the classical RK4 method with four stages (k1, k2, k3, k4).
        Next state: y[n+1] = y[n] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4).
        Optimized with JAX's JIT compilation and scan functionality.
    """
    dt = ts[1] - ts[0]
    print(dt)

    def rho_nxt_RK45(state, t, rhs_func, args):
        rho, k1 = state
        k1 = rhs_func(t, rho, args)
        k2 = rhs_func(t + dt/2, rho + (dt/2) * k1, args)
        k3 = rhs_func(t + dt/2, rho + (dt/2) * k2, args)
        k4 = rhs_func(t + dt, rho + dt * k3, args)
        rho_nxt = rho + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        state=(rho_nxt, k4)
        return state, [p(rho_nxt, args) for p in postprocesses]
        
    jitted_rho_nxt_RK45 = jax.jit(rho_nxt_RK45, static_argnums=(2,))

    @jax.jit
    def jax_scan_compatible_rho_nxt_RK45(state, t):
        return rho_nxt_RK45(state, t, rhs_func=rhs_func, args=args)

    k1 = rhs_func(ts[0], d_ini, args)
    initial_state=(d_ini, k1 )
    
    final_state, postprocessed_rhos=jax.lax.scan(jax_scan_compatible_rho_nxt_RK45, initial_state, ts)
    return final_state[0], postprocessed_rhos

def DP54_solver(rhs_func, ts, d_ini, args, postprocesses):
    """Solves an ODE using the Dormand-Prince 5(4) method with JAX acceleration.

    Args:
        rhs_func (callable): Function defining the ODE's right-hand side (dy/dt = rhs_func(t, y, args)).
        ts (array): Array of time points for the solution.
        d_ini (array): Initial condition(s) of the dependent variable(s).
        args (tuple): Additional arguments to pass to rhs_func.
        postprocesses (list): List of callable post-processing functions to apply to each step's solution.

    Returns:
        tuple: Solution array and list of post-processed results from jax.lax.scan.

    Notes:
        Implements the Dormand-Prince 5(4) method with seven stages and 5th-order accuracy.
        Uses coefficients from the Dormand-Prince tableau for high precision.
        Optimized with JAX's JIT compilation and scan functionality.
    """
    dt = ts[1] - ts[0]
    print(dt)
    def rho_nxt_DP54(state, t, rhs_func, args):
        # Coefficients from Dormand-Prince 5(4) tableau
        #Copied from: https://numerary.readthedocs.io/en/latest/dormand-prince-method.html
        a21 = 1/5
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
        b1, b2, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
        
        
        rho, k1 = state
        #k1 = rhs_func(t, rho, args)
        k2 = rhs_func(t + dt * a21, rho + dt * a21 * k1, args)
        k3 = rhs_func(t + dt * 3/10, rho + dt * (a31 * k1 + a32 * k2), args)
        k4 = rhs_func(t + dt * 4/5, rho + dt * (a41 * k1 + a42 * k2 + a43 * k3), args)
        k5 = rhs_func(t + dt * 1.0, rho + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), args)
        k6 = rhs_func(t + dt * 1.0, rho + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), args)
        k7 = rhs_func(t + dt * 1.0, rho + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6), args)
        
        rho_nxt = rho + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)
        state=(rho_nxt, k7)
        
        return state, [p(rho_nxt, args) for p in postprocesses]

    jitted_rho_nxt_DP54 = jax.jit(rho_nxt_DP54, static_argnums=(2,))

    @jax.jit
    def jax_scan_compatible_rho_nxt_DP54(state, t):
        return rho_nxt_DP54(state, t, rhs_func=rhs_func, args=args)
    k1 = rhs_func(ts[0], d_ini, args)
    initial_state=(d_ini, k1 )
    final_state, postprocessed_rhos = jax.lax.scan(jax_scan_compatible_rho_nxt_DP54, initial_state, ts)    
    return final_state[0], postprocessed_rhos
