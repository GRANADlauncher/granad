import jax
import jax.numpy as jnp

def quadrupole_operator(stack):        
    dip = position_operator(stack)
    term = jnp.einsum("ijk,jlm->ilkm", dip, dip)
    diag = jnp.einsum("ijk,jlk->il", dip, dip)
    diag = jnp.einsum("ij,kl->ijkl", diag, jnp.eye(term.shape[-1]))
    return 3 * term - diag


def position_operator(stack):
    N = stack.positions.shape[0]
    pos = jnp.zeros((N, N, 3))
    for i in range(3):
        pos = pos.at[:, :, i].set(jnp.diag(stack.positions[:, i] / 2))
    if not stack.transitions is None:
        for key, value in stack.transitions.items():
            value = jnp.array(value)
            i, j = indices(stack, key[0]), indices(stack, key[1])
            k = value.nonzero()[0]
            pos = pos.at[i, j, k].set(value[k])
    return pos + jnp.transpose(pos, (1, 0, 2))


def velocity_operator(stack):
    if stack.transitions is None:
        x_times_h = jnp.einsum("ij,ir->ijr", stack.hamiltonian, stack.positions)
        h_times_x = jnp.einsum("ij,jr->ijr", stack.hamiltonian, stack.positions)
    else:
        positions = position_operator(stack)
        x_times_h = jnp.einsum("kj,ikr->ijr", stack.hamiltonian, positions)
        h_times_x = jnp.einsum("ik,kjr->ijr", stack.hamiltonian, positions)
    return -1j * (x_times_h - h_times_x)



def dos(stack, omega: float, broadening: float = 0.1) -> jax.Array:
    """IP-DOS of a nanomaterial stack.

    - `stack`: a stack object
    - `omega`: frequency
    - `broadening`: numerical brodening parameter to replace Dirac Deltas with
    """

    broadening = 1 / broadening
    prefactor = 1 / (jnp.sqrt(2 * jnp.pi) * broadening)
    gaussians = jnp.exp(-((stack.energies - omega) ** 2) / 2 * broadening**2)
    return prefactor * jnp.sum(gaussians)


def ldos(
    stack, omega: float, site_index: int, broadening: float = 0.1
) -> jax.Array:
    """IP-LDOS of a nanomaterial stack.

    - `stack`: a stack object
    - `omega`: frequency
    - `site_index`: the site index to evaluate the LDOS at
    - `broadening`: numerical brodening parameter to replace Dirac Deltas with
    """

    broadening = 1 / broadening
    weight = jnp.abs(stack.eigenvectors[site_index, :]) ** 2
    prefactor = 1 / (jnp.sqrt(2 * jnp.pi) * broadening)
    gaussians = jnp.exp(-((stack.energies - omega) ** 2) / 2 * broadening**2)
    return prefactor * jnp.sum(weight * gaussians)


def transition_energies(stack) -> jax.Array:
    """Computes independent-particle transition energies associated with the TB-Hamiltonian of a stack.

    - `stack`:

    **Returns:**

    array, the element `arr[i,j]` contains the transition energy from `i` to `j`
    """
    return jnp.abs(jnp.expand_dims(stack.energies, 1) - stack.energies)


def wigner_weisskopf(stack, component: int = 0) -> jax.Array:
    """Calculcates Wigner-Weisskopf transiton rates.

        - `stack`:
        - `component`: component of the dipolar transition to take $(0,1,2) \rightarrow (x,y,z)$`


    **Returns:**

     array, the element `arr[i,j]` contains the transition rate from `i` to `j`
    """
    charge = 1.602e-19
    eps_0 = 8.85 * 1e-12
    hbar = 1.0545718 * 1e-34
    c = 3e8  # 137 (a.u.)
    factor = 1.6e-29 * charge / (3 * jnp.pi * eps_0 * hbar**2 * c**3)
    te = transition_energies(stack)
    return (
        (te * (te > stack.eps)) ** 3
        * jnp.squeeze(transition_dipole_moments(stack)[:, :, component] ** 2)
        * factor
    )


# TODO: this should be changed in accordance with the newly defined position operator
def transition_dipole_moments(stack) -> jax.Array:
    r"""Compute transition dipole moments for all states $i,j$ as $<i | \hat{r} | j>$.

        - `stack`: stack object with N orbitals

    **Returns:**

     dipole moments as a complex $N \\times N \\times 3$ - matrix, where the last component is the direction of the dipole moment.
    """
    return jnp.einsum(
        "li,lj,lk", stack.eigenvectors.conj(), stack.eigenvectors, stack.positions
    ) * jnp.expand_dims(
        jnp.ones_like(stack.eigenvectors) - jnp.eye(stack.eigenvectors.shape[0]), 2
    )


## INTERACTION
def epi(stack, rho: jax.Array, omega: float, epsilon: float = None) -> float:
    r"""Calculates the EPI (Energy-based plasmonicity index) of a mode at $\hbar\omega$ in the absorption spectrum of a structure.

        - `stack`: stack object
        - `rho`: density matrix
        - `omega`: energy at which the system has been CW-illuminated ($\hbar\omega$ in eV)
        - `epsilon`: small broadening parameter to ensure numerical stability, if `None`, stack.eps is chosen

    **Returns:**

    , a number between `0` (single-particle-like) and `1` (plasmonic).
    """
    epsilon = stack.eps if epsilon is None else epsilon
    rho_without_diagonal = jnp.abs(rho - jnp.diag(jnp.diag(rho)))
    rho_normalized = rho_without_diagonal / jnp.linalg.norm(rho_without_diagonal)
    te = transition_energies(stack)
    excitonic_transitions = (
        rho_normalized / (te * (te > stack.eps) - omega + 1j * epsilon) ** 2
    )
    return 1 - jnp.sum(jnp.abs(excitonic_transitions * rho_normalized)) / (
        jnp.linalg.norm(rho_normalized) * jnp.linalg.norm(excitonic_transitions)
    )

def indices(stack, orbital_id: str) -> jax.Array:
    """Gets indices of a specific orbital.

        Can be used to calculate, e.g. positions and energies corresponding to that orbital in the stack.

        - `stack`: stack object
        - `orbital_id`: orbital identifier as contained in :class:`granad.numerics.Stack.unique_ids`

    **Returns:**

     array corresponding to the indices of the orbitals, such that e.g. `stack.energies[indices]` gives the energies associated with the orbitals.
    """
    return jnp.nonzero(stack.ids == stack.unique_ids.index(orbital_id))[0]


# TODO: this should be changed in accordance with the newly defined position operator
def induced_dipole_moment(stack, rhos_diag: jax.Array) -> jax.Array:
    """
        Calculates the induced dipole moment for a collection of density matrices.

        - `stack`: stack object
        - `rhos_diag`: $N \\times m$ time-dependent site occupation matrix, indexed by rhos_diag[timestep,site_number]

    **Returns:**

    , $N \\times 3$ matrix containing the induced dipole moment $(p_x, p_y, p_z)$ at $N$ times
    """
    return (
        (jnp.diag(to_site_basis(stack, stack.rho_stat)) - rhos_diag).real
        @ stack.positions
        * stack.electrons
    )


def induced_field(
    stack, positions: jax.Array, density_matrix: jax.Array = None
) -> jax.Array:
    """Classical approximation to the induced (local) field in a stack.

    - `stack`: a stack object
    - `positions`: positions to evaluate the field on, must be of shape N x 3
    :density_matrix: if given, compute the field corresponding to this density matrix. otherwise, use `stack.rho_0`.
    """

    # determine whether to use argument or the state of the stack
    density_matrix = density_matrix if density_matrix is not None else stack.rho_0

    # distance vector array from field sources to positions to evaluate field on
    vec_r = stack.positions[:, None] - positions

    # scalar distances
    denominator = jnp.linalg.norm(vec_r, axis=2) ** 3

    # normalize distance vector array
    point_charge = jnp.nan_to_num(
        vec_r / denominator[:, :, None], posinf=0.0, neginf=0.0
    )

    # compute charge via occupations in site basis
    charge = stack.electrons * jnp.diag(to_site_basis(stack, density_matrix)).real

    # induced field is a sum of point charges, i.e. \vec{r} / r^3
    e_field = 14.39 * jnp.sum(point_charge * charge[:, None, None], axis=0)
    return e_field
