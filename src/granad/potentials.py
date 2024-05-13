import jax
import jax.numpy as jnp

# TODO: not really pythonic naming style ...        
def DipolePulse( dipole_moment, source_location, omega = None, sigma = None, t0 = 0.0, kick = False ):
    """Function to compute the potential due to a pulsed dipole. The potential can optionally include a 'kick' which is an instantaneous spike at a specific time.
    If the dipole is placed at a position occupied by orbitals, its contribution will be set to zero.

    Args:
        dipole_moment: Vector representing the dipole moment in xyz-components.
        source_location: Location of the source of the dipole in xyz-coordinates.
        omega: Angular frequency of the oscillation (default is None).
        sigma: Standard deviation of the pulse's temporal Gaussian profile (default is None).
        t0: Time at which the pulse is centered (default is 0.0).
        kick: If True, lets the spatial profile of the dipole kick only at time t0 (default is False) and discards omega, sigma.

    Returns:
        Function that computes the dipole potential at a given time and location, with adjustments for distance and orientation relative to the dipole.

    Note:
       Recommended only with solver=diffrax.Dopri8.
    """
    
    loc = jnp.array( source_location )[:,None]
    dip = jnp.array( dipole_moment )
    f = lambda t : jnp.cos(omega * t) * jnp.exp( -(t-t0)**2 / sigma**2 ) 
    if kick == True:
        f = lambda t : jnp.abs(t - t0) < 1e-10
    
    def pot( t, r, args ):
        distances = args.dipole_operator.diagonal(axis1=-1, axis2=-2) - loc
        r_term = (dip @ distances) / jnp.linalg.norm( distances, axis = 0 )
        return jnp.diag( jnp.nan_to_num(r_term) * f(t) )

    return pot

def WavePulse( amplitudes, omega = None, sigma = None, t0 = 0.0, kick = False ):
    """Function to compute the wave potential using amplitude modulation. This function creates a pulse with temporal Gaussian characteristics and can include an optional 'kick' which introduces an instantaneous amplitude peak.

    Args:
        amplitudes: List of amplitudes for the wave components in xyz-directions.
        omega: Angular frequency of the wave oscillation (default is None).
        sigma: Standard deviation of the Gaussian pulse in time (default is None).
        t0: Central time around which the pulse peaks (default is 0.0).
        kick: If True, lets the spatial profile of the wave kick only at time t0 (default is False) and discards omega, sigma.

    Returns:
        Function that computes the potential at a given time and location, incorporating the wave characteristics and specified modulations.

    Note:
       This function, when not kicked, computes the same term as `Pulse`.
    """
    amplitudes = jnp.array(amplitudes)
    
    f = lambda t : amplitudes * jnp.cos(omega * t) * jnp.exp( -(t-t0)**2 / sigma**2 ) 
    if kick == True:
        f = lambda t : amplitudes * (jnp.abs(t - t0) < 1e-10)
    
    def pot( t, r, args ):
        return jnp.einsum('Kij,K->ij', args.dipole_operator, f(t))

    return pot
    
# TODO: clean up, RWA messy, allocates 3 potentially big intermediate matrices
def DipoleGauge(illumination, use_rwa = False, intra_only = False):     
    """Dipole gauge coupling to an external electric field, represented as $E \cdot \hat{P}$. The dipole / polarization operator is
    defined by $P^{c}_{ij} = <i|\hat{r}_c|j>$, where $i,j$ correspond to localized (TB) orbitals, such that $\hat{r}^c|i> = r^c{i}|i>$ in absence of dipole transitions.

    Args:
        illumination (callable): Function that returns the electric field at a given time.
        use_rwa (bool): If True, uses the rotating wave approximation which simplifies the calculations by considering only resonant terms.
        intra_only (bool): If True, subtracts the diagonal of the potential matrix, focusing only on the interactions between different elements.

    Returns:
        Function: Computes the electric potential based on the given illumination and options for RWA and intramolecular interactions.
    """
    def electric_potential(t, r, args):
        return jnp.einsum(einsum_string, args.dipole_operator, illumination(t).real)

    def electric_potential_rwa(t, r, args):
        # the missing real part is crucial here! the RWA (for real dipole moments) makes the fields complex and divides by 2
        total_field_potential = jnp.einsum(einsum_string, args.dipole_operator, illumination(t))

        # Get the indices for the lower triangle, excluding the diagonal
        lower_indices = jnp.tril_indices(total_field_potential.shape[0], -1)

        # Replace elements in the lower triangle with their complex conjugates    
        tmp = total_field_potential.at[lower_indices].set( jnp.conj(total_field_potential[lower_indices]) )

        # make hermitian again
        return tmp - 1j*jnp.diag(tmp.diagonal().imag)

    # TODO: redundant evaluation
    maybe_diag = lambda f : f
    if intra_only == True:
        maybe_diag = lambda f : lambda t, r, args : f(t,r,args) - jnp.diag( f(t,r,args).diagonal() )
        
    einsum_string =  'Kij,K->ij' if illumination(0.).shape == (3,) else 'Kij,iK->ij'    

    if use_rwa == True:
        return maybe_diag(electric_potential_rwa)
    return maybe_diag(electric_potential)

def Induced():
    """Calculates the induced potential, which propagates the coulomb effect of induced charges in the system according to $\sim \sum_r q_r/|r-r'|$.

    Returns:
        Function: Computes the induced potential at a given time and location based on charge propagation.
    """
    def inner(t, r, args):
        field = jnp.einsum("ijK,j->iK", args.propagator, -args.electrons*r.diagonal())
        return jnp.einsum("Kij,iK->ij", args.dipole_operator, field.real)
    return inner

def Paramagnetic(vector_potential):
    """Paramagnetic Coulomb gauge coupling to an external vector potential represented as $\sim A \hat{v}$. 

    Args:
        vector_potential (callable): Function that returns the vector potential at a given time.

    Returns:
        Function: Computes the interaction of the vector potential with the velocity operator.
    """
    def inner(t, r, args):
        # ~ A p
        q = 1
        return -q * jnp.einsum("Kij, iK -> ij", args.velocity_operator, vector_potential(t))
    return inner

def Diamagnetic(vector_potential):
    """Diamagnetic Coulomb gauge coupling to an external vector potential represented as $\sim A^2$. 

    Args:
        vector_potential (callable): Function that returns the vector potential at a given time.

    Returns:
        Function: Computes the square of the vector potential, representing diamagnetic interactions.
    """
    def inner(t, r, args):
        # ~ A^2
        q = m = 1
        return jnp.diag(q**2 / m * 0.5 * jnp.sum(vector_potential(t)**2, axis=1))
    return inner

def Coulomb():
    """Calculates the induced Coulomb potential based on deviations from a stationary density matrix, represented as $\sim \lambda C(\\rho-\\rho_0)$. Here, $\lambda$ is a scaling factor.

    Returns:
        Function: Computes the Coulomb interaction scaled by deviations from the stationary state.
    """
    return lambda t, r, args: jnp.diag(args.coulomb_scaled @ (r-args.stationary_density_matrix).diagonal() * args.electrons )

def BareHamiltonian():
    """Represents the unperturbed single-particle tight-binding mean field Hamiltonian, denoted as $= h^{(0)}$.

    Returns:
        Function: Provides the bare Hamiltonian matrix, representing the unperturbed state of the system.
    """
    return lambda t, r, args: args.hamiltonian
  
