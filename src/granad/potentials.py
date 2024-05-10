import jax
import jax.numpy as jnp

# TODO: not really pythonic naming style ...    
def Wave( amplitudes, omega, sigma, t0, kick = False ):
    """Wave potential"""
    def pot( t, r, args ):        
        return
    
def Dipole( amplitudes, omega, sigma, t0, kick = False ):
    """Dipole potential"""
    def pot( t, r, args ):        
        return
    
# TODO: clean up, RWA messy, allocates 3 potentially big intermediate matrices
def DipoleGauge(illumination, use_rwa = False, intra_only = False):     
    """Dipole gauge coupling to external E-field $\sim \vec{E} \hat{\vec{P}}$"""
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

def Induced( add_induced = False ):
    """Induced potential. Propagates induced charges $\sim \sum_r q_r/|r-r'|$"""
    if add_induced == False:
        return lambda t, r, args : 0.0
    def inner(t, r, args):
        field = jnp.einsum("ijK,j->iK", args.propagator, -args.electrons*r.diagonal())
        return jnp.einsum("Kij,iK->ij", args.dipole_operator, field.real)
    return inner

def Paramagnetic(vector_potential):
    """Paramagnetic Coulomb gauge coupling to external vector potential $\sim A \hat{\vec{v}}$"""
    def inner(t, r, args):
        # ~ A p
        q = 1
        return -q * jnp.einsum("Kij, iK -> ij", args.velocity_operator, vector_potential(t))
    return inner

def Diamagnetic(vector_potential):
    """Diamagnetic Coulomb gauge coupling to external vector potential $\sim A^2$"""
    def inner(t, r, args):
        # ~ A^2
        q = m = 1
        return jnp.diag(q**2 / m * 0.5 * jnp.sum(vector_potential(t)**2, axis=1))
    return inner

def Coulomb():
    """Induced coulomb potential $\sim \lambda C(\rho-\rho_0)$, where $\lambda$ is a scaling factor."""
    return lambda t, r, args: jnp.diag(args.coulomb_scaled @ (r-args.stationary_density_matrix).diagonal() * args.electrons )

def BareHamiltonian():
    """Unperturbed single particle TB MF Hamiltonian $= h^{(0)}$"""
    return lambda t, r, args: args.hamiltonian

