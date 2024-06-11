import jax
import jax.numpy as jnp

def DecoherenceTime():
    """Function for modelling dissipation according to the relaxation approximation.
    """
    return lambda t,r,args: -(r - args.stationary_density_matrix) * args.relaxation_rate

def SaturationLindblad(saturation):
    """Function for modelling dissipation according to the saturated lindblad equation as detailed in https://link.aps.org/doi/10.1103/PhysRevA.109.022237.
    """
    saturation = jax.vmap(saturation, 0, 0)

    def inner(t, r, args):
        # convert rho to energy basis
        r = args.eigenvectors.conj().T @ r @ args.eigenvectors

        # extract occupations
        diag = jnp.diag(r) * args.electrons

        # apply the saturation functional to turn off elements in the gamma matrix
        gamma = args.relaxation_rate.astype(complex) * saturation(diag)[None, :]

        a = jnp.diag(gamma.T @ jnp.diag(r))
        mat = jnp.diag(jnp.sum(gamma, axis=1))
        b = -1 / 2 * (mat @ r + r @ mat)
        val = a + b

        return args.eigenvectors @ val @ args.eigenvectors.conj().T

    return inner
