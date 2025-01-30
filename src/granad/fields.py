import jax
import jax.numpy as jnp

def Wave(
    amplitudes: list[float],
    frequency: float,
):
    """Function for computing time-harmonic electric fields.

    Args:
        amplitudes: electric field amplitudes in xyz-components
        frequency: frequency

    Returns:
       Function that computes the electric field as a functon of time
    """
    def _field(t, real = True):
        val = (jnp.exp(1j * frequency * t) * static_part)
        return val.real if real else val
    static_part = jnp.array(amplitudes)
    return _field

def Ramp(
    amplitudes: list[float],
    frequency: float,
    ramp_duration: float,
    time_ramp: float,
):
    """Function for computing ramping up time-harmonic electric fields.

    Args:
        amplitudes: electric field amplitudes in xyz-components
        frequency: frequency
        ramp_duration: specifies how long does the electric field ramps up
        time_ramp: specifies time at which the field starts to ramp up

    Returns:
       Function that computes the electric field as a functon of time
    """
    def _field(t, real = True):        
        val =  (
            static_part
            * jnp.exp(1j * frequency * t)
            / (1 + 1.0 * jnp.exp(-ramp_constant * (t - time_ramp)))
        )
        return val.real if real else val
    static_part = jnp.array(amplitudes)
    p = 0.99
    ramp_constant = 2 * jnp.log(p / (1 - p)) / ramp_duration
    return _field


def Pulse(
    amplitudes: list[float],
    frequency: float,
    peak: float,
    fwhm: float,
):
    """Function for computing temporally located time-harmonics electric fields. The pulse is implemented as a temporal Gaussian.

    Args:
        amplitudes: electric field amplitudes in xyz-components
        frequency: frequency of the electric field
        peak: time where the pulse reaches its peak
        fwhm: full width at half maximum

    Returns:
       Function that computes the electric field
    """
    def _field(t, real = True):
        val = (
            static_part
            * jnp.exp(-1j * jnp.pi / 2 + 1j * frequency * (t - peak))
            * jnp.exp(-((t - peak) ** 2) / sigma**2)
        )
        return val.real if real else val
    static_part = jnp.array(amplitudes)
    sigma = fwhm / (2.0 * jnp.sqrt(jnp.log(2)))
    return _field
