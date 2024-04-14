import jax.numpy as jnp

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
