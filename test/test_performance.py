import unittest

import jax
import jax.numpy as jnp
import numpy as np

from granad import _numerics
from granad.periodic import Periodic


class PerformanceRegressionTests(unittest.TestCase):
    def test_conductivity_matches_full_jacobian(self):
        energies = jnp.array([[-1000., -.3], [.2, 1000.], [-.1, .4]])
        velocities = jnp.arange(36.).reshape(3, 3, 2, 2) * (1 + .2j)
        omegas = jnp.array([.1, .7])
        for beta in (0., 1., 100., jnp.inf):
            with self.subTest(beta=beta):
                derivative = jax.jacrev(_numerics.fermi)(energies, beta, 0.)
                expected = (
                    (1j / (omegas + .1j))[:, None, None]
                    * jnp.einsum('kiki,ckii,dkii->cd', -derivative, velocities, velocities)
                    / energies.shape[0]
                )
                actual = Periodic.get_ip_conductivity_intra(
                    velocities, energies, omegas, 0., beta
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(actual.shape, (2, 3, 3))

    def test_sweep_preserves_results_and_compiled_function(self):
        traces = []

        @jax.jit
        def compiled(x):
            traces.append(True)
            return x * 2

        @_numerics.iterate
        def calculate(x, offset=0):
            return compiled(jnp.asarray(x)) + offset

        result = calculate(x=[1., 2.], offset=[10., 20.])
        np.testing.assert_array_equal(result, [[12., 22.], [14., 24.]])
        self.assertEqual(len(traces), 1)
        self.assertEqual(float(calculate(x=3.)), 6.)

        @_numerics.iterate
        def pair(x):
            return x, -x

        self.assertEqual(pair(x=[1, 2]), ([1, 2], [-1, -2]))

    def test_batch_outputs_and_final_state(self):
        times = jnp.array([[0., 1.], [2., 3.], [4., 5.]])
        for batches in (times[:1], times):
            for outputs in (0, 2):
                with self.subTest(batches=len(batches), outputs=outputs):
                    def integrate(state, ts, args):
                        values = state + jnp.arange(1, len(ts) + 1)
                        return values[-1], [values, values[:, None] * 2][:outputs]

                    final, result = _numerics.td_run(0., integrate, batches, None)
                    expected = jnp.arange(1, batches.size + 1)
                    self.assertEqual(float(final), batches.size)
                    self.assertEqual(len(result), outputs)
                    if outputs:
                        np.testing.assert_array_equal(result[0], expected)
                        np.testing.assert_array_equal(result[1], expected[:, None] * 2)


if __name__ == '__main__':
    unittest.main()
