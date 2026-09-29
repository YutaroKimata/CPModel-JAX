import runpy
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cpmodel_jax import newton_krylov, solve_eha

example = runpy.run_path(Path(__file__).resolve().parents[1] / "examples/nafta_example.py")

ARCHIVE_PATH = example["ARCHIVE_PATH"]
NAFTA = example["NAFTA"]
load_nafta = example["load_nafta"]
welfare_decomposition = example["welfare_decomposition"]

PUBLISHED_WELFARE = np.array([
    [1.31, -0.41, 1.72, 1.72],
    [-0.06, -0.11, 0.04, 0.32],
    [0.08, 0.04, 0.04, 0.11],
])


def test_nafta_replication():
    inputs = load_nafta(ARCHIVE_PATH)
    result = solve_eha(**inputs)

    welfare = welfare_decomposition(inputs, result)[list(NAFTA.values())]

    np.testing.assert_allclose(
        welfare,
        PUBLISHED_WELFARE,
        atol=0.005,
        rtol=0,
    )
    assert max(result["diagnostics"].values()) < 2e-5



def test_newton_krylov():
    def linear(z, data):
        return data["matrix"] @ z - data["target"]

    expected = jnp.array([100.0, -80.0])
    matrix = jnp.array([[2.0, 1.0], [-1.0, 3.0]])
    data = {"matrix": matrix, "target": matrix @ expected}

    result = newton_krylov(linear, data, jnp.zeros(2))
    np.testing.assert_allclose(result["z"], expected, rtol=1e-6)

    def exponential(z, data):
        return jnp.exp(z) - data

    result = newton_krylov(exponential, jnp.array([2.0]), jnp.array([-3.0]))
    np.testing.assert_allclose(result["z"], np.log(2), atol=3e-6)
