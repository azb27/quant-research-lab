
import numpy as np
from projects.statarb_kalman import kalman_regression, KalmanParams

def test_kalman_tracks_linear_relation():
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.standard_normal(n).cumsum()
    true_a, true_b = 0.5, 1.8
    y = true_a + true_b*x + 0.1*rng.standard_normal(n)

    a, b = kalman_regression(x, y, KalmanParams(q_alpha=1e-6, q_beta=1e-6, r_obs=1e-2))
    assert abs(np.median(b[-200:]) - true_b) < 0.1
