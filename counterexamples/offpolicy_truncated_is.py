import numpy as np
from scipy.special import comb as combination

import sys
sys.path.append('.')
from counterexamples.onpolicy_binary import max_norm


def expected_beta(t, prob_pi):
    pr = prob_pi  # For brevity
    assert 0 < pr < 1

    beta = 0.0
    for k in range(t):  # For k in [0, t-1]
        behavior_prob = pow(0.5, t-1)
        target_prob = pow(pr, k) * pow(1-pr, t-1-k)

        # Truncated IS:
        beta += combination(t-1, k) * min(behavior_prob, 2 * pr * target_prob)

        # IS sanity check: should give us ||Z||=0
        # beta += combination(t-1, k) * 2 * pr * target_prob

    return beta


def main():
    p = 0.6
    gamma = 0.94

    P_pi = np.array([[p, 1-p], [p, 1-p]])
    B_tm1 = np.eye(*P_pi.shape)  # B_0 = I

    # Calculate Z
    Z = np.zeros_like(B_tm1)
    N = 1_000  # Number of terms for approximating the infinite geometric series
    for t in range(1, N + 1):
        beta1 = expected_beta(t, p)
        beta2 = expected_beta(t, 1-p)
        B_t = 0.5 * np.array([[beta1, beta2], [beta1, beta2]])

        Z += pow(gamma, t) * ((B_tm1 @ P_pi) - B_t)

        if np.isnan(Z).any():
            raise RuntimeError("nan detected, N may be too large")

        # Set up B_{t-1} for the next iteration
        B_tm1 = B_t.copy()

    print('Z =\n', Z)
    print()
    print('||Z|| =', max_norm(Z))  # Should be 1, may have small rounding error


if __name__ == '__main__':
    main()
