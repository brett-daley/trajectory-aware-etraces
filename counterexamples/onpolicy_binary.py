import numpy as np


def max_norm(Z):
    return np.max(np.abs(Z).sum(axis=1))


def main():
    P_pi = 0.5 * np.ones([2, 2])
    gamma = 2/3

    # Given our definition of P_pi, B_t is constant for all t >= 1
    B_t = 0.5 * np.array([[1, 0], [1, 0]])

    # Calculate Z
    Z = np.zeros_like(B_t)
    N = 1_000  # Number of terms for approximating the infinite geometric series
    for t in range(1, N + 1):
        # B_0 = I
        # B_{t-1} = B_t for t > 1
        B_tm1 = np.eye(*B_t.shape) if (t == 1) else B_t.copy()

        Z += pow(gamma, t) * ((B_tm1 @ P_pi) - B_t)

    print('Z =\n', Z)
    print()
    print('||Z|| =', max_norm(Z))  # Should be 1, may have small rounding error


if __name__ == '__main__':
    main()
