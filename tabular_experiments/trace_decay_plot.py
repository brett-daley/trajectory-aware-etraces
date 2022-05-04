from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Max episode length is 9, plus 1 for t=0
    MAXLEN = 10
    X = np.arange(MAXLEN)

    plt.figure()
    for estimator in ['Retrace', 'IS']:
        trajectories = []
        for seed in range(5):
            eligibilities = np.loadtxt(f'data/{estimator}_{seed}.txt')
            trajectories.append(eligibilities[:MAXLEN])

        Y = np.mean(trajectories, axis=0)
        error = np.std(trajectories, axis=0, ddof=1)
        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, Y - error, Y + error, alpha=0.25, linewidth=0)

    # Plot theoretical Q(lambda)
    plt.plot(X, np.power(0.81, X), label=r"Q($\lambda$)")

    plt.xlim([0, MAXLEN - 1])
    plt.ylim([0.0, 1.0])
    plt.savefig('trace_decay_average.png')
