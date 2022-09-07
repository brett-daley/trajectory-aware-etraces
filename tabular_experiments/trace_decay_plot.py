from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from alpha_sweep import preformat_plots, postformat_plots


if __name__ == '__main__':
    # Max episode length is 9, plus 1 for t=0
    MAXLEN = 10
    X = np.arange(MAXLEN)

    preformat_plots()
    plt.figure()

    n = 100
    for estimator in ['Retrace', 'Truncated IS', 'Recursive Retrace', 'Moretrace']:
        trajectories = []
        for seed in range(n):
            eligibilities = np.loadtxt(f"data/{estimator.replace(' ', '')}_{seed}.txt")
            trajectories.append(eligibilities[:MAXLEN])

        Y = np.mean(trajectories, axis=0)
        assert len(trajectories) == n
        # 95% confidence interval
        error = 1.96 * np.std(trajectories, axis=0, ddof=1) / np.sqrt(n)
        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, Y - error, Y + error, alpha=0.25, linewidth=0)

    plt.title("Grid Walk")
    plt.xlabel("Timesteps")
    plt.ylabel("Eligibility of Initial State")

    plt.xlim([0, MAXLEN - 1])
    plt.ylim([0.0, 1.0])

    postformat_plots()

    plt.savefig('trace_decay_average.png')
    plt.savefig('trace_decay_average.pdf', format='pdf')
