from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Max episode length is 9, plus 1 for t=0
    MAXLEN = 10
    X = np.arange(MAXLEN)

    plt.figure()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rcParams.update({'font.size': 22})

    for estimator in ['Retrace', 'Moretrace', 'IS']:
        trajectories = []
        for seed in range(24):
            eligibilities = np.loadtxt(f'data/{estimator}_{seed}.txt')
            trajectories.append(eligibilities[:MAXLEN])

        Y = np.mean(trajectories, axis=0)
        n = len(trajectories)
        # 95% confidence interval
        error = 1.96 * np.std(trajectories, axis=0, ddof=1) / np.sqrt(n)
        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, Y - error, Y + error, alpha=0.25, linewidth=0)

    plt.title("19-State Random Walk")
    plt.xlabel("Timesteps")
    plt.ylabel("Eligibility of Initial State")

    plt.xlim([0, MAXLEN - 1])
    plt.ylim([0.0, 1.0])
    plt.grid(visible=True, which='both', axis='both')

    plt.legend(loc='best', framealpha=1.0, fontsize=16)

    ax = plt.gca()
    ax.set_aspect(1.0 / ax.get_data_ratio())
    plt.gcf().set_size_inches(6.4, 6.4)

    plt.tight_layout(pad=0.05)
    plt.savefig('trace_decay_average.png')
    # plt.savefig('trace_decay_average.pdf', format='pdf')
