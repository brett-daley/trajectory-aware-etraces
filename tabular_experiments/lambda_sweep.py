from collections import defaultdict
import os

import gym_classics
import matplotlib.pyplot as plt
import numpy as np

from training import run_sweep_V, run_sweep_Q


if __name__ == '__main__':
    discount = 1.0
    return_estimators = ['Retrace', 'Moretrace']
    lambda_values = [0.025 * i for i in range(41)]
    learning_rates = np.linspace(0, 1, 26)
    seeds = range(100)

    # behavior_policy = np.array([0.5, 0.5])
    # target_policy = np.array([0.5, 0.5])
    # results = run_sweep_V('19Walk-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    behavior_policy = np.array([0.6, 0.4])
    target_policy = np.array([0.5, 0.5])
    results = run_sweep_Q('19Walk-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    # Plot Best RMS vs Lambda
    plt.figure()
    plt.xlim([0, 1])
    plt.xticks(np.linspace(0.0, 1.0, 10 + 1))
    plt.ylim([0.0, 0.35])

    for estimator in return_estimators:
        X, Y, ERROR = [], [], []
        for lambd in lambda_values:
            X.append(lambd)
            means, errors = zip(*(
                results[(estimator, lambd, lr)] for lr in learning_rates)
            )
            argmin = min(range(len(means)), key=lambda i: means[i])
            Y.append(means[argmin])
            ERROR.append(errors[argmin])

        X, Y, ERROR = map(np.array, [X, Y, ERROR])
        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

        plt.title("Random Walk")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"Minimum RMS Error")
        plot_path = os.path.join('plots', 'best-lr.png')
        plt.legend(loc='lower left')
        plt.savefig(plot_path)
