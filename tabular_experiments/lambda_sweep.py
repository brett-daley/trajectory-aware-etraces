from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np

from training import run_sweep_V, run_sweep_Q
import walk19_no_op


if __name__ == '__main__':
    discount = 1.0
    return_estimators = ['Retrace', 'Moretrace']
    lambda_values = sorted(
        [round(0.05 * i, 2) for i in range(21)] \
        + [0.96, 0.97, 0.98, 0.99])
    learning_rates = np.exp(np.linspace(-5, 0, 51))
    seeds = range(10)

    # behavior_policy = np.array([0.5, 0.5])
    # target_policy = np.array([0.5, 0.5])
    # results = run_sweep_V('19Walk-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    behavior_policy = np.array([1/3, 1/3, 1/3])
    target_policy = np.array([1/6, 1/6, 2/3])
    results = run_sweep_V('19WalkNoOp-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    # Plot Best RMS vs Lambda
    plt.figure()
    plt.xlim([0, 1])
    plt.xticks(np.linspace(0.0, 1.0, 10 + 1))
    plt.ylim([0.25, 0.55])

    for estimator in return_estimators:
        X, Y, ERROR = [], [], []
        for lambd in lambda_values:
            X.append(lambd)
            means, stds = zip(*(
                results[(estimator, lambd, lr)] for lr in learning_rates)
            )
            argmin = min(range(len(means)), key=lambda i: means[i])
            Y.append(means[argmin])
            ERROR.append(stds[argmin])

        X, Y, ERROR = map(np.array, [X, Y, ERROR])
        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

        plt.title(r"Random Walk" '\n' r"(best RMS for each $\lambda$)")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"Minimum RMS Error")
        plot_path = os.path.join('plots', 'best-lr.png')
        plt.savefig(plot_path)
