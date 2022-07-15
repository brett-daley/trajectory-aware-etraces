import os

import gym_classics
import matplotlib.pyplot as plt
import numpy as np

from alpha_sweep import preformat_plots, postformat_plots
from training import run_sweep_Q


if __name__ == '__main__':
    discount = 1.0
    return_estimators = ['Retrace', 'Moretrace', 'Moretrace2', 'Recursive Retrace', 'Truncated IS']
    lambda_values = np.linspace(0, 1, 21)
    learning_rates = np.linspace(0, 1, 21)[1:-1]  # Don't test alpha = {0,1}
    seeds = range(100)

    behavior_policy = np.array([0.5, 0.5])
    target_policy = np.array([0.1, 0.9])
    results = run_sweep_Q('19Walk-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    preformat_plots()

    # Rank hyperparameters based on this performance metric:
    def performance(rms_errors):
        # AUC over last 10% of training
        return np.mean(rms_errors[:, -5:], axis=1)

    # Plot RMS vs Learning Rate
    for estimator in return_estimators:
        best = float('inf')
        params = None

        for lambd in lambda_values:
            for lr in learning_rates:
                key = (estimator, lambd, lr)
                rms_errors = results[key]

                perf = np.mean(performance(rms_errors))

                if perf < best:
                    best = perf
                    params = (lambd, lr)

        # Get the errors corresponding to the best hyperparameters we found
        rms_errors = results[(estimator, *params)]
        print(estimator, f"lambda={params[0]}", f"lr={params[1]}")
        Y = np.mean(rms_errors, axis=0)
        X = np.arange(len(Y))
        # 99% confidence interval
        ERROR = 2.576 * np.std(rms_errors, axis=0, ddof=1) / np.sqrt(len(seeds))

        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

    plt.xlim([0, len(Y) - 1])
    plt.ylim([0.0, 1.0])

    plt.title(r"Random Walk")
    plt.xlabel("Episodes")
    plt.ylabel("RMS Error")

    postformat_plots()

    plot_path = os.path.join('plots', 'hp_sweep_learning_curves')
    plt.savefig(plot_path  + '.png')
    plt.savefig(plot_path + '.pdf', format='pdf')
