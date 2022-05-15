import os

import gym_classics
import matplotlib.pyplot as plt
import numpy as np

from alpha_sweep import preformat_plots, postformat_plots
from training import run_sweep_V, run_sweep_Q


if __name__ == '__main__':
    discount = 1.0
    return_estimators = ['Retrace', 'Moretrace']
    lambda_values = np.linspace(0, 1, 51)
    learning_rates = [0.2, 0.4, 0.6, 0.8, 1.0]
    seeds = range(50)

    # behavior_policy = np.array([0.5, 0.5])
    # target_policy = np.array([0.5, 0.5])
    # results = run_sweep_V('19Walk-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    behavior_policy = np.array([0.5, 0.5])
    target_policy = np.array([0.1, 0.9])
    results = run_sweep_Q('19Walk-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    preformat_plots()

    # Plot RMS vs Lambda
    for lr in learning_rates:
        plt.figure()

        for estimator in return_estimators:
            X, Y, ERROR = [], [], []
            for lambd in lambda_values:
                key = (estimator, lambd, lr)
                final_error = results[key][:, -1]
                mean = np.mean(final_error)
                # 99% confidence interval
                confidence = 2.576 * np.std(final_error, ddof=1) / np.sqrt(len(seeds))

                # Hide error bars if we exceed the y-axis
                if mean >= 1.0:
                    confidence = 0.0

                X.append(lambd)
                Y.append(mean)
                ERROR.append(confidence)

            X, Y, ERROR = map(np.array, [X, Y, ERROR])
            plt.plot(X, Y, label=estimator)
            plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

        plt.xlim([0, 1])
        plt.xticks(np.linspace(0.0, 1.0, 10 + 1))
        plt.ylim([0.0, 1.0])

        str_lr = str(int(lr)) if int(lr) == lr else str(lr)
        plt.title(r"Random Walk ($\alpha=" + str_lr + r"$)")
        plt.xlabel(r"$\lambda$")
        plt.ylabel("RMS Error")

        postformat_plots()

        plot_path = os.path.join('plots', 'alpha-' + str(lr))
        plt.savefig(plot_path  + '.png')
        plt.savefig(plot_path + '.pdf', format='pdf')
