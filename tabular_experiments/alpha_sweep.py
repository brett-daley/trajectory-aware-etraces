from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np

from training import run_sweep_V, run_sweep_Q
import walk19_no_op


if __name__ == '__main__':
    discount = 1.0
    return_estimators = ['Retrace', 'Moretrace']
    lambda_values = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
    learning_rates = np.exp(np.linspace(-5, 0, 51))
    seeds = range(10)

    # behavior_policy = np.array([0.5, 0.5])
    # target_policy = np.array([0.5, 0.5])
    # results = run_sweep_V('19Walk-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    behavior_policy = np.array([1/3, 1/3, 1/3])
    target_policy = np.array([1/6, 1/6, 2/3])
    results = run_sweep_V('19WalkNoOp-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    # Plot RMS vs Learning Rate
    for lambd in lambda_values:
        plt.figure()
        plt.xlim([0, 1])
        plt.xticks(np.linspace(0, 1, 10 + 1))
        plt.ylim([0.25, 0.55])

        for estimator in return_estimators:
            X, Y, ERROR = [], [], []
            for lr in learning_rates:
                key = (estimator, lambd, lr)
                mean, std = results[key]
                X.append(lr)
                Y.append(mean)
                ERROR.append(std)

            X, Y, ERROR = map(np.array, [X, Y, ERROR])
            plt.plot(X, Y, label=estimator)
            plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

        str_lambd = str(int(lambd)) if int(lambd) == lambd else str(lambd)
        plt.title(r"Random Walk" '\n' r"($\lambda=" + str_lambd + r"$)".format(str_lambd))
        plt.xlabel(r"$\alpha$")
        plt.ylabel("RMS Error")
        plot_path = os.path.join('plots', 'lambda-' + str(lambd) + '.png')
        plt.savefig(plot_path)
