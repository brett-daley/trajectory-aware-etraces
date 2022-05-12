from collections import defaultdict
import os

import gym_classics
import matplotlib.pyplot as plt
import numpy as np

from training import run_sweep_V, run_sweep_Q


if __name__ == '__main__':
    discount = 1.0
    return_estimators = ['Retrace', 'Moretrace']
    lambda_values = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    learning_rates = np.linspace(0, 1, 51)
    seeds = range(50)

    # behavior_policy = np.array([0.5, 0.5])
    # target_policy = np.array([0.5, 0.5])
    # results = run_sweep_V('19Walk-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    behavior_policy = np.array([0.5, 0.5])
    target_policy = np.array([0.4, 0.6])
    results = run_sweep_Q('19Walk-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    # Set font size
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rcParams.update({'font.size': 22})

    # Plot RMS vs Learning Rate
    for lambd in lambda_values:
        plt.figure()
        plt.xlim([0, 1])
        plt.xticks(np.linspace(0, 1, 10 + 1))
        plt.ylim([0.0, 1.0])

        for estimator in return_estimators:
            X, Y, ERROR = [], [], []
            for lr in learning_rates:
                key = (estimator, lambd, lr)
                mean, error = results[key]
                X.append(lr)
                Y.append(mean)
                ERROR.append(error)

            X, Y, ERROR = map(np.array, [X, Y, ERROR])
            plt.plot(X, Y, label=estimator)
            plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

        # Make axes square
        ax = plt.gca()
        ax.set_aspect(1.0 / ax.get_data_ratio())
        plt.gcf().set_size_inches(6.4, 6.4)

        str_lambd = str(int(lambd)) if int(lambd) == lambd else str(lambd)
        plt.title(r"Random Walk ($\lambda=" + str_lambd + r"$)")
        plt.xlabel(r"$\alpha$")
        plt.ylabel("RMS Error")
        plot_path = os.path.join('plots', 'lambda-' + str(lambd) + '.png')

        plt.legend(loc='best', framealpha=1.0, fontsize=16)

        plt.tight_layout(pad=0.05)
        plot_path = os.path.join('plots', 'lambda-' + str(lambd))
        plt.savefig(plot_path  + '.png')
        # plt.savefig(plot_path + 'pdf', format='pdf')
