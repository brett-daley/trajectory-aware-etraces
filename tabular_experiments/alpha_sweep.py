import os

import gym_classics
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

from training import run_sweep_V, run_sweep_Q


def preformat_plots():
    # Set font size
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rcParams.update({'font.size': 22})

    # Set color cycle to
    # Belize Hole, Pomegranate, Nephritis, Midnight Blue
    # (see https://flatuicolors.com/palette/defo for all colors)
    color_cycler = cycler('color', ['#2980b9', '#c0392b', '#27ae60', '#2c3e50'])
    plt.rcParams.update({'axes.prop_cycle': color_cycler})


def postformat_plots():
    # Make axes square
    ax = plt.gca()
    ax.set_aspect(1.0 / ax.get_data_ratio())
    plt.gcf().set_size_inches(6.4, 6.4)

    plt.legend(loc='best', frameon=False, framealpha=0.0, fontsize=16)

    # Turn off top/right borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout(pad=0)


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
    target_policy = np.array([0.1, 0.9])
    results = run_sweep_Q('19Walk-v0', behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds)

    preformat_plots()

    # Plot RMS vs Learning Rate
    for lambd in lambda_values:
        plt.figure()

        for estimator in return_estimators:
            X, Y, ERROR = [], [], []
            for lr in learning_rates:
                key = (estimator, lambd, lr)
                final_error = results[key][:, -1]
                mean = np.mean(final_error)
                # 99% confidence interval
                confidence = 2.576 * np.std(final_error, ddof=1) / np.sqrt(len(seeds))

                # Hide error bars if we exceed the y-axis
                if mean >= 1.0:
                    confidence = 0.0

                X.append(lr)
                Y.append(mean)
                ERROR.append(confidence)

            X, Y, ERROR = map(np.array, [X, Y, ERROR])
            plt.plot(X, Y, label=estimator)
            plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

        plt.xlim([0, 1])
        plt.xticks(np.linspace(0, 1, 10 + 1))
        plt.ylim([0.0, 1.0])

        str_lambd = str(int(lambd)) if int(lambd) == lambd else str(lambd)
        plt.title(r"Random Walk ($\lambda=" + str_lambd + r"$)")
        plt.xlabel(r"$\alpha$")
        plt.ylabel("RMS Error")

        postformat_plots()

        plot_path = os.path.join('plots', 'lambda-' + str(lambd))
        plt.savefig(plot_path  + '.png')
        plt.savefig(plot_path + '.pdf', format='pdf')
