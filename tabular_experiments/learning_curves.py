import os

import gym_classics
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

from grid_search import DISCOUNT
from training import run_sweep_Q


def preformat_plots():
    # Set font size
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rcParams.update({'font.size': 22})

    # Set color cycle to
    # Midnight Blue, Belize Hole, Nephritis, Pomegranate
    # (see https://flatuicolors.com/palette/defo for all colors)
    color_cycler = cycler('color', ['#2c3e50', '#2980b9', '#27ae60', '#c0392b'])
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

    plt.tight_layout(pad=0.02)


def generate_seeds(meta_seed, n):
    np_random = np.random.RandomState(meta_seed)  # Set seed for reproducibility
    # Randomize the seeds here to avoid any possible bias from reusing seeds from the hyperparameter search
    return [np_random.randint(2**31) for _ in range(100)]


def plot_learning_curves(env_id, behavior_policy, target_policy, algo_specs, n_episodes, title):
    plt.figure()
    preformat_plots()

    seeds = generate_seeds(meta_seed=0, n=100)

    # Plot RMS vs Learning Rate
    for estimator, params in algo_specs.items():
        lambd, alpha = params
        results = run_sweep_Q(env_id, behavior_policy, target_policy, DISCOUNT, [estimator], [lambd], [alpha], seeds, n_episodes)

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

    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("RMS Error")

    postformat_plots()

    plot_path = os.path.join('plots', env_id)
    plt.savefig(plot_path  + '.png')
    plt.savefig(plot_path + '.pdf', format='pdf')


if __name__ == '__main__':
    # Random Walk
    # Actions: left, right
    behavior_policy = np.array([0.5, 0.5])
    target_policy = np.array([0.1, 0.9])
    algo_specs = {
        # estimator -> (lambda, alpha)
        'Retrace': (0.95, 0.975),
        'Truncated IS': (0.95, 0.75),
        'Recursive Retrace': (0.95, 0.95)
    }
    plot_learning_curves("19Walk-v0", behavior_policy, target_policy, algo_specs, n_episodes=25, title="Linear Walk")

    # Gridwalk
    # Actions: up, right, down, left
    behavior_policy = np.array([0.25, 0.25, 0.25, 0.25])
    target_policy = np.array([0.1, 0.7, 0.1, 0.1])
    algo_specs = {
        # estimator -> (lambda, alpha)
        'Retrace': (0.95, 0.975),
        'Truncated IS': (0.85, 0.975),
        'Recursive Retrace': (0.85, 0.975),
        'Moretrace': (0.7, 0.975)
    }
    plot_learning_curves("GridWalk-v0", behavior_policy, target_policy, algo_specs, n_episodes=200, title="Grid Walk")
