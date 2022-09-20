import os

import matplotlib.pyplot as plt
import numpy as np

from moretrace.experiments.grid_search import DISCOUNT
from moretrace.experiments.plot_formatting import preformat_plots, postformat_plots
from moretrace.experiments.seeding import generate_seeds
from moretrace.experiments.training import run_control_sweep


def plot_learning_curves(env_id, behavior_policy, target_policy, algo_specs, n_timesteps, title):
    plt.figure()
    preformat_plots()

    seeds = generate_seeds(meta_seed=0, n=100)

    # Plot RMS vs Learning Rate
    for estimator, params in algo_specs.items():
        lambd, alpha = params
        results = run_control_sweep(env_id, behavior_policy, target_policy, DISCOUNT, [estimator], [lambd], [alpha], seeds, n_timesteps)

        # Get the episode lengths corresponding to the hyperparameters
        Ys = results[(estimator, *params)]
        Y = np.mean(Ys, axis=0)
        X = np.arange(len(Y))

        # 99% confidence interval
        ERROR = 2.576 * np.std(Ys, axis=0, ddof=1) / np.sqrt(len(seeds))

        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

    plt.xlim([0, len(X)])
    # plt.ylim([0, 100])

    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Episodes")

    postformat_plots()

    plot_path = os.path.join('plots', env_id)
    plt.savefig(plot_path  + '.png')
    plt.savefig(plot_path + '.pdf', format='pdf')


if __name__ == '__main__':
    # Gridwalk
    # Actions: up, right, down, left
    behavior_eps = 0.15
    target_eps = 0.05
    algo_specs = {
        # estimator -> (lambda, alpha)
        'Retrace': (0.5, 0.1),
        # 'Truncated IS': (0.85, 0.975),
        # 'Recursive Retrace': (0.85, 0.975),
        'Moretrace': (0.5, 0.1)
    }
    plot_learning_curves("GridWalk-v0", behavior_eps, target_eps, algo_specs, n_timesteps=5_000, title="Grid Walk")
