import os

import gym_classics
import matplotlib.pyplot as plt
import numpy as np

from moretrace import grid_walk
from moretrace.experiments.grid_search import DISCOUNT, LAMBDA_VALUES, performance
from moretrace.experiments.plot_formatting import preformat_plots, postformat_plots
from moretrace.experiments.seeding import generate_seeds
from moretrace.experiments.training import run_control_sweep


def plot_lambda_sweep(env_id, behavior_policy, target_policy, algo_specs, n_timesteps, title):
    plt.figure()
    preformat_plots()

    seeds = generate_seeds(meta_seed=0, n=100)

    # Plot RMS vs Lambda
    for estimator, best_alphas in algo_specs.items():
        assert len(best_alphas) == len(LAMBDA_VALUES)

        X, Y, ERROR = [], [], []
        for lambd, alpha in zip(LAMBDA_VALUES, best_alphas):
            results = run_control_sweep(env_id, behavior_policy, target_policy, DISCOUNT, [estimator], [lambd], [alpha], seeds, n_timesteps)
            key = (estimator, lambd, alpha)
            rms_errors = results[key]
            episode_metrics = performance(rms_errors)
            mean = np.mean(episode_metrics)
            # 99% confidence interval
            confidence = 2.576 * np.std(episode_metrics, ddof=1) / np.sqrt(len(seeds))

            X.append(lambd)
            Y.append(mean)
            ERROR.append(confidence)

        X, Y, ERROR = map(np.array, [X, Y, ERROR])
        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

        plt.xlim([0, 1])
        plt.xticks(np.linspace(0.0, 1.0, 10 + 1))
        # plt.ylim([0.0, 1.0])

    plt.title(title)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("AUC")

    postformat_plots()

    plot_path = os.path.join('plots', 'lambda-' + env_id)
    plt.savefig(plot_path + '.png')
    plt.savefig(plot_path + '.pdf', format='pdf')


if __name__ == '__main__':
    # Gridwalk
    # Actions: up, right, down, left
    behavior_eps = 0.2
    target_eps = 0.1
    algo_specs = {
        # estimator -> (lambda, alpha)
        'Retrace': [0.1] * 6,
        # 'Truncated IS': [0.1] * 6,
        # 'Recursive Retrace': [0.1] * 6,
        # 'Moretrace': [0.1] * 6,
        'Supertrace': [0.1] * 6
    }
    plot_lambda_sweep("GridWalk-v0", behavior_eps, target_eps, algo_specs, n_timesteps=5_000, title="Grid Walk")
