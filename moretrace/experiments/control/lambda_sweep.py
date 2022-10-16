import os

import gym_classics
import matplotlib.pyplot as plt
import numpy as np

from moretrace import grid_walk
from moretrace.experiments.plot_formatting import preformat_plots, postformat_plots
from moretrace.experiments.seeding import generate_seeds
from moretrace.experiments.training import run_control_sweep


DISCOUNT = 0.9
LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = generate_seeds(meta_seed=0, n=1_000)


def plot_lambda_sweep(env_id, behavior_eps, target_eps, algo_specs, n_timesteps, title):
    plt.figure()
    preformat_plots()

    # Plot RMS vs Lambda
    for estimator, best_alphas in algo_specs.items():
        assert len(best_alphas) == len(LAMBDA_VALUES)
        results = run_control_sweep(env_id, behavior_eps, target_eps, DISCOUNT, [estimator], LAMBDA_VALUES, best_alphas, SEEDS, n_timesteps)

        X, Y, ERROR = [], [], []
        for lambd, alpha in zip(LAMBDA_VALUES, best_alphas):
            key = (estimator, lambd, alpha)
            ys = results[key]
            y = np.mean(ys, axis=0)

            # 95% confidence interval
            confidence = 1.96 * np.std(ys, axis=0, ddof=1) / np.sqrt(len(SEEDS))

            X.append(lambd)
            Y.append(y[-1])
            ERROR.append(confidence[-1])

        X, Y, ERROR = map(np.array, [X, Y, ERROR])
        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

        plt.xlim([0, 1])
        plt.xticks(np.linspace(0.0, 1.0, 10 + 1))
        # plt.ylim([0.0, 1.0])

    plt.title(title)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Episodes")

    postformat_plots()

    plot_path = os.path.join('plots', 'lambda-' + env_id)
    plt.savefig(plot_path + '.png')
    plt.savefig(plot_path + '.pdf', format='pdf')


def main():
    # Gridwalk
    # Actions: up, right, down, left
    behavior_eps = 0.2
    target_eps = 0.1
    algo_specs = {
        # estimator -> (lambda, alpha)
        'Retrace':           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
        'Truncated IS':      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
        'Recursive Retrace': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3],
        'Moretrace':         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
    }
    plot_lambda_sweep("Bifurcation-v0", behavior_eps, target_eps, algo_specs, n_timesteps=2_500, title="Bifurcation")


if __name__ == '__main__':
    main()
