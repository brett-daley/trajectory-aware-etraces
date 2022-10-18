import os

import matplotlib.pyplot as plt
import numpy as np

from moretrace.experiments.plot_formatting import preformat_plots, postformat_plots
from moretrace.experiments.seeding import generate_seeds
from moretrace.experiments.training import run_control_sweep


DISCOUNT = 0.9
SEEDS = generate_seeds(meta_seed=0, n=96)


def plot_learning_curves(env_id, behavior_eps, target_eps, algo_specs, n_timesteps, title, plot_name=None):
    plt.figure()
    preformat_plots()

    # Plot RMS vs Learning Rate
    for estimator, params in algo_specs.items():
        lambd, alpha = params
        results = run_control_sweep(env_id, behavior_eps, target_eps, DISCOUNT, [estimator], [lambd], [alpha], SEEDS, n_timesteps)

        # Get the discounted returns corresponding to the hyperparameters
        Ys = results[(estimator, *params)]
        Y = np.mean(Ys, axis=0)
        X = np.arange(len(Y))

        # 95% confidence interval
        ERROR = 1.96 * np.std(Ys, axis=0, ddof=1) / np.sqrt(len(SEEDS))

        AUCs = np.sum(Ys, axis=1)
        print(estimator, params, np.mean(AUCs),  1.96 * np.std(AUCs, ddof=1) / np.sqrt(len(SEEDS)))

        n = 50  # Downsampling -- set n=1 to keep all data
        X, Y, ERROR = X[::n], Y[::n], ERROR[::n]

        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

    plt.xlim([0, X[-1]])
    # plt.ylim([0, 100])

    SHORTEST_PATH = 7  # For Bifurcation environment
    plt.plot(X, pow(DISCOUNT, SHORTEST_PATH - 1) * np.ones_like(X), linestyle='--', color='black')

    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Discounted Return")

    postformat_plots()

    if plot_name is None:
        plot_name = env_id
    plot_path = os.path.join('plots', plot_name)
    plt.savefig(plot_path + '.png')
    plt.savefig(plot_path + '.pdf', format='pdf')


def main():
    # Gridwalk
    # Actions: up, right, down, left
    behavior_eps = 0.2
    target_eps = 0.1
    for lambd in [0.4]:
        algo_specs = {
            # estimator -> (lambda, alpha)
            'Retrace': (lambd, 0.5),
            # 'Truncated IS': (lambd, 0.5),
            # 'Recursive Retrace': (lambd, 0.7),
            'Moretrace': (lambd, 0.5),
            # 'Moretrace2': (lambd, 0.5)
        }
        plot_learning_curves("Bifurcation-v0", behavior_eps, target_eps, algo_specs, n_timesteps=2_500, title="Bifurcation", plot_name=f"Bifurcation-v0_lambd-{lambd}")


if __name__ == '__main__':
    main()
