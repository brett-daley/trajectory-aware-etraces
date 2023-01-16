import os

import matplotlib.pyplot as plt
import numpy as np

from trajectory_aware_etraces.experiments.control.run_experiments import (DATA_DIR, DISCOUNT,
    LAMBDA_VALUES, ALPHA_VALUES, ESTIMATORS)
from trajectory_aware_etraces.experiments.plot_formatting import preformat_plots, postformat_plots


def load_experiment(root_dir, estimator, lambd, alpha):
    est_no_spaces = estimator.replace(' ', '')
    exp_dir = os.path.join(root_dir, est_no_spaces, f"lambd-{lambd}_alpha-{alpha}")
    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir)]
    return np.array([np.loadtxt(f) for f in files])


def plot_learning_curves(algo_specs, title, plot_name):
    plt.figure()
    preformat_plots()

    root_dir = os.path.join(DATA_DIR, 'test')

    # Plot RMS vs Learning Rate
    for estimator, params in algo_specs.items():
        lambd, alpha = params
        Ys = load_experiment(root_dir, estimator, lambd, alpha)

        Y = np.mean(Ys, axis=0)
        X = np.arange(len(Y))
        # 95% confidence interval
        ERROR = 1.96 * np.std(Ys, axis=0, ddof=1) / np.sqrt(len(Ys))

        AUCs = np.sum(Ys, axis=1)
        print(estimator, params, np.mean(AUCs), 1.96 * np.std(AUCs, ddof=1) / np.sqrt(len(AUCs)))

        n = 50  # Downsampling -- set n=1 to keep all data
        X, Y, ERROR = X[::n], Y[::n], ERROR[::n]

        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

    plt.xlim([0, X[-1]])
    plt.ylim([0, 0.6])

    SHORTEST_PATH = 7  # For Bifurcation environment only
    plt.plot(X, pow(DISCOUNT, SHORTEST_PATH - 1) * np.ones_like(X), linestyle='--', color='black')

    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Discounted Return")

    postformat_plots()

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_path + '.png')
    plt.savefig(plot_path + '.pdf', format='pdf')


def main():
    # Import here to avoid circular import
    from trajectory_aware_etraces.experiments.control.lambda_sweep import ALGO_SPECS
    estimators = list(ALGO_SPECS.keys())

    for i, lambd in enumerate(LAMBDA_VALUES):
        # estimator -> (lambda, alpha)
        algo_specs = {est: (lambd, ALGO_SPECS[est][i]) for est in estimators}
        str_lambda = str(int(lambd) if lambd == int(lambd) else lambd)
        plot_learning_curves(algo_specs, title=fr"Bifurcated Gridworld ($\lambda={str_lambda}$)", plot_name=f"BifurcatedGridworld_lambda-{lambd}")
        print()


if __name__ == '__main__':
    main()