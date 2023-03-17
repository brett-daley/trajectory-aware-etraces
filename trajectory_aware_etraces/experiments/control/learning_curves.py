import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

from trajectory_aware_etraces.experiments.plot_formatting import preformat_plots, postformat_plots


DATA_DIR = None
ENV_ID = None
ENV_STR = None
DISCOUNT = None
SHORTEST_PATH = None
ALGORITHMS_TO_ALPHAS = None


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
    for algo, params in algo_specs.items():
        lambd, alpha = params
        Ys = load_experiment(root_dir, algo, lambd, alpha)

        Y = np.mean(Ys, axis=0)
        X = np.arange(len(Y))
        # 95% confidence interval
        ERROR = 1.96 * np.std(Ys, axis=0, ddof=1) / np.sqrt(len(Ys))

        AUCs = np.sum(Ys, axis=1)
        print(algo, params, np.mean(AUCs), 1.96 * np.std(AUCs, ddof=1) / np.sqrt(len(AUCs)))

        n = 50  # Downsampling -- set n=1 to keep all data
        X, Y, ERROR = X[::n], Y[::n], ERROR[::n]

        plt.plot(X, Y, label=algo)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

    plt.xlim([0, X[-1]])
    # plt.ylim([0, 0.6])

    # Plot horizontal dashed line for optimal discounted return
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
    with open('configs/.grid_search.yml', 'r') as f:
        config = yaml.safe_load(f)
        lambdas = config['lambdas']

    algorithms = list(ALGORITHMS_TO_ALPHAS.keys())

    for i, lambd in enumerate(lambdas):
        # algorithm -> (lambda, alpha)
        algo_specs = {algo: (lambd, ALGORITHMS_TO_ALPHAS[algo][i]) for algo in algorithms}
        str_lambda = str(int(lambd) if lambd == int(lambd) else lambd)
        plot_learning_curves(algo_specs, title=fr"{ENV_STR} ($\lambda={str_lambda}$)",
            plot_name=f"{ENV_ID}_lambda-{lambd}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        DATA_DIR = config['data_dir']
        ENV_ID = config['env_id']
        ENV_STR = config['env_str']
        DISCOUNT = config['discount']
        SHORTEST_PATH = config['shortest_path']
        ALGORITHMS_TO_ALPHAS = config['lambda_sweep_alphas']

    main()
