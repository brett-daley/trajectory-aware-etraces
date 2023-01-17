import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

from trajectory_aware_etraces.experiments.plot_formatting import preformat_plots, postformat_plots


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

data_dir = config['data_dir']
env_str = config['env_str']
discount = config['discount']
shortest_path = config['shortest_path']

search = config['grid_search']
lambdas = search['lambdas']
alphas = search['alphas']

algorithms_to_alphas = config['lambda_sweep_alphas']


def load_experiment(root_dir, estimator, lambd, alpha):
    est_no_spaces = estimator.replace(' ', '')
    exp_dir = os.path.join(root_dir, est_no_spaces, f"lambd-{lambd}_alpha-{alpha}")
    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir)]
    return np.array([np.loadtxt(f) for f in files])


def plot_learning_curves(algo_specs, title, plot_name):
    plt.figure()
    preformat_plots()

    root_dir = os.path.join(data_dir, 'test')

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
    plt.ylim([0, 0.6])

    # Plot horizontal dashed line for optimal discounted return
    plt.plot(X, pow(discount, shortest_path - 1) * np.ones_like(X), linestyle='--', color='black')

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
    algorithms = list(algorithms_to_alphas.keys())

    for i, lambd in enumerate(lambdas):
        # algorithm -> (lambda, alpha)
        algo_specs = {algo: (lambd, algorithms_to_alphas[algo][i]) for algo in algorithms}
        str_lambda = str(int(lambd) if lambd == int(lambd) else lambd)
        plot_learning_curves(algo_specs, title=fr"{env_str} ($\lambda={str_lambda}$)",
            plot_name=f"{env_str.replace(' ', '_')}_lambda-{lambd}")
        print()


if __name__ == '__main__':
    main()
