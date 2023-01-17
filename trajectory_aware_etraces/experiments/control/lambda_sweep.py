import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

from trajectory_aware_etraces.experiments.control.learning_curves import load_experiment
from trajectory_aware_etraces.experiments.plot_formatting import preformat_plots, postformat_plots


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

data_dir = config['data_dir']
plot_dir = config['plot_dir']
env_str = config['env_str']
lambdas = config['grid_search']['lambdas']
algorithms_to_alphas = config['lambda_sweep_alphas']


def main():
    plt.figure()
    preformat_plots()

    root_dir = os.path.join(data_dir, 'test')

    # Plot RMS vs Lambda
    for algorithm, alphas in algorithms_to_alphas.items():
        assert len(alphas) == len(lambdas)

        X, Y, ERROR = [], [], []
        for lambd, alpha in zip(lambdas, alphas):
            Ys = load_experiment(root_dir, algorithm, lambd, alpha)

            AUCs = np.sum(Ys, axis=1)
            mean = np.mean(AUCs)
            # 95% confidence interval
            confidence = 1.96 * np.std(AUCs, ddof=1) / np.sqrt(len(AUCs))

            X.append(lambd)
            Y.append(mean)
            ERROR.append(confidence)

        X, Y, ERROR = map(np.array, [X, Y, ERROR])
        plt.plot(X, Y, label=algorithm)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

        plt.xlim([0, 1])
        plt.xticks(np.linspace(0.0, 1.0, 10 + 1))
        plt.ylim([1160, 1300])

    plt.title(env_str)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Area Under the Curve")

    postformat_plots(aspect=2.0, legend=False)

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_name = f"{env_str.replace(' ', '_')}_lambda-sweep"
    plot_path = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_path + '.png')
    plt.savefig(plot_path + '.pdf', format='pdf')


if __name__ == '__main__':
    main()
