import os

import matplotlib.pyplot as plt
import numpy as np

from moretrace.experiments.control.learning_curves import load_experiment
from moretrace.experiments.control.run_experiments import DATA_DIR, LAMBDA_VALUES
from moretrace.experiments.plot_formatting import preformat_plots, postformat_plots


ALGO_SPECS = {
    # estimator -> [alphas]
    'Retrace':            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 0.7, 0.5],
    'Truncated IS':       [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 0.5, 0.5, 0.5, 0.3],
    'Recursive Retrace':  [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 0.5, 0.5],
    'RBIS':               [0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 0.5, 0.5, 0.5, 0.5, 0.3],
}


def plot_lambda_sweep(algo_specs, title, plot_name):
    plt.figure()
    preformat_plots()

    root_dir = os.path.join(DATA_DIR, 'test')

    # Plot RMS vs Lambda
    for estimator, best_alphas in algo_specs.items():
        assert len(best_alphas) == len(LAMBDA_VALUES)

        X, Y, ERROR = [], [], []
        for lambd, alpha in zip(LAMBDA_VALUES, best_alphas):
            Ys = load_experiment(root_dir, estimator, lambd, alpha)

            AUCs = np.sum(Ys, axis=1)
            mean = np.mean(AUCs)
            # 95% confidence interval
            confidence = 1.96 * np.std(AUCs, ddof=1) / np.sqrt(len(AUCs))

            X.append(lambd)
            Y.append(mean)
            ERROR.append(confidence)

        X, Y, ERROR = map(np.array, [X, Y, ERROR])
        plt.plot(X, Y, label=estimator)
        plt.fill_between(X, (Y - ERROR), (Y + ERROR), alpha=0.25, linewidth=0)

        plt.xlim([0, 1])
        plt.xticks(np.linspace(0.0, 1.0, 10 + 1))
        plt.ylim([1160, 1300])

    plt.title(title)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Area Under the Curve")

    postformat_plots(aspect=2.0)

    plot_path = os.path.join('plots', plot_name)
    plt.savefig(plot_path + '.png')
    plt.savefig(plot_path + '.pdf', format='pdf')


def main():
    plot_lambda_sweep(ALGO_SPECS, title="Bifurcated Gridworld", plot_name="BifurcatedGridworld_lambda-sweep")


if __name__ == '__main__':
    main()
