import os

import numpy as np
import yaml

from trajectory_aware_etraces.experiments.control.learning_curves import load_experiment


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

data_dir = config['data_dir']

search = config['grid_search']
algorithms = search['algorithms']
lambdas = search['lambdas']
alphas = search['alphas']


def main():
    # Grid search uses separate training seeds to avoid biased results
    root_dir = os.path.join(data_dir, 'train')

    for algo in algorithms:
        print(f"{algo}: ---")
        for lambd in lambdas:
            for alpha in alphas:
                Ys = load_experiment(root_dir, algo, lambd, alpha)
                AUCs = np.sum(Ys, axis=1)

                mean = np.mean(AUCs)
                # 95% confidence interval
                confidence = 1.96 * np.std(AUCs, ddof=1) / np.sqrt(len(AUCs))
                print((lambd, alpha), mean, confidence)
            print()
        print()


if __name__ == '__main__':
    main()
