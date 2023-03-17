import argparse
import os

import numpy as np
import yaml

from trajectory_aware_etraces.experiments.control.learning_curves import load_experiment


DATA_DIR = None
ALGORITHMS = None
LAMBDAS = None
ALPHAS = None


def main():
    # Grid search uses separate training seeds to avoid biased results
    root_dir = os.path.join(DATA_DIR, 'train')

    for algo in ALGORITHMS:
        print(f"{algo}: ---")
        for lambd in LAMBDAS:
            for alpha in ALPHAS:
                Ys = load_experiment(root_dir, algo, lambd, alpha)
                AUCs = np.sum(Ys, axis=1)

                mean = np.mean(AUCs)
                # 95% confidence interval
                confidence = 1.96 * np.std(AUCs, ddof=1) / np.sqrt(len(AUCs))
                print((lambd, alpha), mean, confidence)
            print()
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        DATA_DIR = config['data_dir']

    with open('configs/.grid_search.yml', 'r') as f:
        config = yaml.safe_load(f)
        ALGORITHMS = config['algorithms']
        LAMBDAS = config['lambdas']
        ALPHAS = config['alphas']

    main()
