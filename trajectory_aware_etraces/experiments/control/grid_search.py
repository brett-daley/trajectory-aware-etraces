import os

import numpy as np

from trajectory_aware_etraces.experiments.control.learning_curves import load_experiment
from trajectory_aware_etraces.experiments.control.run_experiments import (DATA_DIR, LAMBDA_VALUES, ALPHA_VALUES, ESTIMATORS)


def main():
    # Grid search uses separate training seeds to avoid biased results
    root_dir = os.path.join(DATA_DIR, 'train')

    for est in ESTIMATORS:
        print(f"{est}: ---")
        for lambd in LAMBDA_VALUES:
            for alpha in ALPHA_VALUES:
                Ys = load_experiment(root_dir, est, lambd, alpha)
                AUCs = np.sum(Ys, axis=1)

                mean = np.mean(AUCs)
                # 95% confidence interval
                confidence = 1.96 * np.std(AUCs, ddof=1) / np.sqrt(len(AUCs))
                print((lambd, alpha), mean, confidence)
            print()
        print()


if __name__ == '__main__':
    main()
