import argparse
import os

import numpy as np
import yaml

import trajectory_aware_etraces.envs
from trajectory_aware_etraces.experiments.seeding import generate_seeds
from trajectory_aware_etraces.experiments import training
from trajectory_aware_etraces.experiments.training import run_control_sweep


CONFIG = None

DATA_DIR = None
ENV_ID = None
DISCOUNT = None
TIMESTEPS = None
BEHAVIOR_EPS = None
TARGET_EPS = None

ALGORITHMS = None
LAMBDAS = None
ALPHAS = None


def store_data(results, seeds, root_dir):
    for algo in ALGORITHMS:
        for lambd in LAMBDAS:
            for alpha in ALPHAS:
                algo_no_spaces = algo.replace(' ', '')
                exp_dir = os.path.join(root_dir, algo_no_spaces, f"lambd-{lambd}_alpha-{alpha}")
                os.makedirs(exp_dir, exist_ok=True)

                params = (lambd, alpha)
                Ys = results[(algo, *params)]
                for Y, s in zip(Ys, seeds):
                    path = os.path.join(exp_dir, f"seed-{s}.csv")
                    np.savetxt(path, Y, delimiter=',')


def main():
    run = lambda seeds: run_control_sweep(ENV_ID, BEHAVIOR_EPS, TARGET_EPS, DISCOUNT, ALGORITHMS, LAMBDAS, ALPHAS, seeds, TIMESTEPS)

    # Training data used to hyperparameter selection
    train_seeds = generate_seeds(meta_seed=CONFIG['train_seed'], n=CONFIG['train_trials'])
    results = run(train_seeds)
    store_data(results, train_seeds, root_dir=os.path.join(DATA_DIR, 'train'))

    # Test data used for plotting
    test_seeds = generate_seeds(meta_seed=CONFIG['test_seed'], n=CONFIG['test_trials'])
    results = run(test_seeds)
    store_data(results, test_seeds, root_dir=os.path.join(DATA_DIR, 'test'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = CONFIG = yaml.safe_load(f)
        DATA_DIR = config['data_dir']
        ENV_ID = config['env_id']
        DISCOUNT = config['discount']
        TIMESTEPS = config['timesteps']
        BEHAVIOR_EPS = config['behavior_eps']
        TARGET_EPS = config['target_eps']
        TRAIN_TRIALS = config['train_trials']
        TEST_TRIALS = config['test_trials']
        training.CONFIG = config

    with open('configs/.grid_search.yml', 'r') as f:
        config = yaml.safe_load(f)
        ALGORITHMS = config['algorithms']
        LAMBDAS = config['lambdas']
        ALPHAS = config['alphas']

    main()
