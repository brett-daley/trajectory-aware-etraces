import os

import numpy as np

import trajectory_aware_etraces.envs
from trajectory_aware_etraces.experiments.seeding import generate_seeds
from trajectory_aware_etraces.experiments.training import run_control_sweep


DATA_DIR = 'data'

DISCOUNT = 0.9
LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
ESTIMATORS = ['Retrace', 'Truncated IS', 'Recursive Retrace', 'RBIS']

TRAIN_SEEDS = generate_seeds(meta_seed=0, n=1_000)
TEST_SEEDS = generate_seeds(meta_seed=1, n=1_000)


def store_data(results, seeds, root_dir):
    for est in ESTIMATORS:
        for lambd in LAMBDA_VALUES:
            for alpha in ALPHA_VALUES:
                est_no_spaces = est.replace(' ', '')
                exp_dir = os.path.join(root_dir, est_no_spaces, f"lambd-{lambd}_alpha-{alpha}")
                os.makedirs(exp_dir, exist_ok=True)

                params = (lambd, alpha)
                Ys = results[(est, *params)]
                for Y, s in zip(Ys, seeds):
                    path = os.path.join(exp_dir, f"seed-{s}.csv")
                    np.savetxt(path, Y, delimiter=',')


def main():
    # Gridwalk
    # Actions: up, right, down, left
    env_id = 'BifurcatedGridworld-v0'
    behavior_eps = 0.2
    target_eps = 0.1
    n_timesteps = 3_000

    results = run_control_sweep(env_id, behavior_eps, target_eps, DISCOUNT, ESTIMATORS, LAMBDA_VALUES, ALPHA_VALUES, TRAIN_SEEDS, n_timesteps)
    store_data(results, TRAIN_SEEDS, root_dir=os.path.join(DATA_DIR, 'train'))

    results = run_control_sweep(env_id, behavior_eps, target_eps, DISCOUNT, ESTIMATORS, LAMBDA_VALUES, ALPHA_VALUES, TEST_SEEDS, n_timesteps)
    store_data(results, TEST_SEEDS, root_dir=os.path.join(DATA_DIR, 'test'))


if __name__ == '__main__':
    main()
