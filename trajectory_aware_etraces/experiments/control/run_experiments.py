import os

import numpy as np
import yaml

import trajectory_aware_etraces.envs
from trajectory_aware_etraces.experiments.seeding import generate_seeds
from trajectory_aware_etraces.experiments.training import run_control_sweep


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

data_dir = config['data_dir']
env_id = config['env_id']
discount = config['discount']

timesteps = config['timesteps']
behavior_eps = config['behavior_eps']
target_eps = config['target_eps']

train_trials = config['train_trials']
test_trials = config['test_trials']

search = config['grid_search']
algorithms = search['algorithms']
lambdas = search['lambdas']
alphas = search['alphas']


def store_data(results, seeds, root_dir):
    for algo in algorithms:
        for lambd in lambdas:
            for alpha in alphas:
                algo_no_spaces = algo.replace(' ', '')
                exp_dir = os.path.join(root_dir, algo_no_spaces, f"lambd-{lambd}_alpha-{alpha}")
                os.makedirs(exp_dir, exist_ok=True)

                params = (lambd, alpha)
                Ys = results[(algo, *params)]
                for Y, s in zip(Ys, seeds):
                    path = os.path.join(exp_dir, f"seed-{s}.csv")
                    np.savetxt(path, Y, delimiter=',')


def main():
    run = lambda seeds: run_control_sweep(env_id, behavior_eps, target_eps, discount, algorithms, lambdas, alphas, seeds, timesteps)

    # Training data used to hyperparameter selection
    train_seeds = generate_seeds(meta_seed=config['train_seed'], n=train_trials)
    results = run(train_seeds)
    store_data(results, train_seeds, root_dir=os.path.join(data_dir, 'train'))

    # Test data used for plotting
    test_seeds = generate_seeds(meta_seed=config['test_seed'], n=test_trials)
    results = run(test_seeds)
    store_data(results, test_seeds, root_dir=os.path.join(data_dir, 'test'))


if __name__ == '__main__':
    main()
