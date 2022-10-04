import numpy as np

from moretrace.experiments.training import run_control_sweep


DISCOUNT = 0.9
LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = range(480)


def grid_search(env_id, behavior_policy, target_policy, estimators, n_timesteps):
    results = run_control_sweep(env_id, behavior_eps, target_policy, DISCOUNT, estimators, LAMBDA_VALUES, ALPHA_VALUES, SEEDS, n_timesteps)

    for est in estimators:
        for lambd in LAMBDA_VALUES:
            for alpha in ALPHA_VALUES:
                params = (lambd, alpha)
                # Get the episode lengths corresponding to the hyperparameters
                Ys = results[(est, *params)]
                Y = np.mean(Ys, axis=0)
                # 99% confidence interval
                ERROR = 2.576 * np.std(Ys, axis=0, ddof=1) / np.sqrt(len(SEEDS))

                print(est, params, Y[-1], ERROR[-1])


if __name__ == '__main__':
    # Gridwalk
    # Actions: up, right, down, left
    behavior_eps = 0.2
    target_eps = 0.1
    estimators = ['Retrace', 'Truncated IS', 'Recursive Retrace', 'Newtrace']
    grid_search("GridWalk-v0", behavior_eps, target_eps, estimators, n_timesteps=2_500)
