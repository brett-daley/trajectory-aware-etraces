import gym_classics
import numpy as np

from moretrace import grid_walk
from moretrace.experiments.training import run_sweep_Q


DISCOUNT = 1.0
LAMBDA_VALUES = np.linspace(0, 1, 11)
ALPHA_VALUES = np.linspace(0, 1, 21)[1:-1]  # Don't test alpha={0,1}
SEEDS = range(10)


# Rank hyperparameters based on this performance metric:
def performance(rms_errors):
    # Area under the curve
    return np.mean(rms_errors[:, 0:], axis=1)


def get_best_combo(results, estimator):
    best = float('inf')
    params = None

    for lambd in LAMBDA_VALUES:
        for lr in ALPHA_VALUES:
            key = (estimator, lambd, lr)
            rms_errors = results[key]

            perf = np.mean(performance(rms_errors))

            if perf < best:
                best = perf
                params = (lambd, lr)

    return params


def get_best_alphas(results, estimator):
    alpha_list = []

    for lambd in LAMBDA_VALUES:
        best = float('inf')
        best_alpha = None

        for lr in ALPHA_VALUES:
            key = (estimator, lambd, lr)
            rms_errors = results[key]

            perf = np.mean(performance(rms_errors))

            if perf < best:
                best = perf
                best_alpha = lr

        alpha_list.append(best_alpha)

    return alpha_list


def search_hyperparameters(env_id, behavior_policy, target_policy, return_estimators, n_episodes):
    print(f"--- {env_id}:")
    results = run_sweep_Q(env_id, behavior_policy, target_policy, DISCOUNT, return_estimators, LAMBDA_VALUES, ALPHA_VALUES, SEEDS, n_episodes)

    # Plot RMS vs Learning Rate
    for estimator in return_estimators:
        print(f"{estimator}:")
        params = get_best_combo(results, estimator)
        print("- Best (lambda, alpha):", params)

        alpha_list = get_best_alphas(results, estimator)
        print("- Sweep [best alphas]:", alpha_list)
        print()


if __name__ == '__main__':
    # Random Walk
    # Actions: left, right
    behavior_policy = np.array([0.5, 0.5])
    target_policy = np.array([0.1, 0.9])
    estimators = ['Retrace', 'Truncated IS', 'Recursive Retrace']
    search_hyperparameters("19Walk-v0", behavior_policy, target_policy, estimators, n_episodes=25)

    # Gridwalk
    # Actions: up, right, down, left
    behavior_policy = np.array([0.25, 0.25, 0.25, 0.25])
    target_policy = np.array([0.1, 0.7, 0.1, 0.1])
    estimators = ['Retrace', 'Truncated IS', 'Recursive Retrace', 'Moretrace']
    search_hyperparameters("GridWalk-v0", behavior_policy, target_policy, estimators, n_episodes=200)
