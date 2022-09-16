import numpy as np

from moretrace.experiments.grid_search import DISCOUNT, LAMBDA_VALUES, ALPHA_VALUES, SEEDS,\
    get_best_combo, get_best_alphas
from moretrace.experiments.training import run_prediction_sweep


def search_hyperparameters(env_id, behavior_policy, target_policy, return_estimators, n_episodes):
    print(f"--- {env_id}:")
    results = run_prediction_sweep(env_id, behavior_policy, target_policy, DISCOUNT, return_estimators, LAMBDA_VALUES, ALPHA_VALUES, SEEDS, n_episodes)

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
    behavior_eps = 1.0
    target_policy = np.array([0.1, 0.9])
    estimators = ['Retrace', 'Truncated IS', 'Recursive Retrace']
    search_hyperparameters("19Walk-v0", behavior_eps, target_policy, estimators, n_episodes=25)

    # Gridwalk
    # Actions: up, right, down, left
    behavior_eps = 1.0
    target_policy = np.array([0.1, 0.7, 0.1, 0.1])
    estimators = ['Retrace', 'Truncated IS', 'Recursive Retrace', 'Moretrace']
    search_hyperparameters("GridWalk-v0", behavior_eps, target_policy, estimators, n_episodes=200)
