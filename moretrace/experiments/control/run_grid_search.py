from moretrace.experiments.grid_search import DISCOUNT, LAMBDA_VALUES, ALPHA_VALUES, SEEDS,\
    get_best_combo, get_best_alphas
from moretrace.experiments.training import run_control_sweep


def search_hyperparameters(env_id, behavior_eps, target_policy, return_estimators, n_timesteps):
    print(f"--- {env_id}:")
    results = run_control_sweep(env_id, behavior_eps, target_policy, DISCOUNT, return_estimators, LAMBDA_VALUES, ALPHA_VALUES, SEEDS, n_timesteps)

    # Plot RMS vs Learning Rate
    for estimator in return_estimators:
        print(f"{estimator}:")
        params = get_best_combo(results, estimator)
        print("- Best (lambda, alpha):", params)

        alpha_list = get_best_alphas(results, estimator)
        print("- Sweep [best alphas]:", alpha_list)
        print()


if __name__ == '__main__':
    # Gridwalk
    # Actions: up, right, down, left
    behavior_eps = 0.2
    target_policy = 0.1
    # estimators = ['Retrace', 'Truncated IS', 'Recursive Retrace', 'Moretrace']
    estimators = ['Retrace', 'Moretrace', 'Supertrace']
    search_hyperparameters("GridWalk-v0", behavior_eps, target_policy, estimators, n_timesteps=5_000)
