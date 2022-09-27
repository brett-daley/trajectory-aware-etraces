import numpy as np


DISCOUNT = 0.9
LAMBDA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = range(96)


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
            yields = results[key]

            # TODO: We have this temporarily inverted to maximize AUC
            perf = -1 * np.mean(performance(yields))

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
            yields = results[key]

            # TODO: We have this temporarily inverted to maximize AUC
            perf = -1 * np.mean(performance(yields))

            if perf < best:
                best = perf
                best_alpha = lr

        alpha_list.append(best_alpha)

    return alpha_list
