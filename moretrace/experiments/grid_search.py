import numpy as np


DISCOUNT = 0.9
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
            yields = results[key]

            perf = np.mean(performance(yields))

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

            perf = np.mean(performance(yields))

            if perf < best:
                best = perf
                best_alpha = lr

        alpha_list.append(best_alpha)

    return alpha_list
