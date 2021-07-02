import numpy as np


def get_trace_function(return_estimator, lambd):
    assert 0.0 <= lambd <= 1.0, "lambd must be in the range [0,1]"

    def importance_sampling_trace(action, pi, mu):
        return lambd * pi[action] / mu[action]

    def q_lambda_trace(action, pi, mu):
        return lambd

    def tree_backup_trace(action, pi, mu):
        return lambd * pi[action]

    def retrace_trace(action, pi, mu):
        return lambd * min(1.0, pi[action] / mu[action])

    return {
        'IS': importance_sampling_trace,
        'Qlambda': q_lambda_trace,
        'TB': tree_backup_trace,
        'Retrace': retrace_trace,
    }[return_estimator]


def epsilon_greedy_probabilities(q_values, epsilon):
    assert q_values.ndim == 1, "Q-values must be a 1-dimensional vector"
    n = len(q_values)
    probabilities = (epsilon / n) * np.ones_like(q_values)
    probabilities[np.argmax(q_values)] += (1.0 - epsilon)
    return probabilities
