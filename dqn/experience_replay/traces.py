import numpy as np


def get_trace_function(return_estimator):
    if '-' not in return_estimator:
        return_estimator += '-'
    name, lambd = return_estimator.split('-')

    if name == 'IS':
        assert lambd == '', "IS should not have a lambda value"
    else:
        assert lambd != '', "must specify a lambda value for " + name
        lambd = float(lambd)

    def importance_sampling_trace(action, pi, mu):
        return pi[action] / mu[action]

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
    }[name]


def epsilon_greedy_probabilities(q_values, epsilon):
    assert q_values.ndim == 1, "Q-values must be a 1-dimensional vector"
    n = len(q_values)
    probabilities = (epsilon / n) * np.ones_like(q_values)
    probabilities[np.argmax(q_values)] += (1.0 - epsilon)
    return probabilities
