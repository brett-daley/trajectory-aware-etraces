import numpy as np


def epsilon_greedy_probabilities(q_values, epsilon):
    assert q_values.ndim == 1, "Q-values must be a 1-dimensional vector"
    n = len(q_values)
    probabilities = (epsilon / n) * np.ones_like(q_values)
    probabilities[np.argmax(q_values)] += (1.0 - epsilon)
    return probabilities
