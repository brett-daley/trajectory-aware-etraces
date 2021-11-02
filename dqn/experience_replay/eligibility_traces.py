from abc import ABC, abstractmethod

import numpy as np


class OfflineEligibilityTrace(ABC):
    def __init__(self, discount, lambd, maxlen):
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= lambd <= 1.0
        self._discount = discount
        self._lambd = lambd

        assert maxlen > 0
        self._maxlen = maxlen

    def __call__(self, td_errors, target_probs, behavior_probs, dones):
        assert len(td_errors) == len(target_probs) == len(behavior_probs) == len(dones)

        eligibility = np.zeros(self._maxlen, dtype=np.float64)
        updates = np.zeros(self._maxlen, dtype=np.float64)

        current_episode_start = 0
        for t in range(len(td_errors)):
            assert t < self._maxlen

            sl = slice(current_episode_start, t)

            trace = self._trace_coefficient(target_probs[t], behavior_probs[t])
            eligibility[sl] *= self._discount * trace
            eligibility[t] += 1.0

            sl = slice(current_episode_start, t + 1)

            updates[sl] += td_errors[t] * eligibility[sl]

            if dones[t]:
                current_episode_start = t

        return updates

    @abstractmethod
    def _trace_coefficient(self, target_prob, behavior_prob):
        raise NotImplementedError


class IS(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return self._lambd * target_prob / behavior_prob


class Qlambda(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        return self._lambd


class TB(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        return self._lambd * target_prob


class Retrace(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return self._lambd * min(1.0, target_prob / behavior_prob)


class Moretrace(OfflineEligibilityTrace):
    def _reset(self):
        super()._reset()
        self._eligibility = None
        self._trace_products = np.zeros(self._maxlen, dtype=np.float64)
        self._discounting = np.zeros(self._maxlen, dtype=np.float64)
        self._lambda_products = np.zeros(self._maxlen, dtype=np.float64)

    def update(self, td_error, target_prob, behavior_prob, done):
        assert self._t < self._maxlen

        sl = slice(self._current_episode_start, self._t)

        self._trace_products[sl] *= self._trace_coefficient(target_prob, behavior_prob)
        self._discounting[sl] *= self._discount
        self._lambda_products[sl] *= self._lambd

        self._trace_products[self._t] += 1.0
        self._discounting[self._t] += 1.0
        self._lambda_products[self._t] += 1.0

        sl = slice(self._current_episode_start, self._t + 1)

        self._updates[sl] += td_error * self._discounting[sl] \
            * np.minimum(self._lambda_products[sl], self._trace_products[sl])

        self._t += 1

        if done:
            self._current_episode_start = self._t

    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return target_prob / behavior_prob


def epsilon_greedy_probabilities(q_values, epsilon):
    assert q_values.ndim in {1, 2}, "Q-values must be a 1- or 2-dimensional vector"
    n = q_values.shape[-1]
    probabilities = (epsilon / n) * np.ones_like(q_values)
    probabilities[..., np.argmax(q_values)] += (1.0 - epsilon)
    return probabilities
