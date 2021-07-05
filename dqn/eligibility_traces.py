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
        self._reset()

    def _reset(self):
        self._trace_products = np.zeros(self._maxlen, dtype=np.float64)
        self._discounting = np.zeros(self._maxlen, dtype=np.float64)
        self._updates = np.zeros(self._maxlen, dtype=np.float64)

        self._current_episode_start = 0
        self._t = 0

    def update(self, td_error, target_prob, behavior_prob, done):
        assert self._t < self._maxlen

        sl = slice(self._current_episode_start, self._t)

        trace = self._trace_coefficient(target_prob, behavior_prob)
        self._trace_products[sl] *= trace
        self._discounting[sl] *= self._discount

        self._trace_products[self._t] += 1.0
        self._discounting[self._t] += 1.0

        sl = slice(self._current_episode_start, self._t + 1)

        eligibility = self._discounting[sl] * self._modify_trace_products(self._trace_products[sl])
        self._updates[sl] += td_error * eligibility

        self._t += 1

        if done:
            self._current_episode_start = self._t

    @abstractmethod
    def _trace_coefficient(self, target_prob, behavior_prob):
        raise NotImplementedError

    def _modify_trace_products(self, trace_products):
        return trace_products

    def get_updates_and_reset(self):
        updates = self._updates[:self._t].copy()
        self._reset()
        return updates


class IS(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return self._lambd * (target_prob / behavior_prob)


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


class Moretrace(IS):
    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return self._lambd * (target_prob / behavior_prob)

    def _modify_trace_products(self, trace_products):
        return np.minimum(1.0, trace_products)
