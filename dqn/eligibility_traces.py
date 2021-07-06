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
        self._eligibility = np.zeros(self._maxlen, dtype=np.float64)
        self._updates = np.zeros(self._maxlen, dtype=np.float64)

        self._current_episode_start = 0
        self._t = 0

    def update(self, td_error, target_prob, behavior_prob, done):
        assert self._t < self._maxlen

        sl = slice(self._current_episode_start, self._t)

        self._trace_products[sl] *= self._trace_coefficient(target_prob, behavior_prob)
        self._discounting[sl] *= self._discount
        self._eligibility[sl] *= self._lambd

        self._trace_products[self._t] += 1.0
        self._discounting[self._t] += 1.0
        self._eligibility[self._t] += 1.0

        sl = slice(self._current_episode_start, self._t + 1)

        self._updates[sl] += td_error * self._discounting[sl] * self._eligibility[sl] \
            * self._get_trace_products(sl)

        self._t += 1

        if done:
            self._current_episode_start = self._t

    @abstractmethod
    def _trace_coefficient(self, target_prob, behavior_prob):
        raise NotImplementedError

    def _get_trace_products(self, sl):
        return self._trace_products[sl]

    def get_updates_and_reset(self):
        updates = self._updates[:self._t].copy()
        self._reset()
        return updates


class IS(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return target_prob / behavior_prob


class Qlambda(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        return 1.0


class TB(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        return target_prob


class Retrace(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return min(1.0, target_prob / behavior_prob)


class Moretrace(IS):
    def _get_trace_products(self, sl):
        return np.minimum(1.0, self._trace_products[sl])


class Safetrace(IS):
    def _get_trace_products(self, sl):
        return np.minimum(self._eligibility[sl], self._trace_products[sl])
