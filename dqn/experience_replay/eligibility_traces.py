from abc import ABC, abstractmethod

import numpy as np


class OfflineEligibilityTrace(ABC):
    def __init__(self, discount, lambd):
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= lambd <= 1.0
        self._discount = discount
        self._lambd = lambd

    def __call__(self, td_errors, behavior_probs, target_probs, dones):
        L = len(td_errors)
        for array in [behavior_probs, target_probs, dones]:
            assert array.ndim == 1
            assert L == len(array)

        eligibility = np.zeros(L)
        updates = np.zeros(L)

        current_episode_start = 0
        for t in range(L):  # For each timestep in the trajectory:
            # Decay all past eligibilities
            sl = slice(current_episode_start, t)
            trace = self._trace_coefficient(target_probs[t], behavior_probs[t])
            eligibility[sl] *= self._discount * trace

            # Increment eligibility of current timestep
            eligibility[t] += 1.0

            # Apply current TD error to all past/current timesteps in proportion to eligibilities
            sl = slice(current_episode_start, t + 1)
            updates[sl] += td_errors[t] * eligibility[sl]

            if dones[t]:
                current_episode_start = t + 1

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
    def __init__(self, discount, lambd, p=0.5):
        super().__init__(discount, lambd)
        assert 0.0 <= p <= 1.0
        self._p = p

    def __call__(self, td_errors, behavior_probs, target_probs, dones):
        assert (behavior_probs > 0.0).all()

        L = len(td_errors)
        for array in [behavior_probs, target_probs, dones]:
            assert array.ndim == 1
            assert L == len(array)

        # The eligibility has been split into subcomponents for efficiency
        decay_products = np.ones(L)
        isratio_products = np.ones(L)
        retrace_products = np.ones(L)
        updates = np.zeros(L)

        current_episode_start = 0
        for t in range(L):  # For each timestep in the trajectory:
            # Decay all past eligibilities
            sl = slice(current_episode_start, t)
            decay_products[sl] *= self._discount * self._lambd
            ratio = target_probs[t] / behavior_probs[t]
            isratio_products[sl] *= ratio
            retrace_products[sl] *= min(1.0, ratio)

            # Apply current TD error to all past/current timesteps in proportion to eligibilities
            sl = slice(current_episode_start, t + 1)
            eligibility = decay_products[sl] \
                            * np.minimum(1.0, isratio_products[sl]) ** self._p \
                            * retrace_products[sl] ** (1.0 - self._p)
            updates[sl] += td_errors[t] * eligibility

            if dones[t]:
                current_episode_start = t + 1

        return updates

    def _trace_coefficient(self, target_prob, behavior_prob):
        raise NotImplementedError
