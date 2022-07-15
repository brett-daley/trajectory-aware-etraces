from abc import ABC, abstractmethod

import numpy as np


class OfflineEligibilityTrace(ABC):
    def __init__(self, discount, lambd):
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= lambd <= 1.0
        self.discount = discount
        self.lambd = lambd
        self._on_done()

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
            eligibility[sl] *= self.discount * trace

            # Increment eligibility of current timestep
            eligibility[t] += 1.0

            # Apply current TD error to all past/current timesteps in proportion to eligibilities
            sl = slice(current_episode_start, t + 1)
            updates[sl] += td_errors[t] * eligibility[sl]

            # Uncomment for trace decay experiment:
            # print(eligibility[0])

            if dones[t]:
                current_episode_start = t + 1
                self._on_done()

        return updates

    def _on_done(self):
        pass

    @abstractmethod
    def _trace_coefficient(self, target_prob, behavior_prob):
        raise NotImplementedError


class IS(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return self.lambd * target_prob / behavior_prob


class Qlambda(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        return self.lambd


class TB(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        return self.lambd * target_prob


class Retrace(OfflineEligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return self.lambd * min(1.0, target_prob / behavior_prob)


class RecursiveRetrace(OfflineEligibilityTrace):
    def _on_done(self):
        self._trace_history = 1.0

    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        trace = self.lambd * min(1.0 / (self._trace_history + 1e-6), target_prob / behavior_prob)
        self._trace_history *= trace
        return trace


class Moretrace(OfflineEligibilityTrace):
    def __call__(self, td_errors, behavior_probs, target_probs, dones):
        assert (behavior_probs > 0.0).all()

        L = len(td_errors)
        for array in [behavior_probs, target_probs, dones]:
            assert array.ndim == 1
            assert L == len(array)

        # The eligibility has been split into subcomponents for efficiency
        discount_products = np.zeros(L)
        lambda_products = np.zeros(L)
        isratio_products = np.zeros(L)
        updates = np.zeros(L)

        current_episode_start = 0
        for t in range(L):  # For each timestep in the trajectory:
            # Decay all past eligibilities
            sl = slice(current_episode_start, t)
            discount_products[sl] *= self.discount
            lambda_products[sl] *= self.lambd
            isratio_products[sl] *= (target_probs[t] / behavior_probs[t])

            # Increment eligibility of current timestep
            for array in [discount_products, lambda_products, isratio_products]:
                array[t] += 1.0

            # Apply current TD error to all past/current timesteps in proportion to eligibilities
            sl = slice(current_episode_start, t + 1)
            eligibility = self._compute_eligibility(discount_products[sl], lambda_products[sl], isratio_products[sl])
            updates[sl] += td_errors[t] * eligibility

            # Uncomment for trace decay experiment:
            # print(eligibility[0])

            if dones[t]:
                current_episode_start = t + 1

        return updates

    def _trace_coefficient(self, target_prob, behavior_prob):
        raise NotImplementedError

    def _compute_eligibility(self, discount_products, lambda_products, isratio_products):
        return discount_products * np.minimum(lambda_products, isratio_products)


class Moretrace2(Moretrace):
    def __call__(self, td_errors, behavior_probs, target_probs, dones):
        assert (behavior_probs > 0.0).all()

        L = len(td_errors)
        for array in [behavior_probs, target_probs, dones]:
            assert array.ndim == 1
            assert L == len(array)

        # The eligibility has been split into subcomponents for efficiency
        discount_products = np.zeros(L)
        lambda_products = np.zeros(L)
        isratio_products = np.zeros(L)
        eligibilities = np.ones(L)
        updates = np.zeros(L)

        isratios = behavior_probs / target_probs
        assert len(isratios) == L

        current_episode_start = 0
        for t in range(L):  # For each timestep in the trajectory:
            # Decay all past eligibilities
            sl = slice(current_episode_start, t)
            discount_products[sl] *= self.discount
            lambda_products[sl] *= self.lambd
            isratio_products[sl] *= isratios[t]

            # Increment eligibility of current timestep
            for array in [discount_products, lambda_products, isratio_products]:
                array[t] += 1.0

            # Apply current TD error to all past/current timesteps in proportion to eligibilities
            sl = slice(current_episode_start, t + 1)
            eligibilities[sl] = self._compute_eligibility(lambda_products[sl], isratio_products[sl], isratios[sl], eligibilities[sl])
            updates[sl] += td_errors[t] * discount_products[sl] * eligibilities[sl]

            # Decay must come *after* update
            eligibilities[sl] *= isratios[sl]

            # Uncomment for trace decay experiment:
            # print(eligibility[0])

            if dones[t]:
                current_episode_start = t + 1

        return updates

    def _compute_eligibility(self, lambda_products, isratio_products, isratios, eligibilities):
        return np.minimum(lambda_products, eligibilities)


class TruncatedIS(Moretrace):
    def _compute_eligibility(self, discount_products, lambda_products, isratio_products):
        return discount_products * lambda_products * np.minimum(1.0, isratio_products)
