from abc import ABC, abstractmethod

import numpy as np


# Set True for trace decay experiment:
VERBOSE = False


class EligibilityTrace(ABC):
    def __init__(self, discount, lambd):
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= lambd <= 1.0
        self.discount = discount
        self.lambd = lambd
        self._on_done()

    def _check_input_lengths(self, *args):
        L = len(args[0])
        for array in args:
            assert array.ndim == 1
            assert L == len(array)
        return L

    def __call__(self, td_errors, behavior_probs, target_probs, dones):
        assert (behavior_probs > 0.0).all()
        L = self._check_input_lengths(td_errors, behavior_probs, target_probs, dones)
        eligibilities = np.ones(L)
        updates = np.zeros(L)

        current_episode_start = 0
        for t in range(L):  # For each timestep in the trajectory:
            # Decay all past eligibilities
            sl = slice(current_episode_start, t)
            trace = self._trace_coefficient(target_probs[t], behavior_probs[t])
            eligibilities[sl] *= self.discount * trace

            if VERBOSE:
                if current_episode_start == 0:
                    print(eligibilities[0])

            # Apply current TD error to all past/current timesteps in proportion to eligibilities
            sl = slice(current_episode_start, t + 1)
            updates[sl] += td_errors[t] * eligibilities[sl]

            if dones[t]:
                current_episode_start = t + 1
                self._on_done()

        return updates

    def _on_done(self):
        pass

    @abstractmethod
    def _trace_coefficient(self, target_prob, behavior_prob):
        raise NotImplementedError


class TrajectoryAwareEligibilityTrace(EligibilityTrace):
    def __call__(self, td_errors, behavior_probs, target_probs, dones):
        assert (behavior_probs > 0.0).all()
        L = self._check_input_lengths(td_errors, behavior_probs, target_probs, dones)
        discount_products = np.ones(L)
        lambda_products = np.ones(L)
        isratio_products = np.ones(L)
        eligibilities = np.ones(L)
        updates = np.zeros(L)
        isratios = target_probs / behavior_probs

        current_episode_start = 0
        for t in range(L):  # For each timestep in the trajectory:
            # Decay all past eligibilities
            sl = slice(current_episode_start, t)
            discount_products[sl] *= self.discount
            lambda_products[sl] *= self.lambd
            isratio_products[sl] *= isratios[t]
            eligibilities[sl] = discount_products[sl] * self._compute_eligibility(lambda_products[sl], isratio_products[sl], isratios[sl], eligibilities[sl])

            # Apply current TD error to all past/current timesteps in proportion to eligibilities
            sl = slice(current_episode_start, t + 1)
            updates[sl] += td_errors[t] * eligibilities[sl]

            if VERBOSE:
                if current_episode_start == 0:
                    print(eligibilities[0])

            if dones[t]:
                current_episode_start = t + 1

        return updates

    def _trace_coefficient(self, target_prob, behavior_prob):
        pass

    @abstractmethod
    def _compute_eligibility(self, lambda_products, isratio_products):
        raise NotImplementedError


class IS(EligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return self.lambd * target_prob / behavior_prob


class Qlambda(EligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        return self.lambd


class TB(EligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        return self.lambd * target_prob


class Retrace(EligibilityTrace):
    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        return self.lambd * min(1.0, target_prob / behavior_prob)


class RecursiveRetrace(EligibilityTrace):
    def _on_done(self):
        self._trace_history = 1.0

    def _trace_coefficient(self, target_prob, behavior_prob):
        assert behavior_prob > 0.0
        trace = self.lambd * min(1.0 / (self._trace_history + 1e-6), target_prob / behavior_prob)
        self._trace_history *= trace
        return trace


class Moretrace(TrajectoryAwareEligibilityTrace):
    def _compute_eligibility(self, lambda_products, isratio_products, isratios, eligibilities):
        return np.minimum(lambda_products, isratio_products)


class Moretrace2(TrajectoryAwareEligibilityTrace):
    def _compute_eligibility(self, lambda_products, isratio_products, isratios, eligibilities):
        return np.minimum(lambda_products, eligibilities * isratios)


class RecursiveRetrace2(TrajectoryAwareEligibilityTrace):
    def _compute_eligibility(self, lambda_products, isratio_products, isratios, eligibilities):
        return self.lambd * np.minimum(1.0, eligibilities * isratios)


class TruncatedIS(TrajectoryAwareEligibilityTrace):
    def _compute_eligibility(self, lambda_products, isratio_products, isratios, eligibilities):
        return lambda_products * np.minimum(1.0, isratio_products)
