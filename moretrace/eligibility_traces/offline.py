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
        betas = np.ones(L)
        updates = np.zeros(L)
        isratios = target_probs / behavior_probs

        current_episode_start = 0
        for t in range(L):  # For each timestep in the trajectory:
            # Decay all past eligibilities
            sl = slice(current_episode_start, t)
            discount_products[sl] *= self.discount
            lambda_products[sl] *= self.lambd
            isratio_products[sl] *= isratios[t]
            betas[sl] = self._compute_betas(lambda_products[sl], isratio_products[sl], isratios[t], betas[sl])

            # Apply current TD error to all past/current timesteps in proportion to eligibilities
            sl = slice(current_episode_start, t + 1)
            eligibilities = discount_products[sl] * betas[sl]
            updates[sl] += td_errors[t] * eligibilities

            if VERBOSE:
                if current_episode_start == 0:
                    print(eligibilities[0])

            if dones[t]:
                current_episode_start = t + 1

        return updates

    def _trace_coefficient(self, target_prob, behavior_prob):
        pass

    @abstractmethod
    def _compute_betas(self, lambda_products, isratio_products, isratio, betas):
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


class RBIS(TrajectoryAwareEligibilityTrace):
    def _compute_betas(self, lambda_products, isratio_products, isratio, betas):
        return np.minimum(lambda_product, isratio * betas)


class RecursiveRetrace(TrajectoryAwareEligibilityTrace):
    def _compute_betas(self, lambda_products, isratio_products, isratio, betas):
        return self.lambd * np.minimum(1.0, betas * isratio)


class TruncatedIS(TrajectoryAwareEligibilityTrace):
    def _compute_betas(self, lambda_products, isratio_products, isratio, betas):
        return lambda_products * np.minimum(1.0, isratio_products)
