from abc import ABC, abstractmethod

import numpy as np

from trajectory_aware_etraces.algorithms.vector import Vector


# Use this number as "infinity"
MAX_FLOAT32 = np.finfo(np.float32).max


class EligibilityTraces(ABC):
    def __init__(self, discount, lambd):
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= lambd <= 1.0
        self.discount = discount
        self.lambd = lambd
        self.reset()

    def accumulate_step(self, td_error, behavior_prob, target_prob, done):
        updates = self._process_step(td_error, behavior_prob, target_prob)
        self._accumulated_updates.append(updates)

    def accumulate_trajectory(self, td_errors, behavior_probs, target_probs, dones):
        for step in zip(td_errors, behavior_probs, target_probs, dones):
            self.accumulate_step(*step)

    def _process_step(self, td_error, behavior_prob, target_prob):
        assert 0.0 < behavior_prob <= 1.0
        assert 0.0 <= target_prob <= 1.0
        isratio = target_prob / behavior_prob

        # Decay all past eligibilities
        self.discount_products.multiply(self.discount)
        self.lambda_products.multiply(self.lambd)
        self.isratio_products.assign(
            np.minimum(MAX_FLOAT32, self.isratio_products.numpy() * isratio)
        )
        # NOTE: Beta update must come last or implementation will be incorrect
        self._update_beta(isratio)

        # Increment eligibility for the current state-action pair
        for vector in self.eligibilities:
            vector.append(1.0)

        # Scale current TD error for all timesteps in proportion to their eligibilities
        updates = (self.discount_products.numpy() * self.betas.numpy()) * td_error
        return updates

    def get_updates(self):
        total_updates = reduce_sum(self._accumulated_updates)
        self._accumulated_updates.clear()
        return total_updates

    @abstractmethod
    def _update_beta(self, isratio):
        raise NotImplementedError

    def reset(self):
        self._accumulated_updates = []
        # Here, an "eligibility" is any vector that is incremented by 1 after a state-action visit
        self.discount_products, self.lambda_products, self.isratio_products, self.betas, \
            = self.eligibilities \
            = tuple(Vector() for _ in range(4))


def reduce_sum(arrays):
    T = max([len(a) for a in arrays])
    arrays = [np.pad(a, pad_width=(0, T-len(a))) for a in arrays]
    for a in arrays:
        assert len(a) == T
    return sum(arrays)
