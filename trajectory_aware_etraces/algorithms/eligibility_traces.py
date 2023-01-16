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

        self._episode_buffer = []
        self._total_updates = []

        self._reset_traces()

    def accumulate_step(self, td_error, behavior_prob, target_prob, done):
        updates = self._process_step(td_error, behavior_prob, target_prob)
        self._episode_buffer.append(updates)

        if done:
            self._total_updates.append( reduce_sum(self._episode_buffer) )
            self._reset_traces()

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
        for vector in self.traces:
            vector.append(1.0)

        # Scale current TD error for all timesteps in proportion to their eligibilities
        updates = (self.discount_products.numpy() * self.betas.numpy()) * td_error
        return updates

    def get_updates(self):
        if self._episode_buffer:
            self._total_updates += [reduce_sum(self._episode_buffer)]

        updates = np.concatenate(self._total_updates)
        self._total_updates.clear()
        return updates

    @abstractmethod
    def _update_beta(self, isratio):
        raise NotImplementedError

    def _reset_traces(self):
        self._episode_buffer.clear()

        # Here, "traces" are any vector that is incremented by 1 after a state-action visit
        self.discount_products, self.lambda_products, self.isratio_products, self.betas, \
            = self.traces \
            = tuple(Vector() for _ in range(4))


def reduce_sum(arrays):
    assert arrays
    T = max([len(a) for a in arrays])
    arrays = [np.pad(a, pad_width=(0, T-len(a))) for a in arrays]
    for a in arrays:
        assert len(a) == T
    return sum(arrays)
