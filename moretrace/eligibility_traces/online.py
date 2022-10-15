from abc import ABC, abstractmethod

import numpy as np


# Use this number as "infinity"
MAX_FLOAT32 = np.finfo(np.float32).max


class OnlineEligibilityTrace(ABC):
    def __init__(self, discount, lambd):
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= lambd <= 1.0
        self.discount = discount
        self.lambd = lambd
        self._L = 10_000  # Max array length, must be large enough to fit one episode
        self.reset_traces()

    def set(self, Q, alpha):
        # TODO: Put this in the class constructor
        self.Q = Q
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha

    def step(self, state, action, td_error, behavior_prob, target_prob):
        assert 0.0 < behavior_prob <= 1.0
        assert 0.0 <= target_prob <= 1.0
        isratio = target_prob / behavior_prob

        self.states[self.t] = state
        self.actions[self.t] = action

        # Decay all past eligibilities
        sl = slice(0, self.t)
        self.discount_products[sl] *= self.discount
        self.lambda_products[sl] *= self.lambd
        self.isratio_products[sl] = np.minimum(MAX_FLOAT32, self.isratio_products[sl] * isratio)
        self.pi_products[sl] *= target_prob
        self.mu_products[sl] *= behavior_prob
        self.retrace_products[sl] *= min(1.0, isratio)
        # NOTE: Beta update must come last or implementation will be incorrect
        self.betas[sl] = self._compute_beta(sl, isratio)

        # Increment eligibility for the current state-action pair
        # (Arrays are initialized as ones, so incrementing t handles this implicitly)
        self.t += 1

        # Apply current TD error to all timesteps in proportion to eligibility
        # TODO: Can we vectorize this?
        for k in range(self.t):
            s, a = self.states[k], self.actions[k]
            self.Q[s,a] += self.alpha * (self.discount_products[k] * self.betas[k]) * td_error

    @abstractmethod
    def _compute_beta(self, sl, isratio):
        raise NotImplementedError

    def reset_traces(self):
        self.t = 0  # Current episode length

        # NOTE: We assume states/actions are discrete integers here
        self.states = np.empty(self._L, dtype=np.int32)
        self.actions = np.empty(self._L, dtype=np.int32)

        self.discount_products = self.ones()
        self.lambda_products = self.ones()
        self.isratio_products = self.ones()
        self.betas = self.ones()
        # TODO: pi/mu are not currently being used and could be removed
        self.pi_products = self.ones()
        self.mu_products = self.ones()
        self.retrace_products = self.ones()

    def ones(self, element_dtype=np.float64):
        return np.ones(self._L, dtype=element_dtype)


class Retrace(OnlineEligibilityTrace):
    def _compute_beta(self, sl, isratio):
        trace = self.lambd * min(1.0, isratio)
        return trace * self.betas[sl]


class TruncatedIS(OnlineEligibilityTrace):
    def _compute_beta(self, sl, isratio):
        lambda_product = self.lambda_products[sl]
        isratio_product = self.isratio_products[sl]
        return lambda_product * np.minimum(1.0, isratio_product)


class RecursiveRetrace(OnlineEligibilityTrace):
    def _compute_beta(self, sl, isratio):
        return self.lambd * np.minimum(1.0, isratio * self.betas[sl])


class Moretrace(OnlineEligibilityTrace):
    def _compute_beta(self, sl, isratio):
        lambda_product = self.lambda_products[sl]
        isratio_product = self.isratio_products[sl]
        return np.minimum(lambda_product, isratio_product)


class Moretrace2(OnlineEligibilityTrace):
    def _compute_beta(self, sl, isratio):
        lambda_product = self.lambda_products[sl]
        retrace_product = self.retrace_products[sl]
        return np.minimum(lambda_product, retrace_product)
