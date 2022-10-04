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

        # Decay all past eligibilities
        for k, (s, a) in enumerate(self.sa_pairs):
            self.discount_products[k] *= self.discount
            self.lambda_products[k] *= self.lambd
            self.isratio_products[k] = min(MAX_FLOAT32, self.isratio_products[k] * isratio)
            self.betas[k] = self._compute_beta(k, isratio)
            self.pi_products[k] = self.pi_products[k] * target_prob
            self.mu_products[k] = self.mu_products[k] * behavior_prob

        # Increment eligibility for the current state-action pair
        self.sa_pairs.append((state, action))
        for lst in [self.discount_products, self.lambda_products, self.isratio_products, self.betas, self.pi_products, self.mu_products]:
            lst.append(1.0)

        # Apply current TD error to all timesteps in proportion to eligibility
        for k, (s, a) in enumerate(self.sa_pairs):
            self.Q[s,a] += self.alpha * (self.discount_products[k] * self.betas[k]) * td_error

    @abstractmethod
    def _compute_beta(self, k, isratio):
        assert 0 <= k < len(self.sa_pairs)
        return None

    def reset_traces(self):
        self.sa_pairs = []
        self.discount_products = []
        self.lambda_products = []
        self.isratio_products = []
        self.betas = []
        self.pi_products = []
        self.mu_products = []


class Retrace(OnlineEligibilityTrace):
    def _compute_beta(self, k, isratio):
        super()._compute_beta(k, isratio)
        trace = self.lambd * min(1.0, isratio)
        return trace * self.betas[k]


class TruncatedIS(OnlineEligibilityTrace):
    def _compute_beta(self, k, isratio):
        super()._compute_beta(k, isratio)
        lambda_product = self.lambda_products[k]
        isratio_product = self.isratio_products[k]
        return lambda_product * min(1.0, isratio_product)


class RecursiveRetrace(OnlineEligibilityTrace):
    def _compute_beta(self, k, isratio):
        super()._compute_beta(k, isratio)
        return self.lambd * min(1.0, isratio * self.betas[k])


class Moretrace(OnlineEligibilityTrace):
    def _compute_beta(self, k, isratio):
        super()._compute_beta(k, isratio)
        lambda_product = self.lambda_products[k]
        isratio_product = self.isratio_products[k]
        return min(lambda_product, isratio_product)


class Newtrace(OnlineEligibilityTrace):
    def _compute_beta(self, k, isratio):
        super()._compute_beta(k, isratio)
        lambda_product = self.lambda_products[k]
        f = lambda x: 2.0 - x
        return lambda_product * self.pi_products[k] * f(self.mu_products[k])
