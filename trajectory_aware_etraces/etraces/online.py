from abc import ABC, abstractmethod

import numpy as np

from trajectory_aware_etraces.etraces.vector import Vector


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

        self.states.append(state)
        self.actions.append(action)

        # Decay all past eligibilities
        self.discount_products.multiply(self.discount)
        self.lambda_products.multiply(self.lambd)
        self.isratio_products.assign( np.minimum(MAX_FLOAT32, self.isratio_products.numpy() * isratio) )
        self.pi_products.multiply(target_prob)
        self.mu_products.multiply(behavior_prob)
        self.retrace_products.multiply(min(1.0, isratio))
        # NOTE: Beta update must come last or implementation will be incorrect
        self._update_beta(isratio)

        # Increment eligibility for the current state-action pair
        for vector in self.eligibilities:
            vector.append(1.0)

        # Apply current TD error to all timesteps in proportion to eligibility
        sa_pairs = (self.states.numpy(), self.actions.numpy())
        updates = self.alpha * (self.discount_products.numpy() * self.betas.numpy()) * td_error
        np.add.at(self.Q, sa_pairs, updates)

    @abstractmethod
    def _update_beta(self, isratio):
        raise NotImplementedError

    def reset_traces(self):
        # NOTE: We assume states/actions are discrete integers here
        self.states = Vector(dtype=np.int32)
        self.actions = Vector(dtype=np.int32)

        # Here, an "eligibility" is any vector that is incremented by 1 after a state-action visit
        # TODO: pi/mu are not currently being used and could be removed
        self.discount_products, self.lambda_products, self.isratio_products, self.betas, \
            self.pi_products, self.mu_products, self.retrace_products \
            = self.eligibilities = [Vector() for _ in range(7)]


class Retrace(OnlineEligibilityTrace):
    def _update_beta(self, isratio):
        trace = self.lambd * min(1.0, isratio)
        self.betas.multiply(trace)


class TruncatedIS(OnlineEligibilityTrace):
    def _update_beta(self, isratio):
        lambda_product = self.lambda_products.numpy()
        isratio_product = self.isratio_products.numpy()
        self.betas.assign(lambda_product * np.minimum(1.0, isratio_product))


class RecursiveRetrace(OnlineEligibilityTrace):
    def _update_beta(self, isratio):
        betas = self.betas.numpy()
        self.betas.assign(self.lambd * np.minimum(1.0, isratio * betas))


class RBIS(OnlineEligibilityTrace):
    def _update_beta(self, isratio):
        lambda_product = self.lambda_products.numpy()
        betas = self.betas.numpy()
        self.betas.assign(np.minimum(lambda_product, isratio * betas))
