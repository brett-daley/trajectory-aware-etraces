import numpy as np

from trajectory_aware_etraces.algorithms.eligibility_traces import EligibilityTraces


class Retrace(EligibilityTraces):
    def _update_beta(self, isratio):
        trace = self.lambd * min(1.0, isratio)
        self.betas.multiply(trace)


class TruncatedIS(EligibilityTraces):
    def _update_beta(self, isratio):
        lambda_product = self.lambda_products.numpy()
        isratio_product = self.isratio_products.numpy()
        self.betas.assign(lambda_product * np.minimum(1.0, isratio_product))


class RecursiveRetrace(EligibilityTraces):
    def _update_beta(self, isratio):
        betas = self.betas.numpy()
        self.betas.assign(self.lambd * np.minimum(1.0, isratio * betas))


class RBIS(EligibilityTraces):
    def _update_beta(self, isratio):
        lambda_product = self.lambda_products.numpy()
        betas = self.betas.numpy()
        self.betas.assign(np.minimum(lambda_product, isratio * betas))
