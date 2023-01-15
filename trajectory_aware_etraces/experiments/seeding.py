import numpy as np


def generate_seeds(meta_seed, n):
    np_random = np.random.RandomState(meta_seed)  # Set seed for reproducibility
    # Randomize the seeds here to avoid any possible bias from reusing seeds from the hyperparameter search
    return [np_random.randint(2**31) for _ in range(n)]
