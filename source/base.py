""" This file contains the BaseSeedClass class. This file sets the seed and rng for reproducibility. """

from os import environ

# We set the number of threads to one in order to outperform parallelization
environ["OMP_NUM_THREADS"] = "1"
environ["OPENBLAS_NUM_THREADS"] = "1"
environ["MKL_NUM_THREADS"] = "1"
environ["VECLIB_MAXIMUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class BaseSeededClass:

    def __init__(self, seed=None, **kwargs):

        self.seed = seed if seed is not None else np.random.randint(1e9) # Set the seed for reproducibility, if none, it is set randomnly
        self.rng = np.random.default_rng(self.seed) # Set the random number generator with the seed
        # print(f"[BaseSeededClass] Seed: {self.seed}")

        super().__init__() # required for inheritance