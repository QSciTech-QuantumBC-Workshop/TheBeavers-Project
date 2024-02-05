from typing import Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from ..search_algorithm import SearchAlgorithm, TrialPoint
from ...search_space import SearchSpace
from ...search_algorithms import GPRSearchAlgorithm


class QiskitKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, feature_dimension):
        super().__init__()
        self.feature_dimension = feature_dimension
        adhoc_feature_map = ZZFeatureMap(
            feature_dimension=feature_dimension,
            reps=3,
            entanglement="linear",
        )
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)
        self.adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)

    def __call__(self, X, Y=None, eval_gradient=False):
        return self.adhoc_kernel.evaluate(x_vec=X, y_vec=Y)


class QGPRSearchAlgorithm(GPRSearchAlgorithm):
    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        super().__init__(search_space, **config)
        self.model = None

    def make_model(self):
        self.model = GaussianProcessRegressor(kernel=QiskitKernel(feature_dimension=len(self.search_space.dimensions)))

    def get_next_trial_point(self) -> TrialPoint:
        if self.model is None:
            self.make_model()
        return super().get_next_trial_point()



