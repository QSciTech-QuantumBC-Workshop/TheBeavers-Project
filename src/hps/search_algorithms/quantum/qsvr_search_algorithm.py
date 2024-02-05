from typing import Optional

from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.algorithms import QSVR
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from ..search_algorithm import TrialPoint
from ...search_algorithms import SVRSearchAlgorithm
from ...search_space import SearchSpace


class QKTCallback:
    """Callback wrapper class."""

    def __init__(self) -> None:
        self._data = [[] for i in range(5)]

    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
        """
        Args:
            x0: number of function evaluations
            x1: the parameters
            x2: the function value
            x3: the stepsize
            x4: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]


class QSVRSearchAlgorithm(SVRSearchAlgorithm):
    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        super().__init__(search_space, **config)
        self.model = None

    def make_model(self):
        feature_map = ZFeatureMap(feature_dimension=len(self.search_space.dimensions), reps=3, insert_barriers=True)
        self.model = QSVR(quantum_kernel=FidelityQuantumKernel(feature_map=feature_map))
        return self

    def get_next_trial_point(self) -> TrialPoint:
        if self.model is None:
            self.make_model()
        return super().get_next_trial_point()
