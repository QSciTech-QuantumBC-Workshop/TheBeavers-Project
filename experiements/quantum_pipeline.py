from typing import Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.algorithms import QSVR
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel, FidelityQuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.utils.loss_functions import L2Loss
from sklearn import datasets
from sklearn.model_selection import train_test_split

import hps
from experiements.classical_pipeline import MlpPipeline
from hps import TrialPoint, SearchAlgorithm, SearchSpace


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


class QSVRSearchAlgorithm(SearchAlgorithm):
    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        super().__init__(search_space, **config)
        self.model = None
        self.epsilon = config.get('epsilon', 0.1)
        self.warmup_trials = config.get('warmup_trials', 10)

    def make_qsvr_(self, x, y):
        # training_params = ParameterVector("θ", 1)
        # fm0 = QuantumCircuit(2)
        # fm0.ry(training_params[0], 0)
        # fm0.ry(training_params[0], 1)

        # Use ZZFeatureMap to represent input data
        # fm1 = ZZFeatureMap(2)

        # Create the feature map, composed of our two circuits
        # fm = fm0.compose(fm1)
        feature_map = ZFeatureMap(feature_dimension=len(x[0]), reps=3, insert_barriers=True)

        # Instantiate quantum kernel
        # quant_kernel = TrainableFidelityQuantumKernel(
        #     feature_map=feature_map,
        #     # training_parameters=training_params
        # )

        # Set up the optimizer
        # cb_qkt = QKTCallback()
        # spsa_opt = SPSA(maxiter=10, callback=cb_qkt.callback, learning_rate=0.05, perturbation=0.05)

        # Instantiate a quantum kernel trainer.
        # qkt = QuantumKernelTrainer(
        #     quantum_kernel=quant_kernel, loss="KernelLoss", optimizer=spsa_opt, initial_point=[np.pi / 2]
        # )
        # qka_results = qkt.fit(x, y)
        # optimized_kernel = qka_results.quantum_kernel
        self.model = QSVR(quantum_kernel=FidelityQuantumKernel(feature_map=feature_map))
        return self

    def get_next_trial_point(self) -> TrialPoint:
        is_rnd = np.random.rand() < self.epsilon
        if len(self.history) < self.warmup_trials or is_rnd:
            return TrialPoint(
                point=self.search_space.sample()
            )
        x, y = self.make_x_y_from_history()
        if self.model is None:
            self.make_qsvr_(x, y)
        self.model.fit(x, y)

        linear_space = np.linspace(0, 1, 100)
        linear_x = self.search_space.points_to_numeric([
            [dim.get_linear(v) for dim in self.search_space.dimensions]
            for v in linear_space
        ])
        y_pred = self.model.predict(linear_x)
        next_x = linear_x[np.argmax(y_pred)]
        next_xhp = self.search_space.point_from_numeric(next_x)
        next_hp = self.make_hp_from_x(next_xhp)
        return TrialPoint(point=next_hp)


if __name__ == '__main__':
    x, y, *_ = datasets.make_multilabel_classification()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline = hps.HpSearchPipeline(
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=MlpPipeline,
        search_algorithm=hps.QGPSearchAlgorithm(warmup_trials=2),
        search_space=hps.SearchSpace(
            # hps.Real("learning_rate_init", 1e-4, 1e-1),
            hps.Integer("hidden_layer_size", 8, 512),
            hps.Integer("n_hidden_layer", 1, 10),
            # hps.Categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
        ),
        n_trials=100,
    )
    out = pipeline.run()
    pipeline.plot_score_history(show=True)
    # pipeline.plot_hyperparameters_search(show=True)
    print(out.best_hyperparameters)
    pipeline.search_algorithm.plot_expected_improvement(show=True)
