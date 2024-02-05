import os
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from ..search_algorithm import SearchAlgorithm, TrialPoint
from ...search_space import SearchSpace


class SVRSearchAlgorithm(SearchAlgorithm):
    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        super().__init__(search_space, **config)
        self.model = SVR()
        self.epsilon = config.get('epsilon', 0.1)
        self.warmup_trials = config.get('warmup_trials', 10)
        self.space_quantization = config.get('space_quantization', 100)
        self._fit = False

    def get_next_trial_point(self) -> TrialPoint:
        is_rnd = np.random.rand() < self.epsilon
        if len(self.history) < self.warmup_trials or is_rnd:
            return TrialPoint(
                point=self.search_space.sample()
            )
        x, y = self.make_x_y_from_history()
        self.model.fit(x, y)
        self._fit = True

        linear_space = np.linspace(0, 1, self.space_quantization)
        linear_x = np.stack([linear_space] * len(self.search_space.dimensions), axis=1)
        y_pred = self.model.predict(linear_x)
        next_x = linear_x[np.argmax(y_pred)]
        next_xhp = self.search_space.point_from_linear(next_x)
        next_hp = self.make_hp_from_x(next_xhp)
        return TrialPoint(point=next_hp, best_pred_value=np.max(y_pred))

    def plot_search(
            self,
            *,
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            show: bool = True,
            filename: Optional[str] = None,
            **kwargs
    ):
        x, y = kwargs.get('x', None), kwargs.get('y', None)
        if x is None or y is None:
            x, y = self.make_x_y_from_history()
        if not self._fit:
            self.model.fit(x, y)
            self._fit = True
        linear_x = kwargs.get('linear_x', None)
        if linear_x is None:
            linear_space = np.linspace(0, 1, self.space_quantization)
            linear_x = np.stack([linear_space] * len(self.search_space.dimensions), axis=1)
        y_pred = kwargs.get('y_pred', self.model.predict(linear_x))
        self.search_space.fit_reducer(linear_x, k=1)
        linear_x_1d = np.ravel(self.search_space.reducer_transform(linear_x, k=1))

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.plot(linear_x_1d, y_pred, label='Prediction', color=kwargs.get('color', 'blue'))

        # add points from history as scatter
        x_1d = self.search_space.reducer_transform(x, k=1)
        ax.scatter(
            x_1d, y,
            label='History',
            color=kwargs.get('color', 'green'),
            marker='x'
        )

        ax.legend()
        ax.set_xlabel('Hyperparameter Space [-]')
        ax.set_ylabel('Score [-]')

        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)

        if show:
            plt.show()
        return fig, ax

