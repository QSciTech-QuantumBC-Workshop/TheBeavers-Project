from typing import Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import matplotlib.pyplot as plt
from ..search_algorithm import SearchAlgorithm, TrialPoint
from ...search_space import SearchSpace


class GPSearchAlgorithm(SearchAlgorithm):
    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        super().__init__(search_space, **config)
        self.model = GaussianProcessRegressor()
        self.warmup_trials = config.get('warmup_trials', 10)
        self.space_quantization = config.get('space_quantization', 100)
        self._fit = False

    def get_next_trial_point(self) -> TrialPoint:
        if len(self.history) < self.warmup_trials:
            return TrialPoint(
                point=self.search_space.sample()
            )
        x, y = self.make_x_y_from_history()
        self.model.fit(x, y)
        self._fit = True

        linear_space = np.linspace(0, 1, self.space_quantization)
        linear_x = np.stack([linear_space] * len(self.search_space.dimensions), axis=1)
        ei, mu, sigma = self.expected_improvement(linear_x, y)
        next_x = linear_x[np.argmax(ei)]
        next_xhp = self.search_space.point_from_linear(next_x)
        next_hp = self.make_hp_from_x(next_xhp)
        return TrialPoint(point=next_hp, pred_value=np.max(mu))

    def expected_improvement(self, x, y):
        mu, sigma = self.model.predict(x, return_std=True)
        best_y = np.max(y)
        z = (mu - best_y) / sigma
        ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei, mu, sigma

    def get_best_point(self) -> TrialPoint:
        x, y = self.make_x_y_from_history()
        linear_space = np.linspace(0, 1, self.space_quantization)
        linear_x = np.stack([linear_space] * len(self.search_space.dimensions), axis=1)
        ei, mu, sigma = self.expected_improvement(linear_x, y)
        best_x = linear_x[np.argmax(mu)]
        best_xhp = self.search_space.point_from_linear(best_x)
        best_hp = self.make_hp_from_x(best_xhp)
        return TrialPoint(point=best_hp, value=np.max(y))

    def plot_expected_improvement(
            self,
            *,
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            show: bool = True,
            **kwargs
    ):
        x, y = self.make_x_y_from_history()
        if not self._fit:
            self.model.fit(x, y)
            self._fit = True
        linear_space = np.linspace(0, 1, self.space_quantization)
        linear_x = np.stack([linear_space] * len(self.search_space.dimensions), axis=1)
        self.search_space.fit_reducer(linear_x, k=1)
        linear_x_1d = self.search_space.reducer_transform(linear_x, k=1)
        ei, mu, sigma = self.expected_improvement(linear_x, y)

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.plot(linear_x_1d, ei, label='Expected Improvement', color='red')
        ax.plot(linear_x_1d, mu, label='Mean', color='blue')
        ax.fill_between(linear_x_1d, mu - sigma, mu + sigma, alpha=0.1, color='blue')

        # add points from history as scatter
        x_1d = self.search_space.reducer_transform(x, k=1)
        ax.scatter(
            x_1d, y,
            label='History',
            color='green',
            marker='x'
        )

        ax.legend()
        ax.set_xlabel('Hyperparameter Space [-]')
        ax.set_ylabel('Score [-]')
        if show:
            plt.show()
        return
