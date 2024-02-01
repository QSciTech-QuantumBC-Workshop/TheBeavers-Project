from typing import Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from ..search_algorithm import SearchAlgorithm, TrialPoint
from ...search_space import SearchSpace


class GPSearchAlgorithm(SearchAlgorithm):
    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        super().__init__(search_space, **config)
        self.model = GaussianProcessRegressor()

    def make_x_y_from_history(self) -> Tuple[np.ndarray, np.ndarray]:
        x = self.search_space.points_to_numeric([
            [point.point[key] for key in self.search_space.keys]
            for point in self.history
        ])
        y = np.array([point.value for point in self.history])
        return x, y

    def make_hp_from_x(self, x) -> dict:
        return {
            key: value
            for key, value in zip(self.search_space.keys, x)
        }

    def get_next_trial_point(self) -> TrialPoint:
        if len(self.history) < 2:
            return TrialPoint(
                point=self.search_space.sample()
            )
        x, y = self.make_x_y_from_history()
        self.model.fit(x, y)

        linear_space = np.linspace(0, 1, 100)
        linear_x = self.search_space.points_to_numeric([
            [dim.get_linear(v) for dim in self.search_space.dimensions]
            for v in linear_space
        ])
        y_pred, y_std = self.model.predict(linear_x, return_std=True)
        next_x = linear_x[np.argmax(y_pred)]
        next_xhp = self.search_space.point_from_numeric(next_x)
        next_hp = self.make_hp_from_x(next_xhp)
        return TrialPoint(point=next_hp)

    def update(self, trial_point: TrialPoint):
        self.history.append(trial_point)

    def get_best_point(self) -> TrialPoint:
        best_point = max(self.history, key=lambda point: point.value)
        return best_point

