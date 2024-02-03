from typing import Optional, Tuple

import numpy as np
from sklearn.svm import SVR
from ..search_algorithm import SearchAlgorithm, TrialPoint
from ...search_space import SearchSpace


class SVRSearchAlgorithm(SearchAlgorithm):
    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        super().__init__(search_space, **config)
        self.model = SVR()
        self.epsilon = config.get('epsilon', 0.1)

    def get_next_trial_point(self) -> TrialPoint:
        is_rnd = np.random.rand() < self.epsilon
        if len(self.history) < 2 or is_rnd:
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
        y_pred = self.model.predict(linear_x)
        next_x = linear_x[np.argmax(y_pred)]
        next_xhp = self.search_space.point_from_numeric(next_x)
        next_hp = self.make_hp_from_x(next_xhp)
        return TrialPoint(point=next_hp)


