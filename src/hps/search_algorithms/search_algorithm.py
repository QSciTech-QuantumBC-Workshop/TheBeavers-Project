import numpy as np
from typing import Union, Tuple, Dict, Any, Type, Optional, List
from ..search_space import SearchSpace


class TrialPoint:
    r"""Class representing a trial point in the search space.

    :param point: The point in the search space.
    :type point: Dict[str, Any]
    :param value: The value of the point in the search space.
    :type value: Optional[Any]

    :ivar point: The point in the search space.
    :ivar value: The value of the point in the search space.
    """
    def __init__(
            self,
            point: Dict[str, Any],
            value: Optional[Any] = None,
            pred_value: Optional[Any] = None
    ):
        self.point = point
        self.value = value
        self.pred_value = pred_value

    def __repr__(self):
        return f"{self.__class__.__name__}(point={self.point}, value={self.value}, pred_value={self.pred_value})"


class SearchHistory:
    r"""Class representing the history of the search algorithm.

    :ivar history: The history of the search algorithm.
    """
    def __init__(self):
        self.history: List[TrialPoint] = []

    @property
    def points(self):
        return [tp.point for tp in self.history]

    @property
    def values(self):
        return [tp.value for tp in self.history]

    def get_ordered_points(self, order: List[str]) -> List[List[Any]]:
        """
        Get the points where the keys are ordered according to the order list.

        :param order: The order of the keys.
        :type order: List[str]
        :return: The points where the keys are ordered according to the order list.
        :rtype: List[List[Any]]
        """
        return [[point[key] for key in order] for point in self.points]

    def update(self, trial_point: TrialPoint):
        self.history.append(trial_point)

    def append(self, trial_point: TrialPoint):
        self.update(trial_point)

    def __getitem__(self, item):
        return self.history[item]

    def __len__(self):
        return len(self.history)

    def get_best_point(self) -> TrialPoint:
        r"""Get the best point in the search space.

        :return: The best point in the search space.
        """
        return max(self.history, key=lambda tp: tp.value)

    def get_pred_best_point(self) -> TrialPoint:
        r"""Get the best point in the search space.

        :return: The best point in the search space.
        """
        return max(self.history, key=lambda tp: tp.pred_value)


class SearchAlgorithm:
    r"""Base class for search algorithms.

    :param search_space: The search space to use for searching the best hyperparameters.
    :type search_space: Optional[SearchSpace]
    :param config: Additional configuration parameters.
    :type config: Any

    :ivar search_space: The search space to use for searching the best hyperparameters.
    :ivar config: Additional configuration parameters.
    :ivar history: The history of the search algorithm.
    """
    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        self.search_space: SearchSpace = search_space
        self.config = config
        self.history: SearchHistory = SearchHistory()

    def set_search_space(self, search_space: SearchSpace):
        r"""Set the search space to use for searching the best hyperparameters.

        :param search_space: The search space to use for searching the best hyperparameters.
        :type search_space: SearchSpace
        """
        self.search_space: SearchSpace = search_space

    def get_next_trial_point(self) -> TrialPoint:
        r"""Get the next trial point in the search space.

        :return: The next trial point in the search space.
        """
        raise NotImplementedError

    def update(self, trial_point: TrialPoint):
        self.history.append(trial_point)

    def get_best_point(self) -> TrialPoint:
        r"""Get the best point in the search space.

        :return: The best point in the search space.
        """
        return self.history.get_best_point()

    def make_x_y_from_history(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Make X and Y from the history of the search algorithm. X is the points in the search space and Y is the values
        of the points in the search space.

        :return: A tuple containing X and Y.
        """
        x = self.search_space.points_to_linear(self.history.get_ordered_points(self.search_space.keys))
        y = np.array(self.history.values)
        return x, y

    def make_hp_from_x(self, x) -> dict:
        r"""
        Make a hyperparameter dictionary from a point in the search space.
        """
        return {
            key: value
            for key, value in zip(self.search_space.keys, x)
        }

