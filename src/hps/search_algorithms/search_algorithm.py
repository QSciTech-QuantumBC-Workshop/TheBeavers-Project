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
    def __init__(self, point: Dict[str, Any], value: Optional[Any] = None):
        self.point = point
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}(point={self.point}, value={self.value})"


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
        self.history: List[TrialPoint] = []

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
        return max(self.history, key=lambda tp: tp.value)

