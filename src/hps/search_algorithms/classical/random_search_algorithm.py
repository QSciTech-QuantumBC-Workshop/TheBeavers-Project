from typing import Optional

from ..search_algorithm import SearchAlgorithm, TrialPoint
from ...search_space import SearchSpace


class RandomSearchAlgorithm(SearchAlgorithm):
    r"""Class for random search algorithm.

    :param search_space: The search space to use for searching the best hyperparameters.
    :type search_space: Optional[SearchSpace]
    :param config: Additional configuration parameters.
    :type config: Any

    :ivar search_space: The search space to use for searching the best hyperparameters.
    :ivar config: Additional configuration parameters.
    :ivar history: The history of the search algorithm.
    """
    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        super().__init__(search_space, **config)

    def get_next_trial_point(self) -> TrialPoint:
        r"""Get the next trial point in the search space.

        :return: The next trial point in the search space.
        """
        point = self.search_space.get_random_point()
        return TrialPoint(point)
