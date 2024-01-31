from typing import Dict, Any

import numpy as np


class SearchSpace:
    r"""Class representing a search space.

    :param space: The search space. It is a dictionary where the keys are the names of the parameters
        and the values are the values or the ranges of the parameters.
    :type space: Dict[str, Any]

    :ivar space: The search space.
    """
    def __init__(self, space: Dict[str, Any]):
        self.space = space

    def get_random_point(self) -> Dict[str, Any]:
        r"""Get a random point in the search space.

        :return: A random point in the search space.
        """
        point = {}
        for param, values in self.space.items():
            if isinstance(values, list):
                point[param] = np.random.choice(values)
            else:
                point[param] = np.random.uniform(values[0], values[1])
        return point
