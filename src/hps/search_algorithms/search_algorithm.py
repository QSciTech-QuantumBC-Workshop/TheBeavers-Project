import os
import pickle
import json

import numpy as np
from typing import Union, Tuple, Dict, Any, Type, Optional, List

from matplotlib import pyplot as plt

from ..search_space import SearchSpace
from ..tools import to_json


class TrialPoint:
    r"""Class representing a trial point in the search space.

    :param point: The point in the search space.
    :type point: Dict[str, Any]
    :param value: The value of the point in the search space.
    :type value: Optional[Any]
    :param best_pred_value: The best predicted value of the point in the search space.
    :type best_pred_value: Optional[Any]

    :ivar point: The point in the search space.
    :ivar value: The value of the point in the search space.
    :ivar best_pred_value: The best predicted value of the point in the search space.
    """
    def __init__(
            self,
            point: Dict[str, Any],
            value: Optional[Any] = None,
            best_pred_value: Optional[Any] = None
    ):
        self.point = point
        self.value = value
        self.best_pred_value = best_pred_value

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"point={self.point}, "
                f"value={self.value}, "
                f"best_pred_value={self.best_pred_value}"
                f")")

    def to_json(self, filename: Optional[str] = None):
        data = {
            "point": self.point,
            "value": self.value,
            "best_pred_value": self.best_pred_value
        }
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
        return data


class SearchHistory:
    r"""Class representing the history of the search algorithm.

    :ivar history: The history of the search algorithm.
    """
    def __init__(self, history: Optional[List[TrialPoint]] = None, **kwargs):
        self.history: List[TrialPoint] = history or []

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
        filtered = [tp for tp in self.history if tp.best_pred_value is not None]
        if len(filtered) == 0:
            return TrialPoint(point={}, value=None, best_pred_value=None)
        return max(filtered, key=lambda tp: tp.best_pred_value)

    def to_json(self, filename: Optional[str] = None):
        data = [tp.to_json() for tp in self.history]
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
        return data

    def __json__(self):
        return self.to_json()


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

    NON_SERIALIZE_ATTRIBUTES = ["model"]

    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        self.search_space: SearchSpace = search_space
        self.config = config
        self.history: SearchHistory = config.get('history', SearchHistory())
        self.warmup_history: SearchHistory = config.get('warmup_history', SearchHistory())
        self.warmup_x, self.warmup_y = self.make_x_y_from_history(self.warmup_history)
        self.space_quantization = config.get('space_quantization', 100)

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

    def make_x_y_from_history(self, history: Optional[SearchHistory] = None) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Make X and Y from the history of the search algorithm. X is the points in the search space and Y is the values
        of the points in the search space.

        :param history: The history of the search algorithm.
        :type history: Optional[SearchHistory]
        :return: A tuple containing X and Y.
        """
        if history is None:
            history = self.history
        if len(history) == 0:
            return np.array([]), np.array([])
        x = self.search_space.points_to_linear(history.get_ordered_points(self.search_space.keys))
        y = np.array(history.values)
        return x, y

    def make_hp_from_x(self, x) -> dict:
        r"""
        Make a hyperparameter dictionary from a point in the search space.
        """
        return {
            key: value
            for key, value in zip(self.search_space.keys, x)
        }

    def plot_search(
            self,
            *,
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            show: bool = True,
            filename: Optional[str] = None,
            **kwargs
    ):
        if kwargs.get("as_violin", False):
            return self.plot_violin_search(fig=fig, ax=ax, show=show, filename=filename, **kwargs)
        x, y = kwargs.get('x', None), kwargs.get('y', None)
        if x is None or y is None:
            x, y = self.make_x_y_from_history()
        linear_x = kwargs.get('linear_x', None)
        if linear_x is None:
            linear_space = np.linspace(0, 1, self.space_quantization)
            linear_x = np.stack([linear_space] * len(self.search_space.dimensions), axis=1)
        self.search_space.fit_reducer(linear_x, k=1, if_not_fitted=True)
        if fig is None or ax is None:
            fig, ax = plt.subplots()

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

    def plot_violin_search(
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
        linear_x = kwargs.get('linear_x', None)
        if linear_x is None:
            linear_space = np.linspace(0, 1, self.space_quantization)
            linear_x = np.stack([linear_space] * len(self.search_space.dimensions), axis=1)
        self.search_space.fit_reducer(linear_x, k=1, if_not_fitted=True)
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # add points from history as scatter
        x_1d = self.search_space.reducer_transform(x, k=1)
        violin_position = kwargs.get('violin_position', 0)
        ax.violinplot(
            x_1d,
            widths=0.1,
            showmeans=True
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

    @classmethod
    def from_pickle(cls, path: str) -> "SearchAlgorithm":
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_pickle_or_new(cls, path: str, **kwargs) -> "SearchAlgorithm":
        if os.path.exists(path):
            return cls.from_pickle(path)
        return cls(**kwargs)

    def __json__(self):
        data = {
            k: to_json(v) for k, v in self.__dict__.items()
            if k not in self.NON_SERIALIZE_ATTRIBUTES
        }
        return to_json(data)
