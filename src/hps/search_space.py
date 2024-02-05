from typing import Dict, Any, Optional, List

import numpy as np

from hps.tools import UMAP_PCA_1d_REDUCER


class Dimension:
    r"""Base class for a dimension.

    :param name: The name of the dimension.
    :type name: str
    """
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def get_rnd(self):
        raise NotImplementedError

    def get_linear(self, x: float):
        raise NotImplementedError

    def to_linear(self, value: Any) -> float:
        raise NotImplementedError

    def from_linear(self, value: float) -> Any:
        return self.get_linear(value)

    def to_numeric(self, value: Any) -> float:
        r"""Convert the value to a numeric representation.

        :param value: The value to convert to a numeric representation.
        :type value: Any
        :return: The numeric representation of the value.
        """
        raise NotImplementedError

    def from_numeric(self, value: float) -> Any:
        r"""Convert the value to a numeric representation.

        :param value: The value to convert to a numeric representation.
        :type value: Any
        :return: The numeric representation of the value.
        """
        raise NotImplementedError


class Real(Dimension):
    r"""Class representing a real number.

    :param name: The name of the real number.
    :type name: str
    :param low: The lower bound of the real number.
    :type low: float
    :param high: The upper bound of the real number.
    :type high: float
    """
    def __init__(self, name: str, low: float, high: float):
        super().__init__(name)
        self.low = low
        self.high = high

    def __repr__(self):
        return f"Real({self.low}, {self.high})"

    def get_rnd(self):
        return np.random.uniform(self.low, self.high)

    def get_linear(self, x: float):
        """
        Linearly interpolate between the bounds.

        :param x: The value to interpolate. It should be between 0 and 1.
        :type x: float
        :return: The interpolated value.
        """
        return self.low + (self.high - self.low) * x

    def to_linear(self, value: float) -> float:
        """
        Linearly interpolate between the bounds.

        :param value: The value to interpolate.
        :type value: float
        :return: The interpolated value.
        """
        return (value - self.low) / (self.high - self.low)

    def to_numeric(self, value: float) -> float:
        r"""Convert the value to a numeric representation.

        :param value: The value to convert to a numeric representation.
        :type value: float
        :return: The numeric representation of the value.
        """
        return value

    def from_numeric(self, value: float) -> float:
        r"""Convert the value to a numeric representation.

        :param value: The value to convert to a numeric representation.
        :type value: float
        :return: The numeric representation of the value.
        """
        return value


class Integer(Dimension):
    r"""Class representing an integer.

    :param name: The name of the integer.
    :type name: str
    :param low: The lower bound of the integer.
    :type low: int
    :param high: The upper bound of the integer.
    :type high: int
    """
    def __init__(self, name: str, low: int, high: int):
        super().__init__(name)
        self.low = low
        self.high = high

    def __repr__(self):
        return f"Integer({self.low}, {self.high})"

    def get_rnd(self):
        return np.random.randint(self.low, self.high)

    def get_linear(self, x: float):
        """
        Linearly interpolate between the bounds.

        :param x: The value to interpolate. It should be between 0 and 1.
        :type x: float
        :return: The interpolated value.
        """
        return int(self.low + (self.high - self.low) * x)

    def to_linear(self, value: int) -> float:
        """
        Linearly interpolate between the bounds.

        :param value: The value to interpolate.
        :type value: int
        :return: The interpolated value.
        """
        return (value - self.low) / (self.high - self.low)

    def to_numeric(self, value: int) -> float:
        r"""Convert the value to a numeric representation.

        :param value: The value to convert to a numeric representation.
        :type value: int
        :return: The numeric representation of the value.
        """
        return value

    def from_numeric(self, value: float) -> int:
        r"""Convert the value to a numeric representation.

        :param value: The value to convert to a numeric representation.
        :type value: float
        :return: The numeric representation of the value.
        """
        return int(value)


class Categorical(Dimension):
    r"""Class representing a categorical variable.

    :param name: The name of the categorical variable.
    :type name: str
    :param values: The possible values of the categorical variable.
    :type values: List[Any]
    """
    def __init__(self, name: str, values: list):
        super().__init__(name)
        self.values = values

    def __repr__(self):
        return f"Categorical({self.values})"

    def get_rnd(self):
        return np.random.choice(self.values)

    def get_linear(self, x: float):
        """
        Linearly interpolate between the bounds.

        :param x: The value to interpolate. It should be between 0 and 1.
        :type x: float
        :return: The interpolated value.
        """
        return self.values[int(x * (len(self.values) - 1))]

    def to_linear(self, value: Any) -> float:
        """
        Linearly interpolate between the bounds.

        :param value: The value to interpolate.
        :type value: Any
        :return: The interpolated value.
        """
        return self.values.index(value) / (len(self.values) - 1)

    def to_numeric(self, value: Any) -> float:
        r"""Convert the value to a numeric representation.

        :param value: The value to convert to a numeric representation.
        :type value: Any
        :return: The numeric representation of the value.
        """
        return self.values.index(value)

    def from_numeric(self, value: float) -> Any:
        r"""Convert the value to a numeric representation.

        :param value: The value to convert to a numeric representation.
        :type value: float
        :return: The numeric representation of the value.
        """
        return self.values[int(value)]


class SearchSpace:
    r"""Class representing a search space.

    :param space: The search space. It is a dictionary where the keys are the names of the parameters
        and the values are the values or the ranges of the parameters.
    :type space: Optional[Dict[str, Dimension]]

    """
    def __init__(
            self,
            *args,
            **additional_dimensions
    ):
        self._space: Dict[str, Dimension] = {}
        self._space.update(additional_dimensions)
        for dim in args:
            if isinstance(dim, Dimension):
                self._space[dim.name] = dim

        assert all(isinstance(dim, Dimension) for dim in self._space.values()), (
            "All values in the space should be of type Dimension."
        )

        self._keys = list(self._space.keys())
        self.reducers = {}

    @property
    def dimensions(self):
        return list(self._space.values())

    @property
    def keys(self):
        return self._keys

    def get_random_point(self) -> Dict[str, Any]:
        r"""Get a random point in the search space.

        :return: A random point in the search space.
        """
        point = {}
        for param, values in self._space.items():
            point[param] = values.get_rnd()
        return point

    def sample(self) -> Dict[str, Any]:
        r"""Get a random point in the search space.

        :return: A random point in the search space.
        """
        return self.get_random_point()

    def point_to_numeric(self, point: List[Any]) -> np.ndarray:
        r"""Convert the point to a numeric representation.

        :param point: The point to convert to a numeric representation.
        :type point: List[Any]
        :return: The numeric representation of the point.
        """
        return np.array([
            self._space[k].to_numeric(v)
            for k, v in zip(self.keys, point)
        ])

    def point_to_linear(self, point: List[Any]) -> np.ndarray:
        r"""Convert the point to a linear representation.

        :param point: The point to convert to a linear representation.
        :type point: List[Any]
        :return: The linear representation of the point.
        """
        return np.array([
            self._space[k].to_linear(v)
            for k, v in zip(self.keys, point)
        ])

    def point_from_linear(self, point: np.ndarray) -> List[Any]:
        r"""Convert the point to a linear representation.

        :param point: The point to convert to a linear representation.
        :type point: List[Any]
        :return: The linear representation of the point.
        """
        return [
            self._space[k].from_linear(v)
            for k, v in zip(self.keys, point)
        ]

    def points_to_numeric(self, points: List[List[Any]]) -> np.ndarray:
        r"""Convert the points to a numeric representation.

        :param points: The points to convert to a numeric representation.
        :type points: Dict[str, Any]
        :return: The numeric representation of the points.
        """
        return np.array([
            self.point_to_numeric(point)
            for point in points
        ])

    def point_from_numeric(self, point: np.ndarray) -> List[Any]:
        r"""Convert the point to a numeric representation.

        :param point: The point to convert to a numeric representation.
        :type point: List[Any]
        :return: The numeric representation of the point.
        """
        return [
            self._space[k].from_numeric(v)
            for k, v in zip(self.keys, point)
        ]

    def points_from_numeric(self, points: np.ndarray) -> List[List[Any]]:
        r"""Convert the points to a numeric representation.

        :param points: The points to convert to a numeric representation.
        :type points: Dict[str, Any]
        :return: The numeric representation of the points.
        """
        return [
            self.point_from_numeric(point)
            for point in points
        ]

    def points_to_linear(self, points: List[List[Any]]) -> np.ndarray:
        r"""Convert the points to a linear representation.

        :param points: The points to convert to a linear representation.
        :type points: np.ndarray
        :return: The linear representation of the points.
        """
        return np.stack([self.point_to_linear(point) for point in points], axis=0)

    def points_from_linear(self, points: List[List[float]]) -> List[List[Any]]:
        r"""Convert the points to a linear representation.

        :param points: The points to convert to a linear representation.
        :type points: List[List[float]]
        :return: The linear representation of the points.
        """
        return [self.point_from_linear(point) for point in points]

    def fit_reducer(self, x: np.ndarray, k: int = 1):
        reducer = self.get_reducer(k)
        reducer.fit(x)
        return reducer

    def get_reducer(self, k: int = 1):
        import umap
        if k == 1:
            reducer = self.reducers.get(k, UMAP_PCA_1d_REDUCER())
        else:
            reducer = self.reducers.get(k, umap.UMAP(n_components=k))
        self.reducers[k] = reducer
        return reducer

    def reducer_transform(self, x: np.ndarray, k: int = 1):
        reducer = self.get_reducer(k)
        return reducer.transform(x)

    def points_to_kd_linear(
            self,
            points: List[Any],
            k: int = 1,
            skip_if_k_eq_dim: bool = True,
            fit_reducer: bool = False,
    ) -> List[float]:
        r"""Convert the points to a linear representation of a k-dimensional space.
        The dimensionality of the initial space is reduced using UMAP.

        :param points: The points to convert to a linear representation.
        :type points: List[Any]
        :param k: The number of dimensions to use.
        :type k: int
        :return: The linear representation of the points.
        """
        initial_space = np.asarray(self.points_to_linear(points))
        if skip_if_k_eq_dim and k == len(self.dimensions):
            return initial_space
        if fit_reducer:
            self.fit_reducer(initial_space, k)
        return self.reducer_transform(initial_space, k)

