import os
import numpy as np
from typing import Union, Tuple, Dict, Any, Type, Optional
from .base_pipeline import BasePipeline, PipelineRunOutput
from .ml_pipeline import MLPipeline
from ..search_algorithms.search_algorithm import SearchAlgorithm, TrialPoint
from ..search_space import SearchSpace
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from numbers import Number


class HpSearchPipeline(BasePipeline):
    r"""Base class for hyperparameter search pipelines.


    :param dataset: The dataset to use for training and testing the model.
        It can be a string representing the name of the dataset that will be loaded or a
        tuple containing the data points and the target.
    :type dataset: Union[str, Tuple[np.ndarray, np.ndarray]]
    :param ml_pipeline_cls: The machine learning pipeline class to use for the search.
    :type ml_pipeline_cls: Type[MLPipeline]
    :param search_space: The search space to use for the hyperparameter search.
    :type search_space: Optional[SearchSpace]
    :param search_algorithm: The search algorithm to use for the hyperparameter search.
    :type search_algorithm: SearchAlgorithm
    :param ml_pipeline_config: Additional configuration parameters for the machine learning pipeline.
    :type ml_pipeline_config: Optional[Dict[str, Any]]
    :param config: Additional configuration parameters.
    :type config: Any

    :ivar dataset: The dataset to use for training and testing the model.
    :ivar ml_pipeline_cls: The machine learning pipeline class to use for the search.
    :ivar ml_pipeline_config: Additional configuration parameters for the machine learning pipeline.
    :ivar search_space: The search space to use for the hyperparameter search.
    :ivar search_algorithm: The search algorithm to use for the hyperparameter search.

    """
    def __init__(
            self,
            dataset: Union[str, Tuple[np.ndarray, np.ndarray]],
            test_dataset: Union[str, Tuple[np.ndarray, np.ndarray]],
            ml_pipeline_cls: Type[MLPipeline],
            search_algorithm: SearchAlgorithm,
            *,
            ml_pipeline_config: Optional[Dict[str, Any]] = None,
            search_space: Optional[SearchSpace] = None,
            **config
    ):
        super().__init__(**config)
        self.dataset: Tuple[np.ndarray, np.ndarray] = dataset
        self.test_dataset = test_dataset
        self.ml_pipeline_cls: Type[MLPipeline] = ml_pipeline_cls
        self.ml_pipeline_config: Dict[str, Any] = ml_pipeline_config or {}
        self.search_space: SearchSpace = search_space
        self.search_algorithm = search_algorithm
        self.best_hyperparameters: Optional[TrialPoint] = None

    def maybe_load_dataset(self, dataset: Union[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(dataset, str):
            return self.load_dataset(dataset)
        return dataset

    def load_dataset(self, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def preprocess_dataset(self, dataset: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return dataset

    def make_ml_pipeline(self, hyperparameters: Dict[str, Any]) -> MLPipeline:
        return self.ml_pipeline_cls(
            dataset=self.dataset,
            hyperparameters=hyperparameters,
            **self.ml_pipeline_config,
        )

    def search_hyperparameters(self, **kwargs) -> TrialPoint:
        if self.search_space is not None:
            self.search_algorithm.set_search_space(self.search_space)
        n_trials = self.config.get('n_trials', 10)
        initial_length = len(self.search_algorithm.history)
        n_trials_todo = max(n_trials - initial_length, 0)
        p_bar = tqdm(range(n_trials_todo), desc=kwargs.get("desc", "Hyperparameter search"))
        for i in p_bar:
            trial_point = self.search_algorithm.get_next_trial_point()
            ml_pipeline = self.make_ml_pipeline(trial_point.point)
            ml_pipeline.run(dataset=self.dataset, i=i+initial_length, n_trials=n_trials)
            trial_point.value = ml_pipeline.get_score(*self.test_dataset)
            self.search_algorithm.update(trial_point)
            h_best_score = self.search_algorithm.history.get_best_point().value
            p_bar.set_postfix(best_score=h_best_score, pred_best_score=trial_point.best_pred_value)
        return self.search_algorithm.get_best_point()

    def run(self, **kwargs) -> PipelineRunOutput:
        self.best_hyperparameters = self.search_hyperparameters(**kwargs)
        ml_pipeline = self.make_ml_pipeline(self.best_hyperparameters.point)
        if self.best_hyperparameters is None:
            ml_pipeline.run(dataset=self.dataset, **kwargs)
            self.best_hyperparameters.value = ml_pipeline.get_score(*self.test_dataset)
        save_path = kwargs.get("save_path", None)
        if save_path is not None:
            self.to_pickle(save_path)
        return PipelineRunOutput(
            best_ml_pipeline=ml_pipeline,
            best_hyperparameters=self.best_hyperparameters,
            history=self.search_algorithm.history,
            search_space=self.search_space,
            search_algorithm=self.search_algorithm,
        )

    def plot_score_history(
            self,
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        history = self.search_algorithm.history
        x = list(range(len(history)))
        y = [float(point.value) for point in history]
        y_std = None
        if kwargs.get("y_cumsum", False):
            y = np.cumsum(y)
        if kwargs.get("y_max_normalize", False):
            y = y / np.max(y)
        if kwargs.get("y_running_mean", False):
            n_mean_pts = kwargs.get("n_mean_pts", max(3, int(0.05 * len(y))))
            # pad the y array to avoid edge effects
            half_window = n_mean_pts // 2
            y = np.pad(y, (half_window, half_window), mode="edge")
            y = np.convolve(y, np.ones(n_mean_pts) / n_mean_pts, mode="same")
            y_std = kwargs.get("y_std_coeff", 1.0) * np.convolve(y, np.ones(n_mean_pts) / n_mean_pts, mode="same")
            y = y[half_window:-half_window]
            y_std = y_std[half_window:-half_window]
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=kwargs.get("figsize", (14, 10)))
        ax.plot(x, y, color=kwargs.get("color", "blue"))
        if y_std is not None:
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.1, color=kwargs.get("color", "blue"))
        ax.set_xlabel("Iteration [-]")
        ax.set_ylabel("Score [-]")
        filename = kwargs.get("filename", None)
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
        if kwargs.get("show", True):
            plt.show()
        return fig, ax

    def plot_hyperparameter_search(
            self,
            hp_name: str,
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        history = self.search_algorithm.history
        x = list(range(len(history)))
        y = [point.point[hp_name] for point in history]
        if not isinstance(y[0], Number):
            y = [str(e) for e in y]
            y_numeric_map = {str(e): i for i, e in enumerate(sorted(set(y)))}
            y_numeric_map_inv = {v: k for k, v in y_numeric_map.items()}
            y = [y_numeric_map[y] for y in y]
        else:
            y_numeric_map, y_numeric_map_inv = None, None
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=kwargs.get("figsize", (14, 10)))
        ax.plot(x, y)
        if y_numeric_map is not None:
            ax.set_yticks(list(y_numeric_map.values()))
            ax.set_yticklabels(list(y_numeric_map_inv.values()))
        if kwargs.get("add_score_points", True):
            scores = [point.value for point in history]
            ax.scatter(x, y, c=scores, cmap="viridis")
            cbar = plt.colorbar(ax.collections[0], ax=ax, orientation="vertical")
            cbar.set_label("Score [-]")
        ax.set_xlabel("Iteration [-]")
        ax.set_ylabel(hp_name)
        if kwargs.get("show", True):
            plt.show()
        return fig, ax

    def plot_hyperparameters_search(
            self,
            fig: Optional[plt.Figure] = None,
            axes: Optional[plt.Axes] = None,
            **kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        show = kwargs.pop("show", True)
        n_rows = int(np.sqrt(len(self.search_space.dimensions)))
        n_cols = int(np.ceil(len(self.search_space.dimensions) / n_rows))
        if fig is None or axes is None:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.ravel(np.asarray([axes]))
        for dim, ax in zip(self.search_space.dimensions, axes):
            self.plot_hyperparameter_search(dim.name, fig=fig, ax=ax, **kwargs)

        if show:
            plt.tight_layout()
            plt.show()
        return fig, axes

    def to_pickle(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path: str) -> "HpSearchPipeline":
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_pickle_or_new(cls, path: str, **kwargs) -> "HpSearchPipeline":
        if os.path.exists(path):
            print(f"Loading pipeline from {path}")
            return cls.from_pickle(path)
        return cls(**kwargs)

