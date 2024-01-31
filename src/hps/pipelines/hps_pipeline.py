import numpy as np
from typing import Union, Tuple, Dict, Any, Type, Optional
from .base_pipeline import BasePipeline, PipelineRunOutput
from .ml_pipeline import MLPipeline
from ..search_algorithms.search_algorithm import SearchAlgorithm, TrialPoint
from ..search_space import SearchSpace


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
            ml_pipeline_cls: Type[MLPipeline],
            search_algorithm: SearchAlgorithm,
            *,
            ml_pipeline_config: Optional[Dict[str, Any]] = None,
            search_space: Optional[SearchSpace] = None,
            **config
    ):
        super().__init__(**config)
        self.dataset = dataset
        self.ml_pipeline_cls = ml_pipeline_cls
        self.ml_pipeline_config = ml_pipeline_config or {}
        self.search_space = search_space
        self.search_algorithm = search_algorithm

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

    def search_hyperparameters(self) -> Dict[str, Any]:
        if self.search_space is not None:
            self.search_algorithm.set_search_space(self.search_space)
        n_trials = self.config.get('n_trials', 10)
        for _ in range(n_trials):
            trial_point = self.search_algorithm.get_next_trial_point()
            ml_pipeline = self.make_ml_pipeline(trial_point.point)
            ml_pipeline.run()
            trial_point.value = ml_pipeline.get_score()
            self.search_algorithm.update(trial_point)
        best_trial_point = max(self.search_algorithm.history, key=lambda tp: tp.value)
        return best_trial_point.point

    def run(self, **kwargs) -> PipelineRunOutput:
        dataset = self.maybe_load_dataset(self.dataset)
        dataset = self.preprocess_dataset(dataset)
        best_hyperparameters = self.search_hyperparameters(dataset)
        ml_pipeline = self.make_ml_pipeline(best_hyperparameters)
        return PipelineRunOutput(
            best_ml_pipeline=ml_pipeline,
            best_hyperparameters=best_hyperparameters
        )



