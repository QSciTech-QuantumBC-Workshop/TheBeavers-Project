from typing import Union, Tuple, Dict, Any

import numpy as np

from .base_pipeline import BasePipeline, PipelineRunOutput


class MLPipeline(BasePipeline):
    r"""Base class for machine learning pipelines.

    :param dataset: The dataset to use for training and testing the model.
        It can be a string representing the name of the dataset that will be loaded or a
        tuple containing the data points and the target.
    :type dataset: Union[str, Tuple[np.ndarray, np.ndarray]]
    :param hyperparameters: The hyperparameters to use for the model.
    :type hyperparameters: Dict[str, Any]
    :param config: Additional configuration parameters.
    :type config: Any

    :ivar dataset: The dataset to use for training and testing the model.
    :ivar hyperparameters: The hyperparameters to use for the model.
    :ivar config: Additional configuration parameters.
    """
    def __init__(
            self,
            dataset: Union[str, Tuple[np.ndarray, np.ndarray]],
            hyperparameters: Dict[str, Any],
            **config
    ):
        super().__init__(**config)
        self.dataset = dataset
        self.hyperparameters = hyperparameters

    def run(self, **kwargs) -> PipelineRunOutput:
        raise NotImplementedError

    def get_score(self) -> float:
        raise NotImplementedError



