from typing import Union, Tuple, Dict, Any, Optional

import numpy as np

from .base_pipeline import BasePipeline, PipelineRunOutput
from ..tools import to_json


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

    @classmethod
    def get_search_space(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_default_hyperparameters(cls, **kwargs):
        raise NotImplementedError

    def __init__(
            self,
            dataset: Union[str, Tuple[np.ndarray, np.ndarray]],
            hyperparameters: Optional[Dict[str, Any]] = None,
            **config
    ):
        super().__init__(**config)
        self.dataset = dataset
        self.hyperparameters = hyperparameters or self.get_default_hyperparameters(**config)

    def run(self, **kwargs) -> PipelineRunOutput:
        raise NotImplementedError

    def get_score(self, test_x, text_y, **kwargs) -> float:
        raise NotImplementedError

    def __json__(self):
        return dict(
            dataset_shapes=(self.dataset[0].shape, self.dataset[1].shape),
            hyperparameters=to_json(self.hyperparameters),
            config=to_json(self.config)
        )


