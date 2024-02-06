from copy import deepcopy
from typing import Tuple, Optional
import os

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import hps
import warnings


class MlpPipeline(hps.MLPipeline):
    search_space = hps.SearchSpace(
        hps.Real("learning_rate_init", 1e-4, 1e-1),
        hps.Integer("hidden_layer_size", 8, 2048),
        hps.Integer("n_hidden_layer", 1, 12),
        hps.Integer("max_iter", 10, 300),
        hps.Categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
        hps.Categorical("activation", ['identity', 'logistic', 'tanh', 'relu']),
    )

    @classmethod
    def get_search_space(cls, **kwargs):
        dimensions_names = kwargs.get('dimensions', cls.search_space.keys)
        dimensions_names = dimensions_names or cls.search_space.keys
        dimensions = [cls.search_space.get_dimension(name) for name in dimensions_names]
        return hps.SearchSpace(*dimensions)

    @classmethod
    def get_default_hyperparameters(cls, **kwargs):
        dimensions_names = kwargs.get('dimensions', cls.search_space.keys)
        dimensions_names = dimensions_names or cls.search_space.keys
        default = {
            "learning_rate_init": 0.001,
            "hidden_layer_size": 100,
            "n_hidden_layer": 1,
            "max_iter": 200,
            "learning_rate": "constant",
            "activation": "relu",
        }
        return {k: default[k] for k in dimensions_names}

    def __init__(self, dataset: Tuple[np.ndarray, np.ndarray], hyperparameters: Optional[dict] = None, **config):
        super().__init__(dataset, hyperparameters, **config)
        self.model = None

    def run(self, **kwargs) -> hps.PipelineRunOutput:
        x, y = self.dataset
        hp = deepcopy(self.hyperparameters)
        hidden_layer_sizes = [hp.pop('hidden_layer_size', 512)] * hp.pop('n_hidden_layer', 1)
        hp['hidden_layer_sizes'] = hidden_layer_sizes
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = MLPClassifier(**hp)
            self.model.fit(x, y)
        return hps.PipelineRunOutput(
            model=self.model,
            dataset=self.dataset
        )

    def get_score(self, test_x, text_y, **kwargs) -> float:
        return self.model.score(test_x, text_y)


class KNNPipeline(hps.MLPipeline):
    search_space = hps.SearchSpace(
        hps.Integer("n_neighbors", 2, 100),
        hps.Real("p", 2, 100),
    )

    def __init__(self, dataset: Tuple[np.ndarray, np.ndarray], hyperparameters: dict, **config):
        super().__init__(dataset, hyperparameters, **config)
        self.model = None

    def run(self, **kwargs) -> hps.PipelineRunOutput:
        x, y = self.dataset
        hp = deepcopy(self.hyperparameters)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = KNeighborsClassifier(**hp)
            self.model.fit(x, y)
        return hps.PipelineRunOutput(
            model=self.model,
            dataset=self.dataset
        )

    def get_score(self, test_x, text_y, **kwargs) -> float:
        return self.model.score(test_x, text_y)


if __name__ == '__main__':
    # x, y = datasets.fetch_olivetti_faces(return_X_y=True)
    np.random.seed(42)
    x, y, *_ = datasets.make_classification(
        n_classes=4, n_features=10, n_redundant=2, n_informative=4, random_state=42, n_clusters_per_class=1
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    sa = hps.GPRSearchAlgorithm(
        warmup_trials=2,
        space_quantization=100,
        sigma_noise=1e-6,
        xi=0.01,
        ei_gif_folder=os.path.join(os.path.dirname(__file__), "ei_gif")
    )
    ml_pipeline_cls = KNNPipeline
    pipeline = hps.HpSearchPipeline(
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=ml_pipeline_cls,
        search_algorithm=sa,
        search_space=ml_pipeline_cls.search_space,
        n_trials=30,
    )
    out = pipeline.run()
    sa.make_gif()
    pipeline.plot_score_history(show=True)
    # pipeline.plot_hyperparameters_search(show=True)
    print(out.best_hyperparameters)
    sa.plot_expected_improvement(show=True, filename=os.path.join(os.path.dirname(__file__), "figures", "ei.pdf"))

