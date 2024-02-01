from typing import Tuple, Optional

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
import hps


class MlpPipeline(hps.MLPipeline):
    def __init__(self, dataset: Tuple[np.ndarray, np.ndarray], hyperparameters: dict, **config):
        super().__init__(dataset, hyperparameters, **config)
        self.model = None

    def run(self, **kwargs) -> hps.PipelineRunOutput:
        x, y = self.dataset
        self.model = MLPClassifier(**self.hyperparameters)
        self.model.fit(x, y)
        return hps.PipelineRunOutput(
            model=self.model,
            dataset=self.dataset
        )

    def get_score(self, test_x, text_y, **kwargs) -> float:
        return self.model.score(test_x, text_y)


class GPSearchAlgorithm(hps.SearchAlgorithm):
    def __init__(self, search_space: Optional[hps.SearchSpace] = None, **config):
        super().__init__(search_space, **config)
        self.model = GaussianProcessRegressor()

    def make_x_y_from_history(self) -> Tuple[np.ndarray, np.ndarray]:
        x = self.search_space.points_to_numeric([
            [point.point[key] for key in self.search_space.keys]
            for point in self.history
        ])
        y = np.array([point.value for point in self.history])
        return x, y

    def make_hp_from_x(self, x) -> dict:
        return {
            key: value
            for key, value in zip(self.search_space.keys, x)
        }

    def get_next_trial_point(self) -> hps.TrialPoint:
        if len(self.history) < 2:
            return hps.TrialPoint(
                point=self.search_space.sample()
            )
        x, y = self.make_x_y_from_history()
        self.model.fit(x, y)

        linear_space = np.linspace(0, 1, 100)
        linear_x = self.search_space.points_to_numeric([
            [dim.get_linear(v) for dim in self.search_space.dimensions]
            for v in linear_space
        ])
        y_pred, y_std = self.model.predict(linear_x, return_std=True)
        next_x = linear_x[np.argmax(y_pred)]
        next_xhp = self.search_space.point_from_numeric(next_x)
        next_hp = self.make_hp_from_x(next_xhp)
        return hps.TrialPoint(point=next_hp)

    def update(self, trial_point: hps.TrialPoint):
        self.history.append(trial_point)

    def get_best_point(self) -> hps.TrialPoint:
        best_point = max(self.history, key=lambda point: point.value)
        return best_point


if __name__ == '__main__':
    x, y = datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline = hps.HpSearchPipeline(
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=MlpPipeline,
        search_algorithm=GPSearchAlgorithm(),
        search_space=hps.SearchSpace(
            hps.Real("learning_rate_init", 1e-4, 1e-1),
            hps.Integer("hidden_layer_sizes", 10, 100),
            hps.Categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
        ),
        n_trials=20,
    )
    out = pipeline.run()
    pipeline.plot_score_history(show=True)
    pipeline.plot_hyperparameters_search(show=True)
    print(out.best_hyperparameters)

