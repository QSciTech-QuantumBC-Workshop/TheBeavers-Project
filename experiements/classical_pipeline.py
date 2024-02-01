from typing import Tuple

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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


if __name__ == '__main__':
    x, y = datasets.load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline = hps.HpSearchPipeline(
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=MlpPipeline,
        search_algorithm=hps.GPSearchAlgorithm(),
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

