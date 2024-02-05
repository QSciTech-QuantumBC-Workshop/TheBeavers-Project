from typing import Optional, Tuple
import os
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm
import matplotlib.pyplot as plt
from ..search_algorithm import SearchAlgorithm, TrialPoint
from ...search_space import SearchSpace


class GPRSearchAlgorithm(SearchAlgorithm):
    def __init__(self, search_space: Optional[SearchSpace] = None, **config):
        super().__init__(search_space, **config)
        self.model = GaussianProcessRegressor(
            kernel=ConstantKernel() * Matern(length_scale=1.0, nu=2.5)
        )
        self.warmup_trials = config.get('warmup_trials', 10)
        self.space_quantization = config.get('space_quantization', 100)
        self._fit = False
        self.ei_gif_folder = config.get('ei_gif_folder', None)
        self.xi = config.get('xi', 0.01)
        self.sigma_noise = config.get('sigma_noise', 1e-6)

    def get_next_trial_point(self) -> TrialPoint:
        if len(self.history) < self.warmup_trials:
            return TrialPoint(
                point=self.search_space.sample()
            )
        x, y = self.make_x_y_from_history()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model.fit(x, y)
        self._fit = True

        linear_space = np.linspace(0, 1, self.space_quantization)
        linear_x = np.stack([linear_space] * len(self.search_space.dimensions), axis=1)
        ei, mu, sigma = self.expected_improvement(linear_x, y)

        if self.ei_gif_folder is not None:
            fig, ax = plt.subplots()
            self.plot_expected_improvement(
                fig=fig,
                ax=ax,
                x=x,
                y=y,
                linear_x=linear_x,
                ei=ei,
                mu=mu,
                sigma=sigma,
                show=False,
                filename=f"{self.ei_gif_folder}/ei_{len(self.history)}.png",
            )
            plt.close(fig)

        next_x = linear_x[np.argmax(ei)]
        next_xhp = self.search_space.point_from_linear(next_x)
        next_hp = self.make_hp_from_x(next_xhp)
        return TrialPoint(point=next_hp, best_pred_value=np.max(mu))

    def expected_improvement(self, x, y):
        r"""Calculate the expected improvement.

        See https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
        for more information.

        :param x: The hyperparameters to calculate the expected improvement for.
        :type x: np.ndarray
        :param y: The scores of the hyperparameters.
        :type y: np.ndarray
        :return: The expected improvement, the mean and the standard deviation.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        mu, sigma = self.model.predict(x, return_std=True)
        best_y = np.max(y)
        noisy_sigma = sigma + self.sigma_noise
        z = (mu - best_y - self.xi) / noisy_sigma
        ei = (mu - best_y - self.xi) * norm.cdf(z) + noisy_sigma * norm.pdf(z)
        return ei, mu, sigma

    def get_best_point(self) -> TrialPoint:
        x, y = self.make_x_y_from_history()
        linear_space = np.linspace(0, 1, self.space_quantization)
        linear_x = np.stack([linear_space] * len(self.search_space.dimensions), axis=1)
        ei, mu, sigma = self.expected_improvement(linear_x, y)
        best_x = linear_x[np.argmax(mu)]
        best_xhp = self.search_space.point_from_linear(best_x)
        best_hp = self.make_hp_from_x(best_xhp)
        return TrialPoint(point=best_hp, value=np.max(y))

    def plot_expected_improvement(
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
        if not self._fit:
            self.model.fit(x, y)
            self._fit = True
        linear_x = kwargs.get('linear_x', None)
        if linear_x is None:
            linear_space = np.linspace(0, 1, self.space_quantization)
            linear_x = np.stack([linear_space] * len(self.search_space.dimensions), axis=1)
        ei, mu, sigma = kwargs.get('ei', None), kwargs.get('mu', None), kwargs.get('sigma', None)
        if ei is None or mu is None or sigma is None:
            ei, mu, sigma = self.expected_improvement(linear_x, y)
        self.search_space.fit_reducer(linear_x, k=1)
        linear_x_1d = np.ravel(self.search_space.reducer_transform(linear_x, k=1))

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.plot(linear_x_1d, mu, label='Mean', color=kwargs.get('color', 'blue'))
        ax.fill_between(linear_x_1d, mu - sigma, mu + sigma, alpha=0.1, color=kwargs.get('color', 'blue'))

        # add points from history as scatter
        x_1d = self.search_space.reducer_transform(x, k=1)
        ax.scatter(
            x_1d, y,
            label='History',
            color=kwargs.get('color', 'green'),
            marker='x'
        )

        # put expected improvement on a second y-axis
        if kwargs.get('show_ei', True):
            ax2 = ax.twinx()
            ax2.plot(linear_x_1d, ei, label='Expected Improvement', color=kwargs.get('color', 'red'), linestyle='--')
            ax2.legend()

        ax.legend()
        ax.set_xlabel('Hyperparameter Space [-]')
        ax.set_ylabel('Score [-]')

        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)

        if show:
            plt.show()
        return fig, ax

    def plot_search(
            self,
            *,
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            show: bool = True,
            filename: Optional[str] = None,
            **kwargs
    ):
        kwargs.setdefault("show_ei", False)
        return self.plot_expected_improvement(fig=fig, ax=ax, show=show, filename=filename, **kwargs)

    def make_gif(self):
        if self.ei_gif_folder is None:
            return None
        import imageio

        folder = self.ei_gif_folder
        output = f"{folder}/ei.gif"
        filenames = [f"{folder}/{f}" for f in os.listdir(folder) if f != "ei.gif"]
        filenames.sort()
        with imageio.get_writer(output, mode='I') as writer:
            for filename in filenames:
                filepath = os.path.join(folder, filename)
                image = imageio.imread(filepath)
                writer.append_data(image)
        return output
