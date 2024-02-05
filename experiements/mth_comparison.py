import argparse
import os
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

try:
    import hps
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    import hps
from classical_pipeline import MlpPipeline
from tools import MPL_RC_DEFAULT_PARAMS

plt.rcParams.update(MPL_RC_DEFAULT_PARAMS)


def qgpr_main(
        x_train, x_test, y_train, y_test,
        ml_pipeline_cls,
        fig: Optional[plt.Figure] = None,
        search_ax: Optional[plt.Axes] = None,
        history_ax: Optional[plt.Axes] = None,
        n_trials=30,
        fig_kwargs=None,
        **kwargs
):
    fig_kwargs = fig_kwargs or {}
    figure_path = os.path.join(os.path.dirname(__file__), "figures", "qgpr")
    sa = hps.QGPRSearchAlgorithm(
        sigma_noise=1e-6,
        xi=0.01,
        # ei_gif_folder=os.path.join(figure_path, "qei_gif"),
        **kwargs
    )
    save_path = os.path.join(figure_path, "qgpr_pipeline.pkl")
    pipeline = hps.HpSearchPipeline.from_pickle_or_new(
        path=save_path,
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=ml_pipeline_cls,
        search_algorithm=sa,
        search_space=ml_pipeline_cls.search_space,
        n_trials=n_trials,
    )
    out = pipeline.run(desc="QGPR Hyperparameters search", save_path=save_path)
    # sa.make_gif()
    # cfig, _ = pipeline.plot_score_history(show=False, filename=os.path.join(figure_path, "qgp_history.pdf"))
    # plt.close(cfig)
    # pipeline.plot_hyperparameters_search(show=True)
    print(f"QGPR Best hyperparameters: {out.best_hyperparameters}")
    # cfig, _ = sa.plot_expected_improvement(show=False, filename=os.path.join(figure_path, "qei.pdf"))
    # plt.close(cfig)
    sa.plot_search(show=False, fig=fig, ax=search_ax, **fig_kwargs)
    pipeline.plot_score_history(show=False, fig=fig, ax=history_ax, **fig_kwargs)
    return out


def qsvr_main(
        x_train, x_test, y_train, y_test,
        ml_pipeline_cls,
        fig: Optional[plt.Figure] = None,
        search_ax: Optional[plt.Axes] = None,
        history_ax: Optional[plt.Axes] = None,
        n_trials=30,
        fig_kwargs=None,
        **kwargs
):
    fig_kwargs = fig_kwargs or {}
    figure_path = os.path.join(os.path.dirname(__file__), "figures", "qsvr")
    sa = hps.QSVRSearchAlgorithm(**kwargs)
    save_path = os.path.join(figure_path, "qsvr_pipeline.pkl")
    pipeline = hps.HpSearchPipeline.from_pickle_or_new(
        path=save_path,
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=ml_pipeline_cls,
        search_algorithm=sa,
        search_space=ml_pipeline_cls.search_space,
        n_trials=n_trials,
    )
    out = pipeline.run(desc="QSVR Hyperparameters search", save_path=save_path)
    print(f"QSVR Best hyperparameters: {out.best_hyperparameters}")
    # cfig, _ = pipeline.plot_score_history(show=False, filename=os.path.join(figure_path, "qsvr_history.pdf"))
    # plt.close(cfig)
    # cfig, _ = sa.plot_search(show=False, filename=os.path.join(figure_path, "qsvr_search.pdf"))
    # plt.close(cfig)
    pipeline.plot_score_history(show=False, fig=fig, ax=history_ax, **fig_kwargs)
    sa.plot_search(show=False, fig=fig, ax=search_ax, **fig_kwargs)
    return out


def gpr_main(
        x_train, x_test, y_train, y_test,
        ml_pipeline_cls,
        fig: Optional[plt.Figure] = None,
        search_ax: Optional[plt.Axes] = None,
        history_ax: Optional[plt.Axes] = None,
        n_trials=30,
        fig_kwargs=None,
        **kwargs
):
    fig_kwargs = fig_kwargs or {}
    figure_path = os.path.join(os.path.dirname(__file__), "figures", "gpr")
    sa = hps.GPRSearchAlgorithm(
        sigma_noise=1e-6,
        xi=0.01,
        # ei_gif_folder=os.path.join(figure_path, "ei_gif"),
        **kwargs
    )
    save_path = os.path.join(figure_path, "gpr_pipeline.pkl")
    pipeline = hps.HpSearchPipeline.from_pickle_or_new(
        path=save_path,
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=ml_pipeline_cls,
        search_algorithm=sa,
        search_space=ml_pipeline_cls.search_space,
        n_trials=n_trials,
    )
    out = pipeline.run(desc="GPR Hyperparameters search", save_path=save_path)
    # sa.make_gif()
    # cfig, _ = pipeline.plot_score_history(show=False, filename=os.path.join(figure_path, "gp_history.pdf"))
    # plt.close(cfig)
    # pipeline.plot_hyperparameters_search(show=True)
    print(f"GPR Best hyperparameters: {out.best_hyperparameters}")
    # cfig, _ = sa.plot_expected_improvement(show=False, filename=os.path.join(figure_path, "ei.pdf"))
    # plt.close(cfig)
    sa.plot_search(show=False, fig=fig, ax=search_ax, **fig_kwargs)
    pipeline.plot_score_history(show=False, fig=fig, ax=history_ax, **fig_kwargs)
    return out


def svr_main(
        x_train, x_test, y_train, y_test,
        ml_pipeline_cls,
        fig: Optional[plt.Figure] = None,
        search_ax: Optional[plt.Axes] = None,
        history_ax: Optional[plt.Axes] = None,
        n_trials=30,
        fig_kwargs=None,
        **kwargs
):
    fig_kwargs = fig_kwargs or {}
    figure_path = os.path.join(os.path.dirname(__file__), "figures", "svr")
    sa = hps.SVRSearchAlgorithm(**kwargs)
    save_path = os.path.join(figure_path, "svr_pipeline.pkl")
    pipeline = hps.HpSearchPipeline.from_pickle_or_new(
        path=save_path,
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=ml_pipeline_cls,
        search_algorithm=sa,
        search_space=ml_pipeline_cls.search_space,
        n_trials=n_trials,
    )
    out = pipeline.run(desc="SVR Hyperparameters search", save_path=save_path)
    print(f"SVR Best hyperparameters: {out.best_hyperparameters}")
    # cfig, _ = pipeline.plot_score_history(show=False, filename=os.path.join(figure_path, "svr_history.pdf"))
    # plt.close(cfig)
    # cfig, _ = sa.plot_search(show=False, filename=os.path.join(figure_path, "svr_search.pdf"))
    # plt.close(cfig)
    pipeline.plot_score_history(show=False, fig=fig, ax=history_ax, **fig_kwargs)
    sa.plot_search(show=False, fig=fig, ax=search_ax, **fig_kwargs)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--warmup_trials", type=int, default=2)
    parser.add_argument("--space_quantization", type=int, default=300)
    return parser.parse_args()


def get_and_plot_dataset():
    import umap
    figure_path = os.path.join(os.path.dirname(__file__), "figures", "dataset")
    x, y, *_ = datasets.make_classification(
        n_classes=2, n_features=12, n_redundant=2, n_informative=4, random_state=42, n_clusters_per_class=2
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_reducer = umap.UMAP(n_components=2)
    x_reduced = x_reducer.fit_transform(x)
    # x_train_reduced, x_test_reduced = x_reducer.transform(x_train), x_reducer.transform(x_test)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for c in np.unique(y):
        mask = y == c
        ax.scatter(x_reduced[mask, 0], x_reduced[mask, 1], label=f"Class {c}")
    ax.legend()
    ax.set_title("Dataset")
    os.makedirs(figure_path, exist_ok=True)
    plt.savefig(os.path.join(figure_path, "dataset.pdf"), bbox_inches="tight", dpi=900)
    # plt.close(fig)
    return x_train, x_test, y_train, y_test


def main():
    args = parse_args()
    np.random.seed(42)
    x_train, x_test, y_train, y_test = get_and_plot_dataset()
    mth_to_main_func = {
        "QSVR": qsvr_main,
        "QGPR": qgpr_main,
        "GPR": gpr_main,
        "SVR": svr_main,
    }
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mth_to_color = {mth: color for mth, color in zip(mth_to_main_func.keys(), colors)}

    fig, axes = plt.subplots(1, 2, figsize=(22, 12), sharey="all")
    for mth, main_func in mth_to_main_func.items():
        main_func(
            x_train, x_test, y_train, y_test,
            ml_pipeline_cls=MlpPipeline,
            fig=fig, search_ax=axes[0], history_ax=axes[1],
            fig_kwargs=dict(color=mth_to_color[mth]),
            n_trials=args.n_trials,
            warmup_trials=args.warmup_trials,
        )
    axes[0].set_title("HP Search")
    axes[1].set_title("Score history")
    axes[1].set_ylabel("")
    for ax in axes:
        ax.set_ylim(0, 1)

    # remove legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    # Make the legend manually in the rightmost plot by creating the patches
    patches = [plt.Line2D([0], [0], color=color, label=label) for label, color in mth_to_color.items()]
    labels = [h.get_label() for h in patches]
    fig.legend(patches, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "figures", f"{'_'.join(mth_to_main_func.keys())}.pdf"),
        bbox_inches="tight", dpi=900
    )
    plt.show()


if __name__ == '__main__':
    main()
