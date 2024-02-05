import argparse
import os
from typing import Optional
from collections import OrderedDict

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
        violin_ax: Optional[plt.Axes] = None,
        n_trials=30,
        fig_kwargs=None,
        **kwargs
):
    fig_kwargs = fig_kwargs or {}
    save_dirname = kwargs.get("save_dirname", "figures")
    figure_path = os.path.join(os.path.dirname(__file__), save_dirname, "qgpr")
    save_path = os.path.join(figure_path, "qgpr_pipeline.pkl")
    pipeline = hps.HpSearchPipeline.from_pickle_or_new(
        path=save_path,
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=ml_pipeline_cls,
        search_algorithm=hps.QGPRSearchAlgorithm(
            search_space=ml_pipeline_cls.search_space,
            sigma_noise=1e-6,
            xi=0.01,
            # ei_gif_folder=os.path.join(figure_path, "qei_gif"),
            **kwargs
        ),
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
    if history_ax is not None:
        pipeline.plot_score_history(show=False, fig=fig, ax=history_ax, **fig_kwargs)
    if search_ax is not None:
        pipeline.search_algorithm.plot_search(show=False, fig=fig, ax=search_ax, **fig_kwargs)
    if violin_ax is not None:
        pipeline.search_algorithm.plot_violin_search(show=False, fig=fig, ax=violin_ax, **fig_kwargs)
    return out


def qsvr_main(
        x_train, x_test, y_train, y_test,
        ml_pipeline_cls,
        fig: Optional[plt.Figure] = None,
        search_ax: Optional[plt.Axes] = None,
        history_ax: Optional[plt.Axes] = None,
        violin_ax: Optional[plt.Axes] = None,
        n_trials=30,
        fig_kwargs=None,
        **kwargs
):
    fig_kwargs = fig_kwargs or {}
    save_dirname = kwargs.get("save_dirname", "figures")
    figure_path = os.path.join(os.path.dirname(__file__), save_dirname, "qsvr")
    save_path = os.path.join(figure_path, "qsvr_pipeline.pkl")
    pipeline = hps.HpSearchPipeline.from_pickle_or_new(
        path=save_path,
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=ml_pipeline_cls,
        search_algorithm=hps.QSVRSearchAlgorithm(search_space=ml_pipeline_cls.search_space, **kwargs),
        search_space=ml_pipeline_cls.search_space,
        n_trials=n_trials,
    )
    out = pipeline.run(desc="QSVR Hyperparameters search", save_path=save_path)
    print(f"QSVR Best hyperparameters: {out.best_hyperparameters}")
    # cfig, _ = pipeline.plot_score_history(show=False, filename=os.path.join(figure_path, "qsvr_history.pdf"))
    # plt.close(cfig)
    # cfig, _ = sa.plot_search(show=False, filename=os.path.join(figure_path, "qsvr_search.pdf"))
    # plt.close(cfig)
    if history_ax is not None:
        pipeline.plot_score_history(show=False, fig=fig, ax=history_ax, **fig_kwargs)
    if search_ax is not None:
        pipeline.search_algorithm.plot_search(show=False, fig=fig, ax=search_ax, **fig_kwargs)
    if violin_ax is not None:
        pipeline.search_algorithm.plot_violin_search(show=False, fig=fig, ax=violin_ax, **fig_kwargs)
    return out


def gpr_main(
        x_train, x_test, y_train, y_test,
        ml_pipeline_cls,
        fig: Optional[plt.Figure] = None,
        search_ax: Optional[plt.Axes] = None,
        history_ax: Optional[plt.Axes] = None,
        violin_ax: Optional[plt.Axes] = None,
        n_trials=30,
        fig_kwargs=None,
        **kwargs
):
    fig_kwargs = fig_kwargs or {}
    save_dirname = kwargs.get("save_dirname", "figures")
    figure_path = os.path.join(os.path.dirname(__file__), save_dirname, "gpr")
    save_path = os.path.join(figure_path, "gpr_pipeline.pkl")
    pipeline = hps.HpSearchPipeline.from_pickle_or_new(
        path=save_path,
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=ml_pipeline_cls,
        search_algorithm=hps.GPRSearchAlgorithm(
            search_space=ml_pipeline_cls.search_space,
            sigma_noise=1e-6,
            xi=0.01,
            # ei_gif_folder=os.path.join(figure_path, "ei_gif"),
            **kwargs
        ),
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
    if history_ax is not None:
        pipeline.plot_score_history(show=False, fig=fig, ax=history_ax, **fig_kwargs)
    if search_ax is not None:
        pipeline.search_algorithm.plot_search(show=False, fig=fig, ax=search_ax, **fig_kwargs)
    if violin_ax is not None:
        pipeline.search_algorithm.plot_violin_search(show=False, fig=fig, ax=violin_ax, **fig_kwargs)
    return out


def svr_main(
        x_train, x_test, y_train, y_test,
        ml_pipeline_cls,
        fig: Optional[plt.Figure] = None,
        search_ax: Optional[plt.Axes] = None,
        history_ax: Optional[plt.Axes] = None,
        violin_ax: Optional[plt.Axes] = None,
        n_trials=30,
        fig_kwargs=None,
        **kwargs
):
    fig_kwargs = fig_kwargs or {}
    save_dirname = kwargs.get("save_dirname", "figures")
    figure_path = os.path.join(os.path.dirname(__file__), save_dirname, "svr")
    save_path = os.path.join(figure_path, "svr_pipeline.pkl")
    pipeline = hps.HpSearchPipeline.from_pickle_or_new(
        path=save_path,
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=ml_pipeline_cls,
        search_algorithm=hps.SVRSearchAlgorithm(search_space=ml_pipeline_cls.search_space, **kwargs),
        search_space=ml_pipeline_cls.search_space,
        n_trials=n_trials,
    )
    out = pipeline.run(desc="SVR Hyperparameters search", save_path=save_path)
    print(f"SVR Best hyperparameters: {out.best_hyperparameters}")
    # cfig, _ = pipeline.plot_score_history(show=False, filename=os.path.join(figure_path, "svr_history.pdf"))
    # plt.close(cfig)
    # cfig, _ = sa.plot_search(show=False, filename=os.path.join(figure_path, "svr_search.pdf"))
    # plt.close(cfig)
    if history_ax is not None:
        pipeline.plot_score_history(show=False, fig=fig, ax=history_ax, **fig_kwargs)
    if search_ax is not None:
        pipeline.search_algorithm.plot_search(show=False, fig=fig, ax=search_ax, **fig_kwargs)
    if violin_ax is not None:
        pipeline.search_algorithm.plot_violin_search(show=False, fig=fig, ax=violin_ax, **fig_kwargs)
    return out


def random_search_main(
        x_train, x_test, y_train, y_test,
        ml_pipeline_cls,
        fig: Optional[plt.Figure] = None,
        search_ax: Optional[plt.Axes] = None,
        history_ax: Optional[plt.Axes] = None,
        violin_ax: Optional[plt.Axes] = None,
        n_trials=30,
        fig_kwargs=None,
        **kwargs
):
    fig_kwargs = fig_kwargs or {}
    save_dirname = kwargs.get("save_dirname", "figures")
    figure_path = os.path.join(os.path.dirname(__file__), save_dirname, "random_search")
    save_path = os.path.join(figure_path, f"{kwargs.get('pipeline_name', 'random_search_pipeline')}.pkl")
    pipeline = hps.HpSearchPipeline.from_pickle_or_new(
        path=save_path,
        dataset=(x_train, y_train),
        test_dataset=(x_test, y_test),
        ml_pipeline_cls=ml_pipeline_cls,
        search_algorithm=hps.RandomSearchAlgorithm(search_space=ml_pipeline_cls.search_space, **kwargs),
        search_space=ml_pipeline_cls.search_space,
        n_trials=n_trials,
    )
    out = pipeline.run(desc="Random Search Hyperparameters search", save_path=save_path)
    print(f"Random Search Best hyperparameters: {out.best_hyperparameters}")
    # cfig, _ = pipeline.plot_score_history(show=False, filename=os.path.join(figure_path, "random_search_history.pdf"))
    # plt.close(cfig)
    # cfig, _ = sa.plot_search(show=False, filename=os.path.join(figure_path, "random_search_search.pdf"))
    # plt.close(cfig)
    if history_ax is not None:
        pipeline.plot_score_history(show=False, fig=fig, ax=history_ax, **fig_kwargs)
    if search_ax is not None:
        pipeline.search_algorithm.plot_search(show=False, fig=fig, ax=search_ax, **fig_kwargs)
    if violin_ax is not None:
        pipeline.search_algorithm.plot_violin_search(show=False, fig=fig, ax=violin_ax, **fig_kwargs)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--warmup_trials", type=int, default=0)
    parser.add_argument("--warmup_rn_history", type=int, default=2)
    parser.add_argument("--space_quantization", type=int, default=1000)
    parser.add_argument("--save_dirname", type=str, default="figures")
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
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    os.makedirs(figure_path, exist_ok=True)
    plt.savefig(os.path.join(figure_path, "dataset.svg"), bbox_inches="tight", dpi=900)
    # plt.close(fig)
    return x_train, x_test, y_train, y_test


def plot_violin_plot(
        ml_pipeline_cls,
        mth_to_main_func,
        mth_to_out,
        mth_to_color,
        args,
):
    fig, axes = plt.subplots(1, 2, figsize=(22, 12), sharey="all")
    # on a ax plot the points distribution in a linear space as a violinplot
    x_dist_ax_idx = 0
    xy_map = {mth: out.search_algorithm.make_x_y_from_history() for mth, out in mth_to_out.items()}
    linear_x = np.stack(
        [np.linspace(0, 1, args.space_quantization)] * len(ml_pipeline_cls.search_space.dimensions),
        axis=1
    )
    ml_pipeline_cls.search_space.fit_reducer(linear_x, k=1, if_not_fitted=True)
    x_1d_map = {mth: np.ravel(ml_pipeline_cls.search_space.reducer_transform(xy[0], k=1)) for mth, xy in xy_map.items()}
    x_1d_map_values = [x_1d_map[mth] for mth in mth_to_out.keys()]
    axes[x_dist_ax_idx].violinplot(x_1d_map_values, showmeans=True, showmedians=False, vert=False)
    # axes[x_dist_ax_idx].set_yticks(range(1, len(mth_to_out) + 1))
    # axes[x_dist_ax_idx].set_yticklabels(mth_to_out.keys())
    axes[x_dist_ax_idx].set_xlabel("Reduced Hyperparameters Space [-]")
    axes[x_dist_ax_idx].set_title("Hyperparameters Distributions")

    # on the axes[0] put a violin plot of the scores for each method
    score_dist_ax_idx = 1
    scores = OrderedDict({mth: [float(point.value) for point in out.history] for mth, out in mth_to_out.items()})
    scores_values = [scores[mth] for mth in mth_to_out.keys()]
    axes[score_dist_ax_idx].violinplot(scores_values, showmeans=True, showmedians=False, vert=False)
    axes[score_dist_ax_idx].set_yticks(range(1, len(scores) + 1))
    axes[score_dist_ax_idx].set_yticklabels(scores.keys())
    axes[score_dist_ax_idx].set_xlabel("Score [-]")
    axes[score_dist_ax_idx].set_title("Scores Distributions")
    axes[score_dist_ax_idx].set_xlim(max(0, np.min(np.asarray(scores_values)) - 0.05), 1)

    # axes[0].set_title("HP Search History")
    # axes[1].set_title("Mean Score History")
    # axes[1].set_ylabel("")
    # for ax in axes:
    #     ax.set_ylim(0, 1)

    # remove legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    # Make the legend manually in the rightmost plot by creating the patches
    # patches = [plt.Line2D([0], [0], color=color, label=label) for label, color in mth_to_color.items()]
    # labels = [h.get_label() for h in patches]
    # fig.legend(patches, labels, loc='upper center')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__), args.save_dirname, f"violin_{'_'.join(mth_to_main_func.keys())}.svg"
        ),
        bbox_inches="tight", dpi=900
    )
    # plt.show()


def plot_dists_over_itr(
        ml_pipeline_cls,
        mth_to_main_func,
        mth_to_out,
        mth_to_color,
        args,
):
    n_running_pts = 6
    half_window = n_running_pts // 2
    fig, axes = plt.subplots(1, 2, figsize=(22, 12), sharex="all")
    # on a ax plot the points distribution in a linear space over the iterations
    x_dist_ax_idx = 0
    xy_map = {mth: out.search_algorithm.make_x_y_from_history() for mth, out in mth_to_out.items()}
    linear_x = np.stack(
        [np.linspace(0, 1, args.space_quantization)] * len(ml_pipeline_cls.search_space.dimensions),
        axis=1
    )
    ml_pipeline_cls.search_space.fit_reducer(linear_x, k=1, if_not_fitted=True)
    x_1d_map = {
        mth: np.ravel(ml_pipeline_cls.search_space.reducer_transform(xy[0], k=1))
        for mth, xy in xy_map.items()
    }
    # compute the cumulative values of the 1d map
    running_mean_x_1d_map = {
        mth: [
            np.mean(x_1d_map[mth][max(0, i-half_window):min(len(x_1d_map[mth]), i+half_window)])
            for i in range(1, len(x_1d_map[mth]) + 1)
        ]
        for mth in mth_to_out.keys()
    }
    running_std_x_1d_map = {
        mth: [
            np.std(x_1d_map[mth][max(0, i-half_window):min(len(x_1d_map[mth]), i+half_window)])
            for i in range(1, len(x_1d_map[mth]) + 1)
        ]
        for mth in mth_to_out.keys()
    }
    for mth, running_mean_x_1d in running_mean_x_1d_map.items():
        axes[x_dist_ax_idx].plot(running_mean_x_1d, label=mth, color=mth_to_color[mth])
        axes[x_dist_ax_idx].fill_between(
            range(len(running_mean_x_1d)),
            np.asarray(running_mean_x_1d) - np.asarray(running_std_x_1d_map[mth]),
            np.asarray(running_mean_x_1d) + np.asarray(running_std_x_1d_map[mth]),
            alpha=0.1, color=mth_to_color[mth]
        )
    axes[x_dist_ax_idx].set_ylabel("Running Reduced Hyperparameters Space [-]")
    axes[x_dist_ax_idx].set_xlabel("Iteration [-]")
    axes[x_dist_ax_idx].set_title("Hyperparameters Distributions")

    # on the axes[0] put a violin plot of the scores for each method
    score_dist_ax_idx = 1
    scores = OrderedDict({mth: [float(point.value) for point in out.history] for mth, out in mth_to_out.items()})
    running_mean_scores = {
        mth: [
            np.mean(scores[mth][max(0, i-half_window):min(len(scores[mth]), i+half_window)])
            for i in range(1, len(scores[mth]) + 1)
        ]
        for mth in mth_to_out.keys()
    }
    running_std_scores = {
        mth: [
            np.std(scores[mth][max(0, i-half_window):min(len(scores[mth]), i+half_window)])
            for i in range(1, len(scores[mth]) + 1)
        ]
        for mth in mth_to_out.keys()
    }
    for mth, running_mean_score in running_mean_scores.items():
        axes[score_dist_ax_idx].plot(running_mean_score, label=mth, color=mth_to_color[mth])
        axes[score_dist_ax_idx].fill_between(
            range(len(running_mean_score)),
            np.asarray(running_mean_score) - np.asarray(running_std_scores[mth]),
            np.asarray(running_mean_score) + np.asarray(running_std_scores[mth]),
            alpha=0.1, color=mth_to_color[mth]
        )
    axes[score_dist_ax_idx].set_ylabel("Running Score [-]")
    axes[score_dist_ax_idx].set_xlabel("Iteration [-]")
    axes[score_dist_ax_idx].set_title("Scores Distributions")

    # remove legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    # Make the legend manually in the rightmost plot by creating the patches
    patches = [plt.Line2D([0], [0], color=color, label=label) for label, color in mth_to_color.items()]
    labels = [h.get_label() for h in patches]
    fig.legend(patches, labels, loc='upper center')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__), args.save_dirname, f"dists_over_itr_{'_'.join(mth_to_main_func.keys())}.svg"
        ),
        bbox_inches="tight", dpi=900
    )
    # plt.show()


def main():
    args = parse_args()
    np.random.seed(42)
    x_train, x_test, y_train, y_test = get_and_plot_dataset()
    mth_to_main_func = OrderedDict({
        "QSVR": qsvr_main,
        "QGPR": qgpr_main,
        "GPR": gpr_main,
        "SVR": svr_main,
        "RandomSearch": random_search_main,
    })
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mth_to_color = {mth: color for mth, color in zip(mth_to_main_func.keys(), colors)}
    mth_to_out = {}
    ml_pipeline_cls = MlpPipeline
    if args.warmup_rn_history > 0:
        rn_out = random_search_main(
            x_train, x_test, y_train, y_test,
            ml_pipeline_cls=ml_pipeline_cls,
            n_trials=args.warmup_rn_history,
            warmup_trials=0,
            pipeline_name="rn_warmup",
            plot=False,
            save_dirname=args.save_dirname,
        )

    for mth, main_func in mth_to_main_func.items():
        out = main_func(
            x_train, x_test, y_train, y_test,
            ml_pipeline_cls=ml_pipeline_cls,
            # fig=fig,
            # search_ax=axes[0],
            # history_ax=axes[1],
            # violin_ax=axes[1],
            fig_kwargs=dict(
                color=mth_to_color[mth],
                y_cumsum=False, y_max_normalize=False, y_running_mean=True, y_std_coeff=0.1,
                plot_pred=False,
            ),
            n_trials=args.n_trials,
            warmup_trials=args.warmup_trials,
            warmup_history=rn_out.history,
            save_dirname=args.save_dirname,
        )
        mth_to_out[mth] = out

    plot_violin_plot(ml_pipeline_cls, mth_to_main_func, mth_to_out, mth_to_color, args)
    plot_dists_over_itr(ml_pipeline_cls, mth_to_main_func, mth_to_out, mth_to_color, args)
    plt.show()


if __name__ == '__main__':
    main()
