import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from rliable import library as rly
from rliable import metrics, plot_utils
from tqdm.contrib.concurrent import process_map


def get_mean_smooth_plot(
    df,
    metric,
    x="train_step",
    smoothing_factor=0.2,
    running_average=10,
    use_ewm=False,
    alpha=0.2,
    color=None,
    label=None,
    ax=None,
    style="-",
    limit=None,
    lw=1,
):
    if limit is not None:
        df = df[df[x] <= limit]
    plot_df = df.groupby([x])[metric].aggregate(["mean", "sem"]).reset_index()
    if use_ewm:
        plot_df["mean"] = plot_df["mean"].transform(
            lambda x: x.ewm(alpha=smoothing_factor, adjust=False).mean()
        )
        plot_df["sem"] = plot_df["sem"].transform(
            lambda x: x.ewm(alpha=smoothing_factor, adjust=False).mean()
        )
    else:
        plot_df["mean"] = plot_df["mean"].transform(
            lambda x: x.rolling(
                window=running_average, min_periods=1, center=True
            ).mean()
        )
        plot_df["sem"] = plot_df["sem"].transform(
            lambda x: x.rolling(
                window=running_average, min_periods=1, center=True
            ).mean()
        )
    ax = sns.lineplot(
        plot_df,
        x=x,
        y="mean",
        color=color,
        ax=ax,
        linestyle=style,
        label=label,
        lw=lw,
    )
    ax = ax.fill_between(
        plot_df[x],
        plot_df["mean"] - plot_df["sem"],
        plot_df["mean"] + plot_df["sem"],
        color=color,
        alpha=alpha,
    )
    return ax


def sort_key_fn(k):
    (source, name), _ = k
    source_map = {"risc": 0, "baseline": 1}
    return source_map[source], name


def create_plot(
    df,
    method_to_color,
    source_to_style,
    source_to_linewidths,
    ykey="returns",
    xlabel="Training Steps",
    ylabel="Return",
    use_ewm=True,
    smoothing_factor=0.04,
    running_average=50,
    title=None,
    limit=None,
    log=False,
    ax=None,
    fig=None,
):
    if ax is None:
        fig, (ax) = plt.subplots(1, 1, figsize=(20 / 3.0, 5))
    grouped_df = df.groupby(["source", "method"])
    for name, group in sorted(grouped_df, key=sort_key_fn):
        get_mean_smooth_plot(
            group,
            ykey,
            color=method_to_color[name[1]],
            style=source_to_style[name[0]],
            label=name[1],
            use_ewm=use_ewm,
            smoothing_factor=smoothing_factor,
            running_average=running_average,
            alpha=0.2,
            ax=ax,
            limit=limit,
            lw=source_to_linewidths[name[0]],
        )
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.grid(color="black", linestyle=":", linewidth=0.7)
    if log:
        ax.set_ylim([-400, 0])
    if title is not None:
        plt.title(title)
    return fig, ax


def make_hline(ax, mean, std, color, style="-.", label=None, xlim=None):
    if xlim is None:
        xlim = ax.get_xlim()
    ax.hlines(mean, *xlim, colors=color, linestyles=style, label=label)
    ax.fill_between(
        xlim,
        (mean - std, mean - std),
        (mean + std, mean + std),
        color=color,
        alpha=0.2,
    )


def _decorate_axis(ax, wrect=10, hrect=10, ticklabelsize="large"):
    """Helper function for decorating plots."""
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))
    return ax


def plot_interval_estimates(
    point_estimates,
    interval_estimates,
    metric_names,
    algorithms=None,
    colors=None,
    color_palette="colorblind",
    max_ticks=4,
    subfigure_width=3.4,
    row_height=0.37,
    xlabel_y_coordinate=-0.1,
    xlabel="Normalized Score",
    **kwargs,
):
    """Plots various metrics with confidence intervals.

    Args:
      point_estimates: Dictionary mapping algorithm to a list or array of point
        estimates of the metrics to plot.
      interval_estimates: Dictionary mapping algorithms to interval estimates
        corresponding to the `point_estimates`. Typically, consists of stratified
        bootstrap CIs.
      metric_names: Names of the metrics corresponding to `point_estimates`.
      algorithms: List of methods used for plotting. If None, defaults to all the
        keys in `point_estimates`.
      colors: Maps each method to a color. If None, then this mapping is created
        based on `color_palette`.
      color_palette: `seaborn.color_palette` object for mapping each method to a
        color.
      max_ticks: Find nice tick locations with no more than `max_ticks`. Passed to
        `plt.MaxNLocator`.
      subfigure_width: Width of each subfigure.
      row_height: Height of each row in a subfigure.
      xlabel_y_coordinate: y-coordinate of the x-axis label.
      xlabel: Label for the x-axis.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      fig: A matplotlib Figure.
      axes: `axes.Axes` or array of Axes.
    """

    if algorithms is None:
        algorithms = point_estimates.keys()
    num_metrics = len(point_estimates[algorithms[0]])
    figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette))
    h = kwargs.pop("interval_height", 0.6)

    for idx, metric_name in enumerate(metric_names):
        for alg_idx, algorithm in enumerate(algorithms):
            ax = axes[idx] if num_metrics > 1 else axes
            # Plot interval estimates.
            lower, upper = interval_estimates[algorithm][:, idx]
            ax.barh(
                y=alg_idx,
                width=upper - lower,
                height=h,
                left=lower,
                color=colors[algorithm],
                #   alpha=0.75,
                label=algorithm,
            )
            # Plot point estimates.
            ax.vlines(
                x=point_estimates[algorithm][idx],
                ymin=alg_idx - (7.5 * h / 16),
                ymax=alg_idx + (6 * h / 16),
                label=algorithm,
                color="k",
                alpha=0.5,
            )

        ax.set_yticks(list(range(len(algorithms))))
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        if idx != 0:
            ax.set_yticks([])
        else:
            ax.set_yticklabels(algorithms, fontsize="x-large")
        ax.set_title(metric_name, fontsize="xx-large")
        ax.tick_params(axis="both", which="major")
        _decorate_axis(ax, ticklabelsize="xx-large", wrect=5)
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25)
    fig.text(0.4, xlabel_y_coordinate, xlabel, ha="center", fontsize="xx-large")
    plt.subplots_adjust(wspace=kwargs.pop("wspace", 0.11), left=0.0)
    return fig, axes


def rliable_data_generation(
    scores, mean, std, min_val, max_val, num_seeds=5, num_trials=1000
):
    generated_data = np.random.rand(num_trials, num_seeds) * std + mean
    # returns = df.sort_values(by=["seed", "environment"])['returns'].to_numpy().reshape(num_seeds, -1)
    best_mean = -np.inf
    best_scores = None
    best_data = None
    best_range = None

    for i in tqdm.trange(num_trials):
        new_data = generated_data[i]
        new_min = min(min_val, new_data.min())
        new_max = max(max_val, new_data.max())
        norm_data = (generated_data[i] - new_min) / (new_max - new_min)
        new_scores = np.concatenate([norm_data[..., None], scores], axis=1)
        aggregate_scores, _ = rly.get_interval_estimates(
            {"method": new_scores}, metrics.aggregate_mean, reps=2000
        )
        score = aggregate_scores["method"]

        if score > best_mean:
            best_mean = score
            best_scores = new_scores
            best_range = (new_min, new_max)
            best_data = new_data

    return best_scores, best_range, best_data


def create_normalized_df(df, overrides=None):
    max_return = df.groupby(["environment"])["returns"].max()
    min_return = df.groupby(["environment"])["returns"].min()
    normalized_df = df.copy()
    for env in df["environment"].unique():
        if overrides is not None and env in overrides:
            high = overrides[env][1]
            low = overrides[env][0]
        else:
            high = max_return[env]
            low = min_return[env]
        normalized_df.loc[normalized_df["environment"] == env, "norm_returns"] = (
            normalized_df.loc[normalized_df["environment"] == env, "returns"] - low
        ) / (high - low)
    return normalized_df


def generate_rliable_plot(
    score_dicts, metric_fns, method_to_color, algorithms, **kwargs
):
    aggregate_scores, aggregate_score_cis = get_rliable_metrics(score_dicts, metric_fns)
    fig, axes = plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=["IQM", "Mean"],
        algorithms=algorithms,
        xlabel="Normalized Score",
        colors=method_to_color,
        **kwargs,
    )
    return fig, axes


def get_rliable_metrics(score_dicts, metric_fns):
    aggregate_func = lambda x: np.array([fn(x) for fn in metric_fns])
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        score_dicts, aggregate_func, reps=2000
    )

    return aggregate_scores, aggregate_score_cis


def create_score_dicts(norm_df, key="norm_returns", num_to_average=5, num_seeds=5):
    score_dicts = {}
    score_df = (
        norm_df.groupby(["environment", "method", "seed"])[key]
        .apply(lambda x: x.iloc[-num_to_average:].mean())
        .reset_index()
    )
    for method in score_df["method"].unique():
        method_df = score_df[score_df["method"] == method]
        scores = method_df.sort_values(by=["seed", "environment"])[key].to_numpy()
        score_dicts[method] = scores.reshape(num_seeds, -1)
    return score_dicts


def get_rliable_scores(df, methods=None, num_to_average=5):
    normalized_df = create_normalized_df(df)
    score_dicts = create_score_dicts(normalized_df, num_to_average=num_to_average)
    if methods is not None:
        score_dicts = {k: v for k, v in score_dicts.items() if k in methods}
    return rly.get_interval_estimates(score_dicts, metrics.aggregate_mean, reps=2000)


def process_to_size(df, steps, interval):
    df = df[df["train_step"] <= steps]
    # Pad df with last element to make sure we have num_steps
    row = df.iloc[[-1]]
    num_missing = (steps - row["train_step"]) // interval
    new_rows = row.loc[row.index.repeat(num_missing)]
    new_rows["train_step"] = (
        float(row["train_step"]) + (np.arange(int(num_missing)) + 1) * interval
    )
    return pd.concat([df, new_rows])


def sample_efficiency_curve(full_df, num_steps, algorithms, limits, method_to_color):
    fig, ax = plt.subplots(figsize=(7, 5))
    ldf = (
        full_df.groupby(["environment", "method", "seed"])
        .apply(lambda x: process_to_size(x, limits[x["environment"].iloc[0]], 10000))
        .reset_index(drop=True)
    )
    norm_df = create_normalized_df(ldf)
    norm_df["progress"] = norm_df.apply(
        lambda x: (x["train_step"] * num_steps) / limits[x["environment"]], axis=1
    )

    norm_df = norm_df[norm_df["progress"].isin(list(range(1, num_steps + 1)))]
    score_dicts = {}
    for method in norm_df["method"].unique():
        #     print(method)
        if method not in algorithms:
            continue
        method_df = norm_df[norm_df["method"] == method]
        scores = method_df.sort_values(by=["seed", "environment", "progress"])[
            "norm_returns"
        ].to_numpy()
        score_dicts[method] = scores.reshape(5, -1, num_steps)
    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])]
    )
    iqm_scores, iqm_cis = rly.get_interval_estimates(score_dicts, iqm, reps=2000)
    ax = plot_utils.plot_sample_efficiency_curve(
        np.arange(1, num_steps + 1) / num_steps,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        xlabel=r"Fraction of Training Run",
        ylabel="IQM of Normalized Score",
        colors=method_to_color,
        ax=ax,
    )
    ax.legend()
    return fig


def run_trial(
    worker_id,
    df,
    mean,
    std,
    num_seeds,
    train_step,
    method,
    environment,
    num_to_average=5,
):
    rng = np.random.default_rng([worker_id])
    new_data = rng.normal(mean, std, num_seeds)
    added_df = pd.DataFrame(
        {
            "returns": new_data,
            "train_step": train_step,
            "method": method,
            "environment": environment,
            "source": "baselines",
            "seed": np.arange(num_seeds),
        }
    )
    new_df = pd.concat([df, added_df])
    scores, _ = get_rliable_scores(
        new_df, methods=[method], num_to_average=num_to_average
    )
    return new_df, scores


def add_generated_data(
    df, environment, method, mean, std, train_step, num_trials=1000, num_seeds=5
):
    best_score = -np.inf
    best_df = None

    # with Pool(8) as p:
    results = process_map(
        run_trial,
        range(num_trials),
        [df] * num_trials,
        [mean] * num_trials,
        [std] * num_trials,
        [num_seeds] * num_trials,
        [train_step] * num_trials,
        [method] * num_trials,
        [environment] * num_trials,
        chunksize=1,
    )

    for new_df, scores in results:
        if scores[method] > best_score:
            best_df = new_df
            best_score = scores[method]

    return best_df


def create_heatmaps(data):
    fig, axs = plt.subplots(2, 5, figsize=(25, 10), width_ratios=[1, 1, 1, 1, 1.1])

    for i in range(5):
        sns.heatmap(
            data["forward/qvals"][i * 4],
            vmax=1,
            linewidth=0.5,
            cmap="viridis",
            ax=axs[0, i],
            cbar=i == 4,
            yticklabels=False,
            xticklabels=False,
        )
        sns.heatmap(
            data["forward/local_visitation"][i * 4 + 1],
            vmax=25,
            linewidth=0.5,
            cmap="viridis",
            ax=axs[1, i],
            cbar=i == 4,
            yticklabels=False,
            xticklabels=False,
        )
        axs[1, i].set_xlabel(f"{(i)*4*2}k steps", fontsize=36)
    axs[0, 0].set_ylabel("Q-Values", fontsize=40)
    axs[1, 0].set_ylabel("Local Visitation", fontsize=40)
    plt.subplots_adjust(wspace=0.03, hspace=0.05)
    return fig
