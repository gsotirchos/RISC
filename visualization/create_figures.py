import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_utils import (
    add_generated_data,
    create_heatmaps,
    create_normalized_df,
    create_plot,
    create_score_dicts,
    generate_rliable_plot,
    get_rliable_metrics,
    plot_interval_estimates,
    sample_efficiency_curve,
    make_hline,
    _decorate_axis,
)
from rliable import metrics

sns.set(font_scale=1.0)
sns.set_theme()
color_palette = sns.color_palette()
sns.set_style("whitegrid", rc={"axes.edgecolor": "black"})

method_to_color = {
    "RISC (ours)": color_palette[0],
    "R3L": color_palette[1],
    "Naive": color_palette[2],
    "FBRL": color_palette[3],
    "FBRL (ours)": color_palette[3],
    "Episodic": color_palette[4],
    "Episodic (ours)": color_palette[4],
    "VapRL": color_palette[5],
    "MEDAL": color_palette[6],
}
limits = {
    "Sawyer Door": 5_000_000,
    "Sawyer Peg": 7_000_000,
    "Tabletop Manipulation": 3_000_000,
    "Franka Kitchen": 5_000_000,
    "Minitaur": 5_000_000,
}


def create_figure_3_top(results_path: Path, output_path: Path):
    with open(results_path / "minigrid_rc.pkl", "rb") as f:
        rc_data = pickle.load(f)
    with open(results_path / "minigrid_risc.pkl", "rb") as f:
        risc_data = pickle.load(f)
    rc_fig = create_heatmaps(rc_data)
    risc_fig = create_heatmaps(risc_data)
    rc_fig.savefig(output_path / "figure_3_top_rc.pdf", bbox_inches="tight")
    risc_fig.savefig(output_path / "figure_3_top_risc.pdf", bbox_inches="tight")


def create_figure_3_bottom(results_path: Path, output_path: Path):
    data = pd.read_csv(results_path / "minigrid_optimal_pd.csv")
    source_to_style = {"risc": "-"}
    source_to_linewidths = {"risc": 2}
    method_to_color = {"RISC": color_palette[0], "Reverse Curriculum": color_palette[3]}
    fig, _ = create_plot(
        data,
        method_to_color,
        source_to_style,
        source_to_linewidths,
        "forward_agent/optimal_pd",
        "Optimal Q-Value\nPercent Difference",
        use_ewm=False,
    )
    fig.savefig(output_path / "figure_3_bottom.pdf", bbox_inches="tight")


def create_figure_4_left(results_path: Path, output_path: Path):
    baselines = pd.read_csv(results_path / "baselines.csv")
    risc_df = pd.read_csv(results_path / "risc.csv")
    full_df = pd.concat([baselines, risc_df])
    ldf = (
        full_df.groupby("environment")
        .apply(lambda x: x[x["train_step"] <= limits[x["environment"].iloc[0]]])
        .reset_index(drop=True)
    )

    # The sample data for this environment is missing. We generate 1000 5 seed samples
    # for each algorithm, and use the sample that results in the best rliable mean, to
    # give the benefit of the doubt to the baselines.
    augmented_df = add_generated_data(
        ldf, "Minitaur", "Episodic", -41.5, 3.4, limits["Minitaur"]
    )
    augmented_df = add_generated_data(
        augmented_df, "Minitaur", "Naive", -1041.10, 44.58, limits["Minitaur"]
    )
    augmented_df = add_generated_data(
        augmented_df, "Minitaur", "R3L", -186.30, 34.79, limits["Minitaur"]
    )
    augmented_df = add_generated_data(
        augmented_df, "Minitaur", "FBRL", -986.34, 67.95, limits["Minitaur"]
    )
    algorithms = list(
        reversed(["Episodic", "RISC (ours)", "MEDAL", "VapRL", "FBRL", "R3L", "Naive"])
    )
    norm_df = create_normalized_df(augmented_df)
    score_dicts = create_score_dicts(norm_df, num_to_average=5)
    fig, axes = generate_rliable_plot(
        score_dicts,
        [metrics.aggregate_iqm, metrics.aggregate_mean],
        method_to_color,
        algorithms,
        xlabel_y_coordinate=-0.2,
    )
    axes[0].grid(False, axis="y")
    fig.savefig(output_path / "figure_4_left.pdf", bbox_inches="tight")


def create_figure_4_right(results_path: Path, output_path: Path):
    baselines = pd.read_csv(results_path / "baselines.csv")
    risc_df = pd.read_csv(results_path / "risc.csv")
    full_df = pd.concat([baselines, risc_df])
    algorithms = ["Episodic", "RISC (ours)", "MEDAL", "VapRL", "FBRL", "R3L", "Naive"]

    fig = sample_efficiency_curve(full_df, 20, algorithms, limits, method_to_color)
    fig.savefig(output_path / "figure_4_right.pdf", bbox_inches="tight")


def create_figure_5(results_path: Path, output_path: Path):
    baselines = pd.read_csv(results_path / "baselines.csv")
    risc_df = pd.read_csv(results_path / "risc.csv")
    full_df = pd.concat([baselines, risc_df])
    ldf = (
        full_df.groupby("environment")
        .apply(lambda x: x[x["train_step"] <= limits[x["environment"].iloc[0]]])
        .reset_index(drop=True)
    )
    source_to_style = {"risc": "-", "baseline": "--"}
    source_to_linewidths = {"risc": 2, "baseline": 1}
    methods = ["Episodic", "RISC (ours)", "MEDAL", "VapRL", "FBRL", "R3L", "Naive"]
    ldf = ldf[ldf["method"].isin(methods)]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for idx, environment in enumerate(
        ["Tabletop Manipulation", "Sawyer Door", "Sawyer Peg", "Minitaur"]
    ):
        df = ldf[ldf["environment"] == environment]

        _, ax = create_plot(
            df,
            method_to_color,
            source_to_style,
            source_to_linewidths,
            title=environment,
            use_ewm=False,
            running_average=10,
            ax=axes[idx],
        )
        _decorate_axis(ax)
        ax.grid(alpha=0.3, linestyle="-")

        if environment == "Minitaur":
            make_hline(
                ax,
                -41.5,
                3.4,
                method_to_color["Episodic"],
                label="Episodic",
                xlim=(0, limits["Minitaur"]),
            )
            make_hline(
                ax,
                -1041.10,
                44.58,
                method_to_color["Naive"],
                label="Naive",
                xlim=(0, limits["Minitaur"]),
            )
            make_hline(
                ax,
                -186.30,
                34.79,
                method_to_color["R3L"],
                label="R3L",
                xlim=(0, limits["Minitaur"]),
            )
            make_hline(
                ax,
                -986.34,
                67.95,
                method_to_color["FBRL"],
                label="FBRL",
                xlim=(0, limits["Minitaur"]),
            )
            ax.grid(alpha=0.08, linestyle="-")
        ax.get_legend().remove()
        ax.set_title(environment)
    axes[0].legend(ncol=7, loc="lower center", bbox_to_anchor=(2.3, -0.35))
    fig.savefig(output_path / f"figure_5.pdf", bbox_inches="tight")


def create_figure_6(results_path: Path, output_path: Path):
    method_to_color = {
        "RISC": color_palette[0],
        "RISC w/o switching": color_palette[6],
        "RISC w/o timeout aware bootstrapping": color_palette[2],
        "FBRL": color_palette[3],
    }

    baselines = pd.read_csv(results_path / "baselines.csv")
    risc_df = pd.read_csv(results_path / "risc.csv")
    full_df = pd.concat([baselines, risc_df])
    limits = {
        "Sawyer Door": 1_000_000,
        "Sawyer Peg": 7_000_000,
        "Tabletop Manipulation": 500_000,
        "Minitaur": 5_000_000,
    }
    ldf = (
        full_df.groupby("environment")
        .apply(lambda x: x[x["train_step"] <= limits[x["environment"].iloc[0]]])
        .reset_index(drop=True)
    )

    # The sample data for this environment is missing. We generate 1000 5 seed samples
    # for each algorithm, and use the sample that results in the best rliable mean, to
    # give the benefit of the doubt to the baselines.
    augmented_df = add_generated_data(
        ldf, "Minitaur", "Episodic", -41.5, 3.4, limits["Minitaur"]
    )
    augmented_df = add_generated_data(
        augmented_df, "Minitaur", "Naive", -1041.10, 44.58, limits["Minitaur"]
    )
    augmented_df = add_generated_data(
        augmented_df, "Minitaur", "R3L", -186.30, 34.79, limits["Minitaur"]
    )
    augmented_df = add_generated_data(
        augmented_df, "Minitaur", "FBRL", -986.34, 67.95, limits["Minitaur"]
    )
    norm_df = create_normalized_df(augmented_df)
    score_dicts = create_score_dicts(norm_df, num_to_average=5)
    score_dicts["RISC"] = score_dicts.pop("RISC (ours)")
    algorithms = list(
        reversed(
            [
                "RISC",
                "RISC w/o switching",
                "RISC w/o timeout aware bootstrapping",
                "FBRL",
            ]
        )
    )
    metric_fns = {
        "IQM": metrics.aggregate_iqm,
        "Mean": metrics.aggregate_mean,
        "P(RISC > Y)": lambda x: metrics.probability_of_improvement(
            score_dicts["RISC"], x
        ),
    }
    score_dicts = {k: v for k, v in score_dicts.items() if k in algorithms}
    aggregate_scores, aggregate_score_cis = get_rliable_metrics(
        score_dicts, list(metric_fns.values())
    )
    aggregate_scores["RISC"][2] = np.nan
    aggregate_score_cis["RISC"][:, 2] = np.nan
    fig, axes = plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=list(metric_fns.keys()),
        algorithms=algorithms,
        xlabel="Normalized Score",
        colors=method_to_color,
        xlabel_y_coordinate=-0.3,
    )
    axes[0].grid(False, axis="y")
    axes[2].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_yticklabels(
        reversed(
            [
                "RISC",
                "RISC w/o switching",
                "RISC w/o timeout aware bootstrapping",
                "FBRL",
            ]
        )
    )
    plt.subplots_adjust(wspace=0.2)
    fig.savefig(output_path / "figure_6.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create",
        "-c",
        default="all",
        choices=[
            "all",
            "figure_3_top",
            "figure_3_bottom",
            "figure_4_left",
            "figure_4_right",
            "figure_5",
            "figure_6",
        ],
        help="Create a specific figure",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        default="results",
        help="Path to the directory containing the data",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="plots",
        help="Path to the directory to save the plots",
    )
    args = parser.parse_args()
    if not args.output.exists() or not args.output.is_dir():
        args.output.mkdir(parents=True)
    if args.create == "all":
        create_figure_3_top(args.data, args.output)
        create_figure_3_bottom(args.data, args.output)
        create_figure_4_left(args.data, args.output)
        create_figure_4_right(args.data, args.output)
        create_figure_5(args.data, args.output)
        create_figure_6(args.data, args.output)
    elif args.create == "figure_3_top":
        create_figure_3_top(args.data, args.output)
    elif args.create == "figure_3_bottom":
        create_figure_3_bottom(args.data, args.output)
    elif args.create == "figure_4_left":
        create_figure_4_left(args.data, args.output)
    elif args.create == "figure_4_right":
        create_figure_4_right(args.data, args.output)
    elif args.create == "figure_5":
        create_figure_5(args.data, args.output)
    elif args.create == "figure_6":
        create_figure_6(args.data, args.output)
    else:
        raise ValueError(f"Unknown figure {args.create}")
