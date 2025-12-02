import argparse
import os
import pickle
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

warnings.filterwarnings("error")

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = "xx-large"
mpl.rcParams['axes.labelsize'] = "xx-large"
mpl.rcParams['legend.fontsize'] = "medium"
# mpl.rcParams['axes.titlesize'] = 30
# mpl.rcParams['axes.labelsize'] = 25
# mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['legend.framealpha'] = 0.7

# __file__ = "/Users/george/Desktop/RISC/risc/visualization/create_figures.py"  # DEBUG


def load_run_data(
    env_filters,
    algo_filters,
    metrics,
    x_axis,
    project_path,
    fetch_data=True,
    data_file=None,
):
    if data_file is None:
        data_file = "data.pkl"
    data_file_path = Path(__file__).resolve().parent / "data.pkl"

    # wandb.login()
    api = wandb.Api()

    runs = api.runs(path=project_path)

    data = {}
    loaded_runs_ids = []

    if os.path.exists(data_file_path):
        with open(data_file_path, "rb") as file:
            data, loaded_runs_ids = pickle.load(file)

    # Create new entries
    for env in env_filters:
        data[env] = data.get(env, {})
        for algo in algo_filters:
            data[env][algo] = data[env].get(algo, {})
            for metric, _ in metrics:
                data[env][algo][metric] = data[env][algo].get(metric, None)

    # Skip checking online if requested
    if not fetch_data:
        assert os.path.exists(data_file_path), f"Data file: {data_file_path} does not exist."
        print(f"Data loaded from {data_file_path}.")
        return data

    for run_idx, run in enumerate(runs):
        # Skip run if loaded already
        if run.id in loaded_runs_ids:
            print(f"[{run_idx}] Skipped {run.name} ({run.id}) as it is loaded already.")
            continue

        env = get_filter_name(run.config, env_filters)
        if env is None:
            print(f"[{run_idx}] Skipped {run.name} ({run.id}) as it is has no matching env.")
            continue

        algo = get_filter_name(run.config, algo_filters)
        if algo is None:
            print(f"[{run_idx}] Skipped {run.name} ({run.id}) as it is has no matching algorithm.")
            continue

        # Skip run if tagged with "ignore"
        if any(tag.lower() == 'ignore' for tag in run.tags):
            print(f"[{run_idx}] Ignored {run.name} ({run.id}).")
            continue

        seed = run.config["kwargs"]["seed"]

        try:
            for metric, _ in metrics:
                if metric not in run.summary.keys():
                    print(f"      (skipped metric {metric} from {run.id})")
                    continue

                history = run.history(keys=[metric], x_axis=x_axis)
                history = history.set_index(x_axis).rename(columns={metric: seed})

                data[env][algo][metric] = merge_data(data[env][algo][metric], history)

            print(f"[{run_idx}] Loaded data for {run.name} ({run.id}) successfully.")
            loaded_runs_ids.append(run.id)

        except Exception as e:
            print(f"Error loading data for run {run.name}: {e}")
            breakpoint()
            continue

    with open(data_file_path, "wb") as file:
        pickle.dump((data, loaded_runs_ids), file)
    print(f"Data saved to {data_file_path}.")

    return data


def check_config_match(config, filter_criteria):
    """Checks if a config matches a single filter criteria dictionary."""
    matches = True
    for keys, criterion in filter_criteria.items():
        current_item = config
        for key in keys:
            if not isinstance(current_item, dict):
                return False
            if key not in current_item:
                current_item = False
                break
            else:
                current_item = current_item[key]
        matches &= criterion(current_item)
        if not matches:
            return False
    return True


def get_filter_name(config, filters):
    for name, filter in filters.items():
        if check_config_match(config, filter):
            return name
    return None


def merge_data(maybe_df, df):
    if maybe_df is None:
        return df
    else:
        try:
            return pd.merge(
                maybe_df,
                df,
                left_index=True,
                right_index=True,
                how="outer"
            )
        except Exception as e:
            print(f"Error merging dataframes: {e}")
            breakpoint()

def convert_to_longform(data):
    plotting_data = pd.DataFrame()

    for env_name, env_data in data.items():
        for algo_name, algo_data in env_data.items():
            for metric_name, metric_df in algo_data.items():
                if metric_df is not None:
                    # Melt the DataFrame to long-form format
                    # This will create a "seed" column and a "metric_value" column
                    df_long = metric_df.melt(
                        ignore_index=False,  # Keep the x_axis index
                        var_name="seed",
                        value_name="metric_value"
                    ).reset_index()

                    # Add columns for the environment, algorithm, and metric
                    df_long["environment"] = env_name
                    df_long["algorithm"] = algo_name
                    df_long["metric"] = metric_name
                    df_long["x_axis"] = df_long.index

                    # Concatenate to the main plotting DataFrame
                    plotting_data = pd.concat([plotting_data, df_long], ignore_index=True)

    return plotting_data


def smoothen_data(data, running_average_window):
    # Apply rolling average to each run"s metric values before grouping
    smoothed_data = pd.DataFrame()
    for _, group in data.groupby(["environment", "algorithm", "metric", "seed"]):
        group["metric_value_smoothed"] = group["metric_value"].rolling(
            window=running_average_window, min_periods=1, center=True
        ).mean()
        smoothed_data = pd.concat([smoothed_data, group])
    return smoothed_data


def calculate_mean_error_stats(data, x_axis):
    # Group smoothed data to calculate the mean and standard deviation
    grouped_stats = data.groupby(
        [x_axis, "environment", "algorithm", "metric"]
    )["metric_value_smoothed"].agg(["mean", "std", "count"]).reset_index()

    # Calculate the SEM
    grouped_stats["sem"] = grouped_stats["std"] / np.sqrt(grouped_stats["count"])

    # Use the "sem" column to calculate your bounds for plotting
    grouped_stats["upper_bound"] = grouped_stats["mean"] + grouped_stats["sem"]
    grouped_stats["lower_bound"] = grouped_stats["mean"] - grouped_stats["sem"]

    return grouped_stats


def _plot_on_axis(
    ax,
    data_frame,
    x_axis,
    algorithms,
    colors,
    metric_display_name,
    xmax_val,
    ymax_val,
    is_left_col,
    is_bottom_row
):
    """
    Helper function to plot a single metric/env combination onto a specific axis.
    Returns handles and labels for the legend.
    """
    algo_colors = np.roll(colors, 1)

    handles = []
    labels = []

    for algorithm in algorithms:
        algo_data = data_frame[data_frame["algorithm"] == algorithm]

        if algo_data.empty:
            continue

        algo_colors = np.roll(algo_colors, -1)

        # Plot and capture the line handle
        line, = ax.plot(
            algo_data[x_axis],
            algo_data["mean"],
            color=algo_colors[0],
            label=algorithm
        )

        # Add to local collection for potential legend export
        handles.append(line)
        labels.append(algorithm)

        ax.fill_between(
            algo_data[x_axis],
            algo_data["lower_bound"],
            algo_data["upper_bound"],
            color=algo_colors[0],
            alpha=0.2
        )

    # --- Formatting ---
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_major_formatter(lambda x, _: str(int(x / 1000)) + "k")
    ax.set_xlim([0, xmax_val])
    ax.set_ylim([0, ymax_val])

    if is_left_col:
        # slightly larger font if it's the main descriptor
        ax.set_ylabel(metric_display_name, fontsize=14, fontweight='bold')

    if not is_bottom_row:
        ax.set_xticklabels([])

    return handles, labels


def plot_data(
    data,
    output_dir,
    x_axis,
    environments,
    algorithms,
    metrics,
    running_average_window,
    xmax,
    ymax=1,
    colors=None,
    figsize=(4, 3),
    experiment_name="experiment",
    plot_title=None,
):
    # --- 1. Preservation of Initial Rolling Logic ---
    for i, algorithm in enumerate(algorithms):
        algorithms[i] = np.roll(algorithm, 1).flatten()
    running_average_window = np.roll(running_average_window, 1).flatten()
    xmax = np.roll(xmax, 1).flatten()
    ymax = np.roll(ymax, 1).flatten()

    long_data = convert_to_longform(data)

    # --- 2. Setup Figure and Grid ---

    single_env_mode = (len(environments) == 1)

    if single_env_mode:
        # Horizontal layout: 1 Row, N Columns (one per metric)
        n_rows = 1
        n_cols = len(metrics)
    else:
        # Standard layout: N Rows (metrics), M Columns (environments)
        n_rows = len(metrics)
        n_cols = len(environments)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        sharex=False,
        sharey=False
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    global_legend_handles = []
    global_legend_labels = []
    legend_captured = False

    # --- 3. Iterate Environments ---
    for col_idx, environment in enumerate(environments):

        # --- Strict Preservation of Per-Env Rolling Logic ---
        running_average_window = np.roll(running_average_window, -1)
        ymax = np.roll(ymax, -1)
        xmax = np.roll(xmax, -1)

        for i, algorithm in enumerate(algorithms):
            algorithms[i] = np.roll(algorithm, -1)
        env_algorithms = [algorithm[0] for algorithm in algorithms]
        # ----------------------------------------------------

        if not single_env_mode:
            axes[0, col_idx].set_title(f"{environment}", fontsize=24, pad=10)

        # --- 4. Iterate Metrics ---
        for row_idx, (metric, metric_display_name) in enumerate(metrics):

            if single_env_mode:
                ax = axes[0, row_idx]
                is_left_col = True
                is_bottom_row = True
            else:
                ax = axes[row_idx, col_idx]
                is_left_col = (col_idx == 0)
                is_bottom_row = (row_idx == n_rows - 1)

            relevant_data = long_data[
                (long_data["environment"] == environment)
                & (long_data["metric"] == metric)
                & (long_data["algorithm"].isin(env_algorithms))
            ]

            if relevant_data.empty:
                ax.set_visible(False)
                continue

            smoothed_data = smoothen_data(relevant_data, running_average_window[0])
            plot_stats = calculate_mean_error_stats(smoothed_data, x_axis)

            handles, labels = _plot_on_axis(
                ax=ax,
                data_frame=plot_stats,
                x_axis=x_axis,
                algorithms=env_algorithms,
                colors=colors,
                metric_display_name=metric_display_name,
                xmax_val=xmax[0],
                ymax_val=ymax[0],
                is_left_col=is_left_col,
                is_bottom_row=is_bottom_row
            )

            if not legend_captured and handles:
                global_legend_handles = handles
                global_legend_labels = labels
                legend_captured = True

    # --- 5. Global Layout, Legend & Saving ---

    if plot_title:
        fig.suptitle(plot_title, fontsize=28, y=0.98 if not single_env_mode else 1.05)

    top_adjust = 0.9 if plot_title else 1.0

    if global_legend_handles:
        if single_env_mode:
            # === OPTION A: Vertical Legend on the Right (Single Env) ===
            fig.legend(
                global_legend_handles,
                global_legend_labels,
                loc='center left',      # Anchor left edge of legend...
                bbox_to_anchor=(1.0, 0.5), # ...to the right edge of the figure
                ncol=1,                 # Vertical stack
                frameon=False,
                fontsize=20
            )
            # Adjust layout: Reserve 15% space on the right side for the legend
            # rect=[left, bottom, right, top]
            plt.tight_layout(rect=[0.02, 0.05, 0.96, top_adjust])

        else:
            # === OPTION B: Horizontal Legend on the Bottom (Multi Env) ===
            fig.legend(
                global_legend_handles,
                global_legend_labels,
                loc='lower center',
                ncol=len(global_legend_labels), # Horizontal stack
                bbox_to_anchor=(0.5, 0.0),
                frameon=False,
                fontsize=20
            )
            # Adjust layout: Reserve space at the bottom for the legend
            plt.tight_layout(rect=[0.02, 0.12, 1, top_adjust])
    else:
        # Fallback if no legend
        plt.tight_layout(rect=[0.02, 0.05, 1, top_adjust])

    output_dir_path = Path(__file__).resolve().parent / output_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)

    file_name = f"{experiment_name}_results_grid.svg"
    output_file_path = output_dir_path / file_name

    plt.savefig(output_file_path, format="svg", bbox_inches="tight")
    print(f"Grid plot saved to {output_file_path}")
    plt.close()


def create_figures(output_dir, entity, project, fetch_data=True):
    base_env_filter = {
        ("kwargs", "env_fn", "kwargs", "env_name"): (lambda _: _ == ""),
        ("kwargs", "env_fn", "kwargs", "action_probability"): (lambda _: _ is False or _ == 1.0),
    }
    base_hallway_filter = {
        **base_env_filter,
        ("kwargs", "env_fn", "kwargs", "env_name"): (lambda _: _ == "MiniGrid-Hallway-v1"),
        ("kwargs", "env_fn", "kwargs", "num_hallways"): (lambda _: _ == 1),
        ("kwargs", "env_fn", "kwargs", "hallway_length"): (lambda _: _ > 0),
    }
    env_filters = {}

    for n in [2, 4, 6]:
        env_filters[f"Hallway {n}-steps"] = {
            **base_hallway_filter,
            ("kwargs", "env_fn", "kwargs", "hallway_length"): (lambda _, n=n: _ == n),
        }
    for env_name in ["FourRooms", "BugTrap", "NineRooms", "NineRoomsLocked"]:
        env_filters[env_name] = {
            **base_env_filter,
            ("kwargs", "env_fn", "kwargs", "env_name"):
            (lambda _, env_name=env_name: _ == f"MiniGrid-{env_name}-v1"),
        }

    env_filters["Hallway 4-steps (probabilistic transitions)"] = {
        **env_filters["Hallway 4-steps"],
        ("kwargs", "env_fn", "kwargs", "action_probability"): (lambda _: _ == 0.8),
    }

    base_sierl_filter = {
        ("kwargs", "agent", "kwargs", "goal_generator", "name"):
        (lambda _: _ == "OmniGoalGenerator"),
        ("kwargs", "agent", "kwargs", "goal_switcher", "kwargs", "switching_probability"):
        (lambda _: _ > 0),
        ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "max_familiarity"):
        (lambda _: _ < 1),
        ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "weights"):
        (lambda _: _ == [1.5, 0, 1.0, 0.5]),
        ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "temperature_schedule",
         "kwargs", "value"): (lambda _: _ == 0.5),
        ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "random_selection"):
        (lambda _: _ is False),
        ("kwargs", "agent", "kwargs", "replay_buffer", "kwargs", "capacity"):
        (lambda _: _ == 100000 or _ == 300000),
    }
    base_qlearning_filter = {
        ("kwargs", "agent", "kwargs", "goal_generator", "name"):
        (lambda _: _ == "NoGoalGenerator"),
        ("kwargs", "env_fn", "kwargs", "novelty_bonus"):
        (lambda _: _ == 0),
        ("kwargs", "train_random_goals"):
        (lambda _: _ is False),
        ("kwargs", "agent", "kwargs", "replay_buffer", "kwargs", "her_ratio"):
        (lambda _: _ == 0),
    }
    base_ablation_filter = {
        ("kwargs", "agent", "kwargs", "goal_generator", "name"):
        (lambda _: _ == "OmniGoalGenerator"),
        ("kwargs", "agent", "kwargs", "goal_switcher", "kwargs", "switching_probability"):
        (lambda _: _ > 0),
        ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "max_familiarity"):
        (lambda _: _ < 1),
        ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "random_selection"):
        (lambda _: _ is False),
    }
    algo_filters = {
        "Q-learning": {
            **base_qlearning_filter
        },
        "Random-goals Q-learning": {
            **base_qlearning_filter,
            ("kwargs", "train_random_goals"): (lambda _: _ is True),
        },
        "HER": {
            **base_qlearning_filter,
            ("kwargs", "agent", "kwargs", "replay_buffer", "kwargs", "her_ratio"):
            (lambda _: _ > 0),
        },
        "Novelty bonuses": {
            **base_qlearning_filter,
            ("kwargs", "env_fn", "kwargs", "novelty_bonus"):
            (lambda _: _ > 0),
        },
        "No frontier filtering": {
            **base_ablation_filter,
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "max_familiarity"):
            (lambda _: _ == 1),
        },
        "No early switching": {
            **base_ablation_filter,
            ("kwargs", "agent", "kwargs", "goal_switcher", "kwargs", "switching_probability"):
            (lambda _: _ == 0),
        },
        "No prioritization": {
            **base_ablation_filter,
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "random_selection"):
            (lambda _: _ is True),
        },
    }
    for f in [0.001, 0.01, 0.1, 0.5, 0.7, 0.8, 0.9, 0.95]:
        algo_filters[f"SIERL F={f}"] = {
            **base_sierl_filter,
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "max_familiarity"):
            (lambda _, f=f: _ == f),
        }
    for t in [0.1, 1.5]:
        algo_filters[f"SIERL T={t}"] = {
            **base_sierl_filter,
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "temperature_schedule",
             "kwargs", "value"): (lambda _, t=t: _ == t),
        }
    weight_variants = {
        "c_c=0": [1.0, 0.0, 0.0, 1.0],
        "c_g=0": [1.0, 0.0, 1.0, 0.0],
        "c_c=c_g": [1.0, 0.0, 1.0, 1.0],
        "c_n=0.5": [0.5, 0.0, 1.0, 0.5],
        "c_n=3.0": [3.0, 0.0, 1.0, 0.5],
    }
    for name, w in weight_variants.items():
        algo_filters[f"SIERL {name}"] = {
            **base_sierl_filter,
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "weights"):
            (lambda _, w=w: _ == w),
        }

    metrics = [
        ["test/0_success", "Main-goal Success"],
        ["test_random_goals/0_success", "Random-goal Success"],
        ["lateral/success", "Sub-goal Success (training)"],
    ]

    x_axis = "train_step"

    project_path = f"{entity}/{project}"

    data = load_run_data(
        env_filters=env_filters,
        algo_filters=algo_filters,
        metrics=metrics,
        x_axis=x_axis,
        project_path=project_path,
        fetch_data=fetch_data,
    )

    colors = [
        '#e41a1c',
        '#4daf4a',
        '#ff7f00',
        '#984ea3',
        '#377eb8',
        '#dede00',
        '#f781bf',
        '#a65628',
        '#999999',
    ]

    plots_args = [
        {
            "experiment_name": "experiments",
            "x_axis": "train_step",
            "environments": [
                "Hallway 2-steps",
                "Hallway 4-steps",
                "Hallway 6-steps",
                "FourRooms",
                "BugTrap",
            ],
            "algorithms": [
                [
                    "SIERL F=0.9",
                    "SIERL F=0.9",
                    "SIERL F=0.95",
                    "SIERL F=0.8",
                    "SIERL F=0.7",
                ],
                "Q-learning",
                "Random-goals Q-learning",
                "HER",
                "Novelty bonuses",
            ],
            "metrics": metrics[:2],
            "running_average_window": 15,
            "colors": colors,
            "xmax": [180000, 110000, 500000, 215000, 350000],
            # "ymax": 1,
            # "figsize": (6, 5),
        },
        {
            "experiment_name": "ablations",
            "x_axis": "train_step",
            "environments": ["Hallway 6-steps", "FourRooms"],
            "algorithms": [
                "SIERL F=0.8",
                "SIERL F=0.95",
                "No early switching",
                "No frontier filtering",
                "No prioritization",
            ],
            "metrics": metrics[:2],
            "running_average_window": 15,
            "colors": colors,
            "xmax": [400000, 215000],
            # "ymax": 1,
            # "figsize": (6, 5),
        },
        {
            "experiment_name": "sensitivity_fthr",
            "x_axis": "train_step",
            "environments": ["Hallway 4-steps"],
            "algorithms": [
                "SIERL F=0.01",
                "SIERL F=0.5",
                "SIERL F=0.8",
                "SIERL F=0.95",
            ],
            "metrics": metrics[:2],
            "running_average_window": 15,
            "colors": colors,
            "xmax": [300000],
            # "ymax": 1,
            # "figsize": (6, 5),
        },
        {
            "experiment_name": "sensitivity_softmax-temp",
            "x_axis": "train_step",
            "environments": ["Hallway 4-steps"],
            "algorithms": [
                "SIERL T=0.1",
                "SIERL F=0.8",
                "SIERL T=1.5",
            ],
            "metrics": metrics[:2],
            "running_average_window": 15,
            "colors": colors,
            "xmax": [250000],
            # "ymax": 1,
            # "figsize": (6, 5),
        },
        {
            "experiment_name": "sensitivity_path-weights",
            "x_axis": "train_step",
            "environments": ["Hallway 4-steps"],
            "algorithms": [
                "SIERL c_g=0",
                "SIERL c_c=0",
                "SIERL c_c=c_g",
            ],
            "metrics": metrics[:2],
            "running_average_window": 15,
            "colors": colors,
            "xmax": [150000],
            # "ymax": 1,
            # "figsize": (6, 5),
        },
        {
            "experiment_name": "sensitivity_novelty-weight",
            "x_axis": "train_step",
            "environments": ["Hallway 4-steps"],
            "algorithms": [
                "SIERL c_n=0.5",
                "SIERL F=0.8",
                "SIERL c_n=3.0",
            ],
            "metrics": metrics[:2],
            "running_average_window": 15,
            "colors": colors,
            "xmax": [150000],
            # "ymax": 1,
            # "figsize": (6, 5),
        },
        {
            "experiment_name": "probabilistic_transitions",
            "x_axis": "train_step",
            "environments": ["Hallway 4-steps (probabilistic transitions)"],
            "algorithms": [
                "SIERL F=0.8",
                "Q-learning",
                "Random-goals Q-learning",
                "HER",
                "Novelty bonuses",
            ],
            "metrics": metrics[:2],
            "running_average_window": 15,
            "colors": colors,
            "xmax": [150000],
            # "ymax": 1,
            # "figsize": (6, 5),
        },
    ]

    for plot_args in plots_args:
        plot_data(data, output_dir, **plot_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default="plots",
        help="Path to the directory to save the plots",
    )
    parser.add_argument(
        "--wandb-entity",
        "-n",
        type=str,
        default="",
        help="Name of the W&B entity that contains the project with the logged runs",
    )
    parser.add_argument(
        "--wandb-project",
        "-p",
        type=str,
        default="Experiments",
        help="The W&B project with the logged runs",
    )
    parser.add_argument(
        "--no-fetch-data",
        action="store_true",
        help="Do not fetch logged data",
    )
    args = parser.parse_args()
    create_figures(
        output_dir=args.output_dir,
        # entity=args.wandb_entity,
        entity="g-sotirchos-tu-delft",
        # project=args.wandb_project,
        project="Experiments",
        # fetch_data=(not args.no_fetch_data),
        fetch_data=False,
    )
