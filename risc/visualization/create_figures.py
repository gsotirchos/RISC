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
# mpl.rcParams['axes.titlesize'] = "xx-large"
# mpl.rcParams['axes.labelsize'] = "xx-large"
# mpl.rcParams['legend.fontsize'] = "medium"
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['legend.fontsize'] = 12
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

    for env in env_filters:
        data[env] = {}
        for algo in algo_filters:
            data[env][algo] = {}
            for metric in metrics:
                data[env][algo][metric] = None

    if os.path.exists(data_file_path):
        with open(data_file_path, "rb") as file:
            data, loaded_runs_ids = pickle.load(file)

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
            for metric in metrics:
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


def plot_data(
    data,
    output_dir,
    x_axis,
    environments,
    algorithms,
    metrics,
    metric_names,
    running_average_window,
    xmax,
    ymax=1,
    legend_loc="center right",
    colors=None,
    figsize=(6, 5),
    experiment_name="experiment",
):
    long_data = convert_to_longform(data)
    smoothed_data = smoothen_data(long_data, running_average_window)
    stats = calculate_mean_error_stats(smoothed_data, x_axis)

    xmax = np.roll(xmax, 1).flatten()
    ymax = np.roll(ymax, 1).flatten()
    for env_idx, environment in enumerate(environments):
        ymax = np.roll(ymax, -1)
        xmax = np.roll(xmax, -1)
        for metric_idx, metric in enumerate(metrics):
            plot_stats = stats[
                (stats["environment"] == environment)
                & (stats["metric"] == metric)
                & (stats["algorithm"].isin(algorithms))
            ]

            if plot_stats.empty:
                continue

            plt.figure(figsize=figsize)

            # Get display names, defaulting to the original name if not in the dict
            metric_display_name = metric_names.get(metric, metric) if metric_names else metric

            # Plot the mean line for each algorithm
            algo_colors = np.roll(colors, 1)
            for algorithm in algorithms:
                algo_data = plot_stats[plot_stats["algorithm"] == algorithm]

                if algo_data.empty:
                    continue

                algo_colors = np.roll(algo_colors, -1)
                plt.plot(
                    algo_data[x_axis],
                    algo_data["mean"],
                    color=algo_colors[0],
                    label=algorithm
                )
                plt.fill_between(
                    algo_data[x_axis],
                    algo_data["lower_bound"],
                    algo_data["upper_bound"],
                    color=algo_colors[0],
                    alpha=0.2
                )

            # Set plot titles and labels
            plt.title(f"{environment}")
            plt.xlabel("Train step")
            plt.ylabel(metric_display_name)

            # Set axis limits
            plt.xlim([0, xmax[0]])
            plt.ylim([0, ymax[0]])

            plt.grid(True)
            plt.legend(title="", loc=legend_loc[env_idx][metric_idx])

            output_dir_path = Path(__file__).resolve().parent / output_dir
            output_dir_path.mkdir(parents=True, exist_ok=True)

            file_name = f"{experiment_name}_{environment}_{metric_display_name}.svg"
            file_name = file_name.replace(" ", "_").replace("/","_")
            output_file_path = output_dir_path / file_name

            # Save the figure with high quality settings
            plt.savefig(output_file_path, format="svg", bbox_inches="tight")
            print(f"Plot saved to {output_file_path}")
            # plt.show()
            plt.close()


def create_figures(output_dir, entity, project, fetch_data=True):
    env_filters = {
        "Hallway 2-steps": {
            ("kwargs", "env_fn", "kwargs", "env_name"): (lambda _: _ == "MiniGrid-Hallway-v1"),
            ("kwargs", "env_fn", "kwargs", "hallway_length"):  (lambda _: _ == 2),
        },
        "Hallway 4-steps": {
            ("kwargs", "env_fn", "kwargs", "env_name"): (lambda _: _ == "MiniGrid-Hallway-v1"),
            ("kwargs", "env_fn", "kwargs", "hallway_length"):  (lambda _: _ == 4),
        },
        "Hallway 6-steps": {
            ("kwargs", "env_fn", "kwargs", "env_name"): (lambda _: _ == "MiniGrid-Hallway-v1"),
            ("kwargs", "env_fn", "kwargs", "hallway_length"):  (lambda _: _ == 6),
        },
        "FourRooms": {
            ("kwargs", "env_fn", "kwargs", "env_name"): (lambda _: _ == "MiniGrid-FourRooms-v1"),
        },
        "BugTrap": {
            ("kwargs", "env_fn", "kwargs", "env_name"): (lambda _: _ == "MiniGrid-BugTrap-v1"),
        },
    }

    algo_filters = {
        "SIERL": {
            ("kwargs", "agent", "kwargs", "goal_generator", "name"):
            (lambda _: _ == "OmniGoalGenerator"),
            ("kwargs", "agent", "kwargs", "goal_switcher", "kwargs", "switching_probability"):
            (lambda _: _ > 0),
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "max_familiarity"):
            (lambda _: _ < 1),
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "random_selection"):
            (lambda _: _ is False),
            ("kwargs", "agent", "kwargs", "replay_buffer", "kwargs", "capacity"):
            (lambda _: _ == 100000 or _ == 300000),
        },
        "Q-learning": {
            ("kwargs", "agent", "kwargs", "goal_generator", "name"):
            (lambda _: _ == "NoGoalGenerator"),
            ("kwargs", "env_fn", "kwargs", "novelty_bonus"):
            (lambda _: _ == 0),
            ("kwargs", "train_random_goals"):
            (lambda _: _ is False),
            ("kwargs", "agent", "kwargs", "replay_buffer", "kwargs", "her_ratio"):
            (lambda _: _ == 0),
        },
        "Random-goals Q-learning": {
            ("kwargs", "agent", "kwargs", "goal_generator", "name"):
            (lambda _: _ == "NoGoalGenerator"),
            ("kwargs", "env_fn", "kwargs", "novelty_bonus"):
            (lambda _: _ == 0),
            ("kwargs", "train_random_goals"):
            (lambda _: _ is True),
            ("kwargs", "agent", "kwargs", "replay_buffer", "kwargs", "her_ratio"):
            (lambda _: _ == 0),
        },
        "HER": {
            ("kwargs", "agent", "kwargs", "goal_generator", "name"):
            (lambda _: _ == "NoGoalGenerator"),
            ("kwargs", "env_fn", "kwargs", "novelty_bonus"):
            (lambda _: _ == 0),
            ("kwargs", "train_random_goals"):
            (lambda _: _ is False),
            ("kwargs", "agent", "kwargs", "replay_buffer", "kwargs", "her_ratio"):
            (lambda _: _ > 0),
        },
        "Novelty bonuses": {
            ("kwargs", "agent", "kwargs", "goal_generator", "name"):
            (lambda _: _ == "NoGoalGenerator"),
            ("kwargs", "env_fn", "kwargs", "novelty_bonus"):
            (lambda _: _ > 0),
            ("kwargs", "train_random_goals"):
            (lambda _: _ is False),
            ("kwargs", "agent", "kwargs", "replay_buffer", "kwargs", "her_ratio"):
            (lambda _: _ == 0),
        },
        "No frontier filtering": {
            ("kwargs", "agent", "kwargs", "goal_generator", "name"):
            (lambda _: _ == "OmniGoalGenerator"),
            ("kwargs", "agent", "kwargs", "goal_switcher", "kwargs", "switching_probability"):
            (lambda _: _ > 0),
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "max_familiarity"):
            (lambda _: _ == 1),
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "random_selection"):
            (lambda _: _ is False),
        },
        "No early switching": {
            ("kwargs", "agent", "kwargs", "goal_generator", "name"):
            (lambda _: _ == "OmniGoalGenerator"),
            ("kwargs", "agent", "kwargs", "goal_switcher", "kwargs", "switching_probability"):
            (lambda _: _ == 0),
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "max_familiarity"):
            (lambda _: _ < 1),
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "random_selection"):
            (lambda _: _ is False),
        },
        "No prioritization": {
            ("kwargs", "agent", "kwargs", "goal_generator", "name"):
            (lambda _: _ == "OmniGoalGenerator"),
            ("kwargs", "agent", "kwargs", "goal_switcher", "kwargs", "switching_probability"):
            (lambda _: _ > 0),
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "max_familiarity"):
            (lambda _: _ < 1),
            ("kwargs", "agent", "kwargs", "goal_generator", "kwargs", "random_selection"):
            (lambda _: _ is True),
        },
    }

    metrics = [
        "test/0_success",
        "test_random_goals/0_success",
        "lateral/success",
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

    metric_names = {
        "test/0_success": "Main-goal Success",
        "test_random_goals/0_success": "Random-goal Success",
        "lateral/success": "Sub-goal Success (training)",
    }

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
                "SIERL",
                "Q-learning",
                "Random-goals Q-learning",
                "HER",
                "Novelty bonuses",
            ],
            "metrics": metrics[:2],
            "metric_names": metric_names,
            "running_average_window": 15,
            "legend_loc": [
                [
                    "lower right",
                    "center right",
                    "lower right",
                ],
                [
                    "lower right",
                    "center right",
                    "lower right",
                ],
                [
                    "lower right",
                    "lower right",
                    "lower right",
                ],
                [
                    "lower right",
                    "center right",
                    "lower right",
                ],
                [
                    "lower right",
                    "upper left",
                    "lower right",
                ],
            ],
            "colors": colors,
            "xmax": [130000, 110000, 650000, 250000, 330000],
            # "ymax": 1,
            # "figsize": (6, 5),
        },
        {
            "experiment_name": "ablations",
            "x_axis": "train_step",
            "environments": ["Hallway 6-steps"],
            "algorithms": [
                "SIERL",
                "No early switching",
                "No frontier filtering",
                "No prioritization",
            ],
            "metrics": metrics[:2],
            "metric_names": metric_names,
            "running_average_window": 15,
            "legend_loc": [
                [
                "lower right",
                "lower right",
                "lower right"
                ]
            ],
            "colors": colors,
            "xmax": [700000],
            # "ymax": 1,
            # "figsize": (6, 5),
        },
        {
            "experiment_name": "ablations",
            "x_axis": "train_step",
            "environments": ["Hallway 6-steps"],
            "algorithms": [
                "SIERL",
                "No early switching",
                "No frontier filtering",
                "No prioritization",
            ],
            "metrics": [metrics[2]],
            "metric_names": metric_names,
            "running_average_window": 70,
            "legend_loc": [
                [
                "lower right",
                "lower right",
                "lower right"
                ]
            ],
            "colors": colors,
            "xmax": [700000],
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
        fetch_data=(not args.no_fetch_data),
        # fetch_data=False,
    )
