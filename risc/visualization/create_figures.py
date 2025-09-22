import argparse
import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import wandb

warnings.filterwarnings("error")


def load_run_data(
    env_filters,
    algo_filters,
    metrics,
    x_axis,
    project_path,
    fetch_data=True,
    data_file="data.pkl",
):
    # wandb.login()
    api = wandb.Api()

    runs = api.runs(path=project_path)

    data = {}
    for env in env_filters:
        data[env] = {}
        for algo in algo_filters:
            data[env][algo] = {}
            for metric in metrics:
                data[env][algo][metric] = None

    if not fetch_data:
        if not os.path.exists(data_file):
            print(f"Cannot load from ./{data_file}, will fetch online...")
        else:
            with open(data_file, "rb") as file:
                return pickle.load(file)
            print(f"Data loaded from ./{data_file}.")

    for i, run in enumerate(runs):
        env = get_filter_name(run.config, env_filters)
        if env is None:
            continue
        algo = get_filter_name(run.config, algo_filters)
        if algo is None:
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

            print(f"({i}) Loaded data for {run.name} ({run.id}) successfully.")

        except Exception as e:
            print(f"Error loading data for run {run.name}: {e}")
            breakpoint()
            continue

    with open(data_file, "wb") as file:
        pickle.dump(data, file)
    print(f"Data saved to ./{data_file}.")

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
                    df_long["env"] = env_name
                    df_long["algo"] = algo_name
                    df_long["metric"] = metric_name
                    df_long["x_axis"] = df_long.index

                    # Concatenate to the main plotting DataFrame
                    plotting_data = pd.concat([plotting_data, df_long], ignore_index=True)

    return plotting_data


def plot_data(data, x_axis, output_dir, running_average=10):
    plotting_data = convert_to_longform(data)
    environments = plotting_data["env"].unique()
    metrics = plotting_data["metric"].unique()

    # Apply rolling average to each run"s metric values before grouping
    smoothed_data = pd.DataFrame()
    for _, group in plotting_data.groupby(["env", "algo", "metric", "seed"]):
        group["metric_value_smoothed"] = group["metric_value"].rolling(
            window=running_average, min_periods=1, center=True
        ).mean()
        smoothed_data = pd.concat([smoothed_data, group])

    # Group the smoothed data to calculate the mean and standard deviation
    grouped_stats = smoothed_data.groupby(
        [x_axis, "env", "algo", "metric"]
    )["metric_value_smoothed"].agg(["mean", "std", "count"]).reset_index()

    # Calculate the SEM
    grouped_stats["sem"] = grouped_stats["std"] / np.sqrt(grouped_stats["count"])

    # Use the "sem" column to calculate your bounds for plotting
    grouped_stats["upper_bound"] = grouped_stats["mean"] + grouped_stats["sem"]
    grouped_stats["lower_bound"] = grouped_stats["mean"] - grouped_stats["sem"]

    for env in environments:
        for metric in metrics:
            # Filter the stats for the current environment and metric
            plot_stats = grouped_stats[
                (grouped_stats["env"] == env) & (grouped_stats["metric"] == metric)
            ]

            if not plot_stats.empty:
                plt.figure(figsize=(10, 6))

                # Plot the mean line for each algorithm
                for algo in plot_stats["algo"].unique():
                    algo_data = plot_stats[plot_stats["algo"] == algo]
                    plt.plot(
                        algo_data[x_axis],
                        algo_data["mean"],
                        label=algo
                    )

                    # Use fill_between for the SEM band
                    plt.fill_between(
                        algo_data[x_axis],
                        algo_data["lower_bound"],
                        algo_data["upper_bound"],
                        alpha=0.2
                    )

                plt.title(f"{env} - {metric}")  # TODO metric name
                plt.xlabel(x_axis)
                plt.ylabel(metric)  # TODO metric name
                # plt.xlim()  # TODO
                # plt.ylim()  # TODO
                plt.grid(True)
                plt.legend(title="Algorithm")
                plt.show()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


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
        # "FourRooms": {
        #     ("kwargs", "env_fn", "kwargs", "env_name"): (lambda _: _ == "MiniGrid-FourRooms-v1"),
        # },
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
            (lambda _: _ == 100000),
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

    plot_data(data, x_axis, output_dir)


def main():
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


if __name__ == "__main__":
    main()

main()
