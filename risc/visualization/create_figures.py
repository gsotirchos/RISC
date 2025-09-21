import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import wandb


def load_run_data(env_filters, algo_filters, metrics, x_axis, project_path, fetch_data=True):
    # wandb.login()
    api = wandb.Api()

    project_path = "g-sotirchos-tu-delft/Experiments"
    runs = api.runs(path=project_path)

    output_dir = "visualization/data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = {}
    for env in env_filters:
        data[env] = {}
        for algo in algo_filters:
            data[env][algo] = {}
            for metric in metrics:
                data[env][algo][metric] = None

    if not fetch_data:
        if not os.path.exists("data.pkl"):
            print("Cannot load from ./data.pkl, will fetch online...")
        else:
            with open('data.pkl', 'rb') as file:
                return pickle.load(file)

    for run in runs:
        env = get_filter_name(run.config, env_filters)
        if env is None:
            continue
        algo = get_filter_name(run.config, algo_filters)
        if algo is None:
            continue
        seed = run.config['kwargs']['seed']

        try:
            for metric in metrics:
                history = run.history(keys=[metric], x_axis=x_axis)
                history = history.set_index(x_axis).rename(columns={metric: seed})

                if history.empty:
                    print(f"No data metric '{metric}' in run {run.name}. Skipping.")
                    continue

                data[env][algo][metric] = merge_data(data[env][algo][metric], history)

            print(f"Data for run {run.name} ({run.id})loaded successfully.")

        except Exception as e:
            print(f"Error loading data for run {run.name}: {e}")
            continue

    with open('data.pkl', 'wb') as file:
        pickle.dump(data, file)

    return data


def check_config_match(config, filter_criteria):
    """Checks if a config matches a single filter criteria dictionary."""
    for keys, required_value in filter_criteria.items():
        current_dict = config
        for key in keys:
            if not isinstance(current_dict, dict) or key not in current_dict:
                return False
            current_dict = current_dict[key]
        if current_dict != required_value:
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
        return pd.merge(
            maybe_df,
            df,
            left_index=True,
            right_index=True,
            how='outer'
        )


def convert_to_longform(data):
    plotting_data = pd.DataFrame()

    for env_name, env_data in data.items():
        for algo_name, algo_data in env_data.items():
            for metric_name, metric_df in algo_data.items():
                if metric_df is not None:
                    # Melt the DataFrame to long-form format
                    # This will create a 'seed' column and a 'metric_value' column
                    df_long = metric_df.melt(
                        ignore_index=False,  # Keep the x_axis index
                        var_name='seed',
                        value_name='metric_value'
                    ).reset_index()

                    # Add columns for the environment, algorithm, and metric
                    df_long['env'] = env_name
                    df_long['algo'] = algo_name
                    df_long['metric'] = metric_name
                    df_long['x_axis'] = df_long.index

                    # Concatenate to the main plotting DataFrame
                    plotting_data = pd.concat([plotting_data, df_long], ignore_index=True)

    return plotting_data


def plot_data(data, x_axis, output_dir, running_average=10):
    plotting_data = convert_to_longform(data)
    environments = plotting_data['env'].unique()
    metrics = plotting_data['metric'].unique()

    # Apply rolling average to each run's metric values before grouping
    smoothed_data = pd.DataFrame()
    for _, group in plotting_data.groupby(['env', 'algo', 'metric', 'seed']):
        group['metric_value_smoothed'] = group['metric_value'].rolling(
            window=running_average, min_periods=1, center=True
        ).mean()
        smoothed_data = pd.concat([smoothed_data, group])

    # Group the smoothed data to calculate the mean and standard deviation
    grouped_stats = smoothed_data.groupby(
        [x_axis, 'env', 'algo', 'metric']
    )['metric_value_smoothed'].agg(['mean', 'std', 'count']).reset_index()

    # Calculate the SEM
    grouped_stats['sem'] = grouped_stats['std'] / np.sqrt(grouped_stats['count'])

    # Use the 'sem' column to calculate your bounds for plotting
    grouped_stats['upper_bound'] = grouped_stats['mean'] + grouped_stats['sem']
    grouped_stats['lower_bound'] = grouped_stats['mean'] - grouped_stats['sem']

    for env in environments:
        for metric in metrics:
            # Filter the stats for the current environment and metric
            plot_stats = grouped_stats[
                (grouped_stats['env'] == env) & (grouped_stats['metric'] == metric)
            ]

            if not plot_stats.empty:
                plt.figure(figsize=(10, 6))

                # Plot the mean line for each algorithm
                for algo in plot_stats['algo'].unique():
                    algo_data = plot_stats[plot_stats['algo'] == algo]
                    plt.plot(
                        algo_data[x_axis],
                        algo_data['mean'],
                        label=algo
                    )

                    # Use fill_between for the SEM band
                    plt.fill_between(
                        algo_data[x_axis],
                        algo_data['lower_bound'],
                        algo_data['upper_bound'],
                        alpha=0.2
                    )

                plt.title(f'{env} - {metric}')  # TODO metric name
                plt.xlabel(x_axis)
                plt.ylabel(metric)  # TODO metric name
                # plt.xlim()  # TODO
                # plt.ylim()  # TODO
                plt.grid(True)
                plt.legend(title='Algorithm')
                plt.show()


def create_figures(output_dir, entity, project, fetch_data=True):
    env_filters = {
        "Hallway 2-steps": {
            ("kwargs", "env_fn", "kwargs", "env_name"): "MiniGrid-Hallway-v1",
            ("kwargs", "env_fn", "kwargs", "hallway_length"): 2,
        },
        # "Hallway 4-steps": {
        #     ("kwargs", "env_fn", "kwargs", "env_name"): "MiniGrid-Hallway-v1",
        #     ("kwargs", "env_fn", "kwargs", "hallway_length"): 4,
        # },
        # "Hallway 6-steps": {
        #     ("kwargs", "env_fn", "kwargs", "env_name"): "MiniGrid-Hallway-v1",
        #     ("kwargs", "env_fn", "kwargs", "hallway_length"): 6,
        # },
        # "FourRooms": {
        #     ("kwargs", "env_fn", "kwargs", "env_name"): "MiniGrid-FourRooms-v1",
        # },
    }
    algo_filters = {
        "SIERL": {
            ("kwargs", "agent", "kwargs", "goal_generator", "name"): "OmniGoalGenerator",
        },
    }
    metrics = [
        "test/0_success",
        "test_random_goals/0_success",
        "lateral/success",
    ]
    x_axis = "train_step"
    project_path = f"{entity}/{project}"
    data = load_run_data(env_filters, algo_filters, metrics, x_axis, project_path, fetch_data)

    plot_data(data, x_axis, output_dir)


# if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    "-o",
    type=Path,
    default="plots",
    help="Path to the directory to save the plots",
)
parser.add_argument(
    '--wandb-entity',
    '-n',
    type=str,
    default="",
    help="Name of the W&B entity that contains the project with the logged runs",
)
parser.add_argument(
    '--wandb-project',
    '-p',
    type=str,
    default="Experiments",
    help="The W&B project with the logged runs",
)
parser.add_argument(
    '--no-fetch-data',
    action='store_true',
    help="Do not fetch logged data",
)
args = parser.parse_args()
create_figures(
    output_dir=args.output_dir,
    entity=args.wandb_entity,
    project=args.wandb_project,
    fetch_data=False,  # (not args.no_fetch_data),
)
