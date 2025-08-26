import logging
import copy
import ale_py
import numpy as np
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from functools import partial

# Imports needed to register experiment components
import agents
import envs
import replays
import runners
import wandb_logger
from hive.main import main

import argparse
from hive.runners.utils import load_config
# from hive.utils import utils
from hive.runners import get_runner


logging.basicConfig(
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to base config file.")
    parser.add_argument(
        "-p",
        "--preset-config",
        help="Path to preset base config in the RLHive repository. These are relative "
        "to the hive/configs/ folder in the repository. For example, the Atari DQN "
        "config would be atari/dqn.yml.",
    )
    parser.add_argument(
        "-a",
        "--agent-config",
        help="Path to the agent config. Overrides settings in base config.",
    )
    parser.add_argument(
        "-e",
        "--env-config",
        help="Path to environment configuration file. Overrides settings in base "
        "config.",
    )
    parser.add_argument(
        "-l",
        "--logger-config",
        help="Path to logger configuration file. Overrides settings in base config.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Whether to resume the experiment from given experiment directory",
    )
    return parser.parse_known_args()[0]


def error(values, relative_weight=0.9):
    return relative_weight * np.mean(values) + (1 - relative_weight) * np.var(values)

def objective(trial, config):
    goal_generator_config = config["kwargs"]["agent"]["kwargs"]["goal_generator"]["kwargs"]
    w_n = trial.suggest_float("w_n", 1.0, 2.0)
    w_c = trial.suggest_float("w_c", 0.5, 1.0)
    w_g = trial.suggest_float("w_g", 0.5, 1.0)
    goal_generator_config["weights"] = [w_n, 0, w_c, w_g]
    goal_generator_config["max_familiarity"] = trial.suggest_float("max_familiarity", 0.9, 1.0)
    # goal_generator_config["frontier_proportion"] = 0.9
    # goal_generator_config["temperature"] = 0.5
    success = np.array([])
    success_random = np.array([])
    for seed in [18995728, 49789456, 50259734]:  # 71729146, 83575762
        config_new = copy.deepcopy(config)
        config_new["kwargs"]["seed"] = seed
        runner_fn, full_config = get_runner(config_new)
        runner = runner_fn()
        runner.register_config(full_config)

        runner.run_training()
        run_success = np.array(runner.test_metrics[runner._agents[0]]["success"])
        run_success_random = np.array(runner.random_test_metrics[runner._agents[0]]["success"])
        success = np.concatenate([success, run_success])
        success_random = np.concatenate([success_random, run_success_random])
        # breakpoint()
    print(f"Run stats:\n  success: {success}\n  success_random: {success_random}")
    return error(1 - success) + error(1 - success_random)

def main():
    args = parse_args()
    if args.config is None and args.preset_config is None:
        raise ValueError("Config needs to be provided")
    config = load_config(
        args.config,
        args.preset_config,
        args.agent_config,
        args.env_config,
        args.logger_config,
    )

    config["kwargs"]["train_steps"] = min(25000, config["kwargs"]["train_steps"])  # max steps
    loggers = config["kwargs"]["loggers"]
    wandb_kwargs = None
    if loggers is not None:
        for logger in loggers:
            if logger["name"] == "WandbLogger":
                wandb_kwargs = copy.deepcopy(logger["kwargs"])
                wandb_kwargs["name"] = "optimizer"
        config["kwargs"]["loggers"] = None  # don"t log individual runs (BUG when logging)

    objective_fn = partial(objective, config=config)
    search_space = {
        "w_n": [1.0, 1.5, 2.0],
        "w_c": [0.5, 1.0],
        "w_g": [0.5, 1.0],
        "max_familiarity": [0.995, 1.0],
    }
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(study_name="Optimize SIERL (categorical)", sampler=sampler)
    if wandb_kwargs is not None:
        wandbc = WeightsAndBiasesCallback(metric_name="success", wandb_kwargs=wandb_kwargs)
        study.optimize(objective_fn, n_trials=150, callbacks=[wandbc])
    else:
        study.optimize(objective_fn, n_trials=150)


if __name__ == "__main__":
    main()
