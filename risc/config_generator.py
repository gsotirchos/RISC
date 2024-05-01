from dataclasses import dataclass, replace, field
from functools import partial
import itertools

PROJECT = "RF-IS"


@dataclass(frozen=True, order=True)
class RunInfo:
    runner: str = ""
    environment: str = ""
    agent: str = ""
    experiment_name: str = ""
    reset_free: bool = False
    eval_every: bool = False
    train_steps: int = 200_000
    test_frequency: int = 1000
    test_episodes: int = 10
    test_max_steps: int = 100
    phase_step_limit: int = 100
    goal_switcher_kwargs: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.environment}-{self.agent}-{self.runner}"


def config_fn_decorator(config_fn):
    """Decorator that wraps a function that returns a config and instead returns
    a partial function that returns a config.
    """

    def generate_wrapped_config_fn(*args, **kwargs):
        return partial(config_fn, *args, **kwargs)

    return generate_wrapped_config_fn


@config_fn_decorator
def generate_dqn_agent_config(
    gc_agent=True,
    her_agent=False,
    rainbow_agent=False,
    replay_capacity=100000,
    **kwargs,
):
    agent_name = ""
    if her_agent:
        agent_name += "HER"
    if gc_agent:
        agent_name += "GoalConditioned"
    if rainbow_agent:
        agent_name += "Rainbow"
    agent_name += "DQNAgent"
    if her_agent:
        replay_buffer = "HERReplayBuffer"
    elif rainbow_agent:
        replay_buffer = "PrioritizedReplayBuffer"
    else:
        replay_buffer = "CircularReplayBuffer"
    config = {
        "name": agent_name,
        "kwargs": {
            "representation_net": {
                "name": "ConvNetwork",
                "kwargs": {
                    "channels": [16, 16, 16],
                    "kernel_sizes": 3,
                    "mlp_layers": [64],
                },
            },
            "epsilon_schedule": {
                "name": "LinearSchedule",
                "kwargs": {"init_value": 1.0, "end_value": 0.1, "steps": 10000},
            },
            "replay_buffer": {
                "name": replay_buffer,
                "kwargs": {"capacity": replay_capacity},
            },
            "target_net_update_schedule": {
                "name": "PeriodicSchedule",
                "kwargs": {"off_value": False, "on_value": True, "period": 500},
            },
            "min_replay_history": 512,
            "discount_rate": 0.95,
            "loss_fn": {"name": "MSELoss"},
            "device": "cuda",
            "batch_size": 128,
            **kwargs,
        },
    }
    if rainbow_agent:
        config["kwargs"].update(
            {
                "double": True,
                "dueling": True,
                "distributional": True,
                "noisy": False,
                "use_eps_greedy": True,
                "v_min": 0,
                "v_max": 2,
            }
        )
    return config


def sac_trunk_architecture():
    return {
        "name": "Sequential",
        "kwargs": {
            "modules": [
                {"name": "Linear", "kwargs": {"out_features": 50}},
                {"name": "LayerNorm", "kwargs": {"normalized_shape": 50}},
                {"name": "Tanh"},
            ],
        },
    }


@config_fn_decorator
def generate_sac_agent_config(
    gc_agent=True,
    replay_capacity=10000000,
    reward_scale_factor=10.0,
    critic_loss_weight=0.5,
    add_trunk_architecture=False,
    **kwargs,
):
    agent_name = "GoalConditionedSACAgent" if gc_agent else "SACAgent"
    config = {
        "name": agent_name,
        "kwargs": {
            "actor_trunk_net": sac_trunk_architecture()
            if add_trunk_architecture
            else None,
            "critic_trunk_net": sac_trunk_architecture()
            if add_trunk_architecture
            else None,
            "actor_net": {
                "name": "MLPNetwork",
                "kwargs": {
                    "hidden_units": [256, 256],
                },
            },
            "critic_net": {
                "name": "MLPNetwork",
                "kwargs": {
                    "hidden_units": [256, 256],
                },
            },
            "actor_optimizer_fn": {"name": "Adam", "kwargs": {"lr": 3e-4}},
            "critic_optimizer_fn": {"name": "Adam", "kwargs": {"lr": 3e-4}},
            "alpha_optimizer_fn": {"name": "Adam", "kwargs": {"lr": 3e-4}},
            "init_fn": {"name": "xavier_uniform"},
            "target_entropy_scale": 0.5,
            "reward_scale_factor": reward_scale_factor,
            "critic_loss_weight": critic_loss_weight,
            "soft_update_fraction": 0.005,
            "policy_update_frequency": 1,
            "discount_rate": 0.99,
            "batch_size": 256,
            "min_replay_history": 10000,
            "replay_buffer": {
                "name": "CircularReplayBuffer",
                "kwargs": {"capacity": replay_capacity},
            },
            "device": "cuda",
            **kwargs,
        },
    }

    return config


@config_fn_decorator
def generate_sac_uncertainty_agent_config(
    replay_capacity=10000000,
    reward_scale_factor=10.0,
    critic_loss_weight=0.5,
    n_heads=1,
    mc_dropout=False,
    mc_dropout_p=0.5,
    mc_samples=1,
    uncertainty_aggregation="std",
    min_aggregation=True,
    **kwargs,
):
    agent_name = "GoalConditionedSACUncertaintyAgent"
    config = {
        "name": agent_name,
        "kwargs": {
            # "actor_trunk_net": sac_trunk_architecture(),
            # "critic_trunk_net": sac_trunk_architecture(),
            "actor_net": {
                "name": "MLPNetwork",
                "kwargs": {
                    "hidden_units": [256, 256],
                },
            },
            "critic_net": {
                "name": "MLPNetwork",
                "kwargs": {
                    "hidden_units": [256, 256],
                },
            },
            "actor_optimizer_fn": {"name": "Adam", "kwargs": {"lr": 3e-4}},
            "critic_optimizer_fn": {"name": "Adam", "kwargs": {"lr": 3e-4}},
            "alpha_optimizer_fn": {"name": "Adam", "kwargs": {"lr": 3e-4}},
            "init_fn": {"name": "xavier_uniform"},
            "target_entropy_scale": 0.5,
            "reward_scale_factor": reward_scale_factor,
            "critic_loss_weight": critic_loss_weight,
            "soft_update_fraction": 0.005,
            "policy_update_frequency": 1,
            "discount_rate": 0.99,
            "batch_size": 256,
            "min_replay_history": 10000,
            "replay_buffer": {
                "name": "CircularReplayBuffer",
                "kwargs": {"capacity": replay_capacity},
            },
            "device": "cuda",
            "n_heads": n_heads,
            "mc_dropout": mc_dropout,
            "mc_dropout_p": mc_dropout_p,
            "mc_samples": mc_samples,
            "uncertainty_aggregation": uncertainty_aggregation,
            "min_aggregation": min_aggregation,
            **kwargs,
        },
    }
    return config


@config_fn_decorator
def generate_runner_config(
    run_info: RunInfo,
    seed=0,
    stack_size=1,
    early_terminal=True,
    send_truncation=False,
):
    runner_name = "rf" if run_info.reset_free else "epi"
    if early_terminal:
        runner_name += "_et"
    if send_truncation:
        runner_name += "_st"
    run_info = replace(run_info, runner=runner_name)
    config = {
        "name": "ResetFreeRunner" if run_info.reset_free else "SingleAgentRunner",
        "kwargs": {
            "experiment_manager": {
                "name": "Experiment",
                "kwargs": {
                    "name": str(run_info),
                    "save_dir": "experiment",
                    "saving_schedule": {
                        # "name": "ConstantSchedule",
                        # "kwargs": {"value": False},
                        "name": "PeriodicSchedule",
                        "kwargs": {
                            "off_value": False,
                            "on_value": True,
                            "period": 1000000,
                        },
                    },
                },
            },
            "train_steps": run_info.train_steps,
            "test_frequency": run_info.test_frequency,
            "test_episodes": run_info.test_episodes,
            "test_max_steps": run_info.test_max_steps,
            "seed": seed,
            "stack_size": stack_size,
            "eval_every": run_info.eval_every,
            "early_terminal": early_terminal,
            "send_truncation": send_truncation,
        },
    }
    if not run_info.reset_free:
        config["kwargs"]["max_steps_per_episode"] = run_info.test_max_steps
    return config, run_info


@config_fn_decorator
def generate_logger_config(run_info: RunInfo, type="aim", tags=[], notes=""):
    if type == "aim":
        return [
            {
                "name": "AimLogger",
                "kwargs": {
                    "project_name": PROJECT,
                    "name": str(run_info),
                    "tags": tags,
                    "experiment_name": run_info.experiment_name,
                },
            }
        ]
    elif type == "comet":
        return [
            {
                "name": "CometLogger",
                "kwargs": {
                    "project_name": PROJECT,
                    "name": str(run_info),
                    "tags": tags,
                },
            }
        ]
    elif type == "wandb":
        return [
            {
                "name": "WandbLogger",
                "kwargs": {
                    "project": PROJECT,
                    "name": str(run_info),
                    "resume": "allow",
                    "start_method": "fork",
                    "tags": tags,
                    "notes": notes,
                },
            }
        ]


@config_fn_decorator
def generate_minigrid_env_config(
    run_info: RunInfo,
    rnc: str,
    environment_name,
    train_steps=100_000,
    symbolic=True,
    video_period=1000,
    local_vis_period=2000,
    vis_frequency=2000,
    test_max_steps=100,
    reset_free=True,
    test_frequency=1000,
    test_episodes=10,
    eval_every=False,
):
    run_info = replace(
        run_info,
        environment=rnc,
        reset_free=reset_free,
        eval_every=eval_every,
        train_steps=train_steps,
        test_frequency=test_frequency,
        test_episodes=test_episodes,
        test_max_steps=test_max_steps,
    )
    if "FourRooms" in environment_name:
        return {
            "name": "minigrid_envs",
            "kwargs": {
                "env_name": "MiniGrid-FourRooms-v1",
                "symbolic": symbolic,
                "video_period": video_period,
                "agent_pos": [1, 1],
                "goal_pos": [17, 17],
                "local_vis_period": local_vis_period,
                "vis_frequency": vis_frequency,
                "reset_free": reset_free,
                "train_max_steps": train_steps,
                "eval_max_steps": test_max_steps,
            },
        }, run_info
    elif "Empty" in environment_name:
        return {
            "name": "minigrid_envs",
            "kwargs": {
                "env_name": "MiniGrid-Empty-16x16-v1",
                "symbolic": symbolic,
                "video_period": video_period,
                "local_vis_period": local_vis_period,
                "vis_frequency": vis_frequency,
                "reset_free": reset_free,
                "train_max_steps": train_steps,
                "eval_max_steps": test_max_steps,
            },
        }, run_info
    elif "TwoRooms" in environment_name:
        return {
            "name": "minigrid_envs",
            "kwargs": {
                "env_name": "MiniGrid-TwoRooms-v1",
                "symbolic": symbolic,
                "video_period": video_period,
                "agent_pos": [1, 1],
                "goal_pos": [17, 8],
                "local_vis_period": local_vis_period,
                "vis_frequency": vis_frequency,
                "reset_free": reset_free,
                "train_max_steps": train_steps,
                "eval_max_steps": test_max_steps,
            },
        }, run_info
    else:
        raise ValueError(f"{environment_name} not supported")


@config_fn_decorator
def generate_earl_env_config(
    run_info: RunInfo,
    rnc: str,
    env_name,
    reward_type: str = "sparse",
    reset_train_env_at_goal: bool = False,
    setup_as_lifelong_learning: bool = False,
    distance_type: str = "l2_cluster",
    reset_free: bool = True,
    test_frequency=10000,
    test_episodes=10,
    debug=False,
    **kwargs,
):
    if debug:
        train_steps = {
            "tabletop_manipulation": 1_000_000,
            "sawyer_door": 1_000_000,
            "sawyer_peg": 1_500_000,
            "kitchen": 1_000_000,
            # "dhand_manipulation": 10_000_000,
            "minitaur": 500_000,
        }
    else:
        train_steps = {
            "tabletop_manipulation": 3_000_000,
            "sawyer_door": 5_000_000,
            "sawyer_peg": 8_000_000,
            "kitchen": 5_000_000,
            # "dhand_manipulation": 10_000_000,
            "minitaur": 5_000_000,
        }

    phase_step_limits = {
        "tabletop_manipulation": 200,
        "sawyer_door": 300,
        "sawyer_peg": 200,
        "kitchen": 400,
        # "dhand_manipulation": 10_000_000,
        "minitaur": 1000,
    }

    goal_switcher_kwargs = {
        "tabletop_manipulation": {
            "switching_probability": 0.01,
            "start": 20,
            "end": 200,
            "num_steps": 100_000,
        },
        "sawyer_door": {
            "switching_probability": 0.005,
            "start": 30,
            "end": 300,
            "num_steps": 500_000,
        },
        "sawyer_peg": {
            "switching_probability": 0.01,
            "start": 20,
            "end": 200,
            "num_steps": 2_000_000,
        },
        "kitchen": {
            "switching_probability": 0.004,
            "start": 40,
            "end": 400,
            "num_steps": 3_000_000,
        },
        "minitaur": {
            "switching_probability": 0.002,
            "start": 100,
            "end": 1000,
            "num_steps": 1_000_000,
        },
    }
    run_info = replace(
        run_info,
        environment=rnc,
        reset_free=reset_free,
        eval_every=False,
        train_steps=train_steps[env_name],
        test_frequency=test_frequency,
        test_episodes=test_episodes,
        test_max_steps=phase_step_limits[env_name],
        phase_step_limit=phase_step_limits[env_name],
        goal_switcher_kwargs=goal_switcher_kwargs[env_name],
    )
    env_config = {
        "name": "earl_envs",
        "kwargs": {
            "env_name": env_name,
            "reward_type": reward_type,
            "reset_train_env_at_goal": reset_train_env_at_goal,
            "setup_as_lifelong_learning": setup_as_lifelong_learning,
            "distance_type": distance_type,
            "reset_free": reset_free,
        },
    }
    env_config["kwargs"].update(kwargs)
    return env_config, run_info


@config_fn_decorator
def generate_goal_generator_fn(
    run_info: RunInfo, generator_name, use_uncertainty_difference=True
):
    if generator_name == "fb":
        run_info = replace(run_info, agent=run_info.agent + "_fb")
        return {
            "name": "FBGoalGenerator",
        }, run_info
    elif generator_name == "stable":
        run_info = replace(run_info, agent=run_info.agent + "_fbstable")
        return {
            "name": "StableFBGoalGenerator",
        }, run_info
    elif generator_name == "vaprl":
        run_info = replace(run_info, agent=run_info.agent + "_vaprl")
        return {
            "name": "VapRLGoalGenerator",
        }, run_info
    elif generator_name == "uncertainty":
        run_info = replace(run_info, agent=run_info.agent + "_unc")
        return {
            "name": "UncertaintyBasedGenerator",
            "kwargs": {
                "score_difference": use_uncertainty_difference,
            },
        }, run_info


@config_fn_decorator
def generate_gc_agent_config(
    run_info: RunInfo,
    base_agent_fn,
    goal_generator_fn,
    goal_switcher="BasicGoalSwitcher",
    goal_switcher_kwargs={},
    replay_capacity=10000000,
    n_sampled_goals=0,
    prioritized_replay_buffer=False,
    **kwargs,
):
    goal_generator_config, run_info = goal_generator_fn(run_info)
    if prioritized_replay_buffer:
        replay_config = {
            "name": "PrioritizedReplayBuffer",
            "kwargs": {"capacity": replay_capacity},
        }
    else:
        replay_config = {
            "name": "RelabelingReplayBuffer",
            "kwargs": {
                "capacity": replay_capacity,
                "n_sampled_goals": n_sampled_goals,
                "dense_batch_relabeling": False,
            },
        }
    gs_kwargs = run_info.goal_switcher_kwargs
    gs_kwargs.update(goal_switcher_kwargs)
    return {
        "name": "GCResetFree",
        "kwargs": {
            "base_agent": base_agent_fn(),
            "goal_generator": goal_generator_config,
            "phase_step_limit": run_info.phase_step_limit,
            "replay_buffer": replay_config,
            "goal_switcher": {"name": goal_switcher, "kwargs": gs_kwargs},
            **kwargs,
        },
    }, run_info


@config_fn_decorator
def generate_is_agent_config(
    run_info: RunInfo,
    base_agent_fn,
    replay_capacity=10000000,
    n_sampled_goals=0,
    prioritized_replay_buffer=False,
    bootstrap_strategy=2,
    switching_strategy=2,
    separate_agents=False,
    keep_consistent_goal=False,
    **kwargs,
):
    # goal_generator_config, run_info = goal_generator_fn(run_info)
    if prioritized_replay_buffer:
        replay_config = {
            "name": "PrioritizedReplayBuffer",
            "kwargs": {"capacity": replay_capacity},
        }
    else:
        replay_config = {
            "name": "RelabelingReplayBuffer",
            "kwargs": {
                "capacity": replay_capacity,
                "n_sampled_goals": n_sampled_goals,
                "dense_batch_relabeling": False,
            },
        }
    return {
        "name": "IntelligentSwitchingAgent",
        "kwargs": {
            "base_agent": base_agent_fn(),
            "phase_step_limit": run_info.phase_step_limit,
            "replay_buffer": replay_config,
            "bootstrap_strategy": bootstrap_strategy,
            "switching_strategy": switching_strategy,
            "separate_agents": separate_agents,
            "keep_consistent_goal": keep_consistent_goal,
            **kwargs,
        },
    }, run_info


@config_fn_decorator
def generate_agent_config(
    run_info: RunInfo,
    agent_name: str,
    agent,
    base_config_fn,
    sample_fn_type="demos",
    goal_batch_size=256,
):
    # agent_name_map = {
    #     "episodic": "epi",
    #     "fb": "fb",
    #     "paired": "paired",
    #     "paired_gc": "paired_gc",
    #     "gc": "gcrf",
    # }
    run_info = replace(run_info, agent=agent_name)
    phase_step_limit = run_info.phase_step_limit
    if agent == "episodic":
        return base_config_fn(), run_info
    elif agent == "gc":
        return base_config_fn(run_info=run_info)
    elif agent == "fb":
        return {
            "name": "ForwardBackwardAgent",
            "kwargs": {
                "forward_agent": base_config_fn(),
                "reset_agent": base_config_fn(),
                "phase_step_limit": phase_step_limit,
            },
        }, run_info
    elif agent == "paired":
        return {
            "name": "PairedAgent",
            "kwargs": {
                "phase_step_limit": phase_step_limit,
                "protagonist": base_config_fn(),
                "antagonist": base_config_fn(),
                "adversary": base_config_fn(),
            },
        }, run_info
    elif agent == "paired_gc":
        # run_info = replace(run_info, agent=run_info.agent + f"_{sample_fn_type}")
        return {
            "name": "PairedGCAgent",
            "kwargs": {
                "phase_step_limit": phase_step_limit,
                "protagonist": base_config_fn(),
                "antagonist": base_config_fn(),
                "adversary": base_config_fn(),
                "goal_generator": {"name": "ScoreBasedGenerator"},
                "sample_fn_type": sample_fn_type,
                "goal_batch_size": goal_batch_size,
            },
        }, run_info
    else:
        raise ValueError(f"{agent} not currently supported")


def build_seeded_configs(
    runner_config_fn,
    agent_config_fn,
    env_config_fn,
    logger_config_fn,
    tags=[],
    notes="",
    experiment_name="",
    num_seeds=5,
    seed_start=0,
):
    for seed in range(seed_start, seed_start + num_seeds):
        run_info = RunInfo(experiment_name=experiment_name)
        env_config, run_info = env_config_fn(run_info=run_info)
        agent_config, run_info = agent_config_fn(run_info=run_info)
        runner_config, run_info = runner_config_fn(run_info=run_info, seed=seed)
        logger_config = logger_config_fn(run_info=run_info, tags=tags, notes=notes)

        runner_config["kwargs"]["env_fn"] = env_config
        runner_config["kwargs"]["agent"] = agent_config
        runner_config["kwargs"]["loggers"] = logger_config
        yield runner_config


def generate_all_configs_from_names(
    runner_names,
    agent_names,
    env_names,
    logger_names,
    tags=[],
    notes="",
    experiment_name="",
    num_seeds=5,
):
    for runner_name in runner_names:
        for env_name in env_names:
            for agent_name in agent_names:
                for logger_name in logger_names:
                    configs = build_configs_from_names(
                        runner_name=runner_name,
                        env_name=env_name,
                        agent_name=agent_name,
                        logger_name=logger_name,
                        tags=tags,
                        notes=notes,
                        experiment_name=experiment_name,
                        num_seeds=num_seeds,
                    )
                    for config in configs:
                        yield config


def generate_all_configs(
    runner_configs,
    agent_configs,
    env_configs,
    logger_configs,
    tags=[],
    experiment_name="",
    num_seeds=5,
):
    for runner_config in runner_configs:
        for env_config in env_configs:
            for agent_config in agent_configs:
                for logger_config in logger_configs:
                    if isinstance(runner_config, str):
                        runner_config = runner_config_fns[runner_config]
                    if isinstance(agent_config, str):
                        agent_config = agent_config_fns[agent_config]
                    if isinstance(env_config, str):
                        env_config = env_config_fns[env_config]
                    if isinstance(logger_config, str):
                        logger_config = logger_config_fns[logger_config]

                    configs = build_seeded_configs(
                        runner_config_fn=runner_config,
                        agent_config_fn=agent_config,
                        env_config_fn=env_config,
                        logger_config_fn=logger_config,
                        tags=tags,
                        num_seeds=num_seeds,
                        experiment_name=experiment_name,
                    )
                    for config in configs:
                        yield config


# missing dhand_manipulation
env_config_fns = {
    "tabletop_manipulation_rf": generate_earl_env_config(
        rnc="tm", env_name="tabletop_manipulation"
    ),
    "tabletop_manipulation_rf_debug": generate_earl_env_config(
        rnc="tm_debug", env_name="tabletop_manipulation", debug=True
    ),
    "tabletop_manipulation_epi": generate_earl_env_config(
        rnc="tm", env_name="tabletop_manipulation", reset_free=False
    ),
    "sawyer_door_rf": generate_earl_env_config(rnc="sd", env_name="sawyer_door"),
    "sawyer_door_rf_debug": generate_earl_env_config(
        rnc="sd_debug", env_name="sawyer_door", debug=True
    ),
    "sawyer_door_epi": generate_earl_env_config(
        rnc="sd", env_name="sawyer_door", reset_free=False
    ),
    "sawyer_peg_rf": generate_earl_env_config(rnc="sp", env_name="sawyer_peg"),
    "sawyer_peg_rf_debug": generate_earl_env_config(
        rnc="sp_debug", env_name="sawyer_peg", debug=True
    ),
    "sawyer_peg_epi": generate_earl_env_config(
        rnc="sp", env_name="sawyer_peg", reset_free=False
    ),
    "minitaur_rf": generate_earl_env_config(
        rnc="minitaur", env_name="minitaur", reward_type="dense"
    ),
    "minitaur_rf_debug": generate_earl_env_config(
        rnc="minitaur_debug", env_name="minitaur", reward_type="dense", debug=True
    ),
    "minitaur_epi": generate_earl_env_config(
        rnc="minitaur_epi", env_name="minitaur", reward_type="dense", reset_free=False
    ),
    "kitchen_rf": generate_earl_env_config(
        rnc="fk", env_name="kitchen", reward_type="dense"
    ),
    "kitchen_rf_debug": generate_earl_env_config(
        rnc="fk_debug", env_name="kitchen", reward_type="dense", debug=True
    ),
    "kitchen_epi": generate_earl_env_config(
        rnc="fk", env_name="kitchen", reward_type="dense", reset_free=False
    ),
    "four_rooms_rf": generate_minigrid_env_config(
        rnc="4r", environment_name="FourRooms", train_steps=50000
    ),
    "four_rooms_rf_every": generate_minigrid_env_config(
        rnc="4r", environment_name="FourRooms", eval_every=True
    ),
    "four_rooms_rf_debug": generate_minigrid_env_config(
        rnc="4r_debug", environment_name="FourRooms", eval_every=True, train_steps=50000
    ),
    "four_rooms_epi": generate_minigrid_env_config(
        rnc="4r", environment_name="FourRooms", reset_free=False
    ),
    "two_rooms_rf": generate_minigrid_env_config(rnc="2r", environment_name="TwoRooms"),
    "two_rooms_epi": generate_minigrid_env_config(
        rnc="2r", environment_name="TwoRooms", reset_free=False
    ),
    "empty_rf": generate_minigrid_env_config(rnc="16e", environment_name="Empty"),
    "empty_epi": generate_minigrid_env_config(
        rnc="16e", environment_name="Empty", reset_free=False
    ),
    "microwave_rf": generate_earl_env_config(
        rnc="microwave",
        env_name="kitchen",
        reward_type="dense",
        kitchen_task="microwave",
    ),
    "light_switch_rf": generate_earl_env_config(
        rnc="light_switch",
        env_name="kitchen",
        reward_type="dense",
        kitchen_task="light_switch",
    ),
    "micro_light_rf": generate_earl_env_config(
        rnc="micro_light",
        env_name="kitchen",
        reward_type="dense",
        kitchen_task="micro_light",
    ),
}
for kitchen_task in [
    "microwave",
    "light_switch",
    "slide_cabinet",
    "hinge_cabinet",
    "micro_slide",
    "micro_light",
    "light_slide",
    "light_hinge",
    "slide_hinge",
]:
    env_config_fns[f"{kitchen_task}_rf"] = generate_earl_env_config(
        rnc=kitchen_task,
        env_name="kitchen",
        reward_type="dense",
        kitchen_task=kitchen_task,
    )

runner_config_fns = {
    "rf": generate_runner_config(),
    "epi": generate_runner_config(),
    "epi_no_term": generate_runner_config(early_terminal=False),
    "epi_time_trunc": generate_runner_config(
        early_terminal=False, send_truncation=True
    ),
    "epi_success_trunc": generate_runner_config(
        early_terminal=True, send_truncation=True
    ),
}
logger_config_fns = {
    "aim": generate_logger_config(type="aim"),
    "comet": generate_logger_config(type="comet"),
    "wandb": generate_logger_config(type="wandb"),
}
agent_config_fns = {
    "episodic_dqn": generate_agent_config(
        agent_name="episodic_dqn",
        agent="episodic",
        base_config_fn=generate_dqn_agent_config,
    ),
    "fb_dqn": generate_agent_config(
        agent_name="fb_dqn", agent="fb", base_config_fn=generate_dqn_agent_config
    ),
    "gc_fb_dqn": generate_agent_config(
        agent_name="gc_fb_dqn",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            separate_agents=True,
            oracle=False,
        ),
    ),
    "gc_fb_dqn_gs": generate_agent_config(
        agent_name="gc_fb_dqn_gs",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="SuccessProbabilityGoalSwitcher",
            separate_agents=True,
        ),
    ),
    "gc_fb_dqn_gs2": generate_agent_config(
        agent_name="gc_fb_dqn_gs2",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="SuccessProbabilityGoalSwitcher",
            separate_agents=True,
            goal_switcher_kwargs={"trajectory_proportion": 0.5},
        ),
    ),
    "gc_fb_dqn_gs_sb": generate_agent_config(
        agent_name="gc_fb_dqn_gs_sb",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={"switch_on_backward": False},
        ),
    ),
    "gc_fb_dqn_gs_sf": generate_agent_config(
        agent_name="gc_fb_dqn_gs_sf",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={"switch_on_forward": False},
        ),
    ),
    "gc_fb_dqn_gsr_sb": generate_agent_config(
        agent_name="gc_fb_dqn_gsr_sb",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="RandomGoalSwitcher",
            goal_switcher_kwargs={"switch_on_backward": False},
        ),
    ),
    "gc_fb_dqn_gsr_sf": generate_agent_config(
        agent_name="gc_fb_dqn_gsr_sf",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="RandomGoalSwitcher",
            goal_switcher_kwargs={"switch_on_forward": False},
        ),
    ),
    "gc_fb_dqn_gsr": generate_agent_config(
        agent_name="gc_fb_dqn_gsr",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="RandomGoalSwitcher",
        ),
    ),
    "gc_fb_dqn_gs_sched": generate_agent_config(
        agent_name="gc_fb_dqn_gs_sched",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="ScheduleGoalSwitcher",
            goal_switcher_kwargs={"start": 5, "end": 50, "num_steps": 10000},
        ),
    ),
    "gc_fb_dqn_pri": generate_agent_config(
        agent_name="gc_fb_dqn_pri",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            prioritized_replay_buffer=True,
        ),
    ),
    "gc_fb_dqn_gs_pri": generate_agent_config(
        agent_name="gc_fb_dqn_gs_pri",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="SuccessProbabilityGoalSwitcher",
            prioritized_replay_buffer=True,
        ),
    ),
    "paired_dqn": generate_agent_config(
        agent_name="paired_dqn",
        agent="paired",
        base_config_fn=generate_dqn_agent_config,
    ),
    "paired_gc_dqn": generate_agent_config(
        agent_name="paired_gc_dqn",
        agent="paired_gc",
        base_config_fn=generate_dqn_agent_config,
    ),
    "episodic_sac": generate_agent_config(
        agent_name="episodic_sac",
        agent="episodic",
        base_config_fn=generate_sac_agent_config(),
    ),
    "episodic_sac_tat": generate_agent_config(
        agent_name="episodic_sac",
        agent="episodic",
        base_config_fn=generate_sac_agent_config(trunc_as_terminal=True),
    ),
    "episodic_sac_rf1": generate_agent_config(
        agent_name="episodic_sac_rf1",
        agent="episodic",
        base_config_fn=generate_sac_agent_config(reward_scale_factor=1.0),
    ),
    "fb_sac": generate_agent_config(
        agent_name="fb_sac",
        agent="fb",
        base_config_fn=generate_sac_agent_config(),
    ),
    "paired_gc_sac_demos": generate_agent_config(
        agent_name="paired_gc_sac_demos",
        agent="paired_gc",
        base_config_fn=generate_sac_agent_config(),
        sample_fn_type="demos",
    ),
    "paired_gc_sac_replay": generate_agent_config(
        agent_name="paired_gc_sac_replay",
        agent="paired_gc",
        base_config_fn=generate_sac_agent_config(),
        sample_fn_type="replay",
    ),
    "gc_fb_sac": generate_agent_config(
        agent_name="gc_fb_sac",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        ),
    ),
    "gc_fb_sac_log": generate_agent_config(
        agent_name="gc_fb_sac_log",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            log_success=True,
        ),
    ),
    "gc_fb_sac_to": generate_agent_config(
        agent_name="gc_fb_sac_to",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            use_termination_signal=False,
        ),
    ),
    "gc_fb_sac_to_sep": generate_agent_config(
        agent_name="gc_fb_sac_to_sep",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            use_termination_signal=False,
            separate_agents=True,
        ),
    ),
    "gc_fb_sac_sep": generate_agent_config(
        agent_name="gc_fb_sac_sep",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            separate_agents=True,
        ),
    ),
    "gc_fb_sac_rf1": generate_agent_config(
        agent_name="gc_fb_sac_rf1",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(reward_scale_factor=1.0),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        ),
    ),
    "gc_fb_sac_sc": generate_agent_config(
        agent_name="gc_fb_sac_sc",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=False),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        ),
    ),
    "gc_fb_sac2": generate_agent_config(
        agent_name="gc_fb_sac2",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        ),
    ),
    "gc_vaprl_sac": generate_agent_config(
        agent_name="gc_vaprl_sac",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(use_value_comp_critic=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="vaprl"),
            n_sampled_goals=4,
            distance_type="timestep",
        ),
    ),
    "gc_vaprl_bug_sac": generate_agent_config(
        agent_name="gc_vaprl_bug_sac",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(
                use_value_comp_critic=True, trunc_as_terminal=True
            ),
            goal_generator_fn=generate_goal_generator_fn(generator_name="vaprl"),
            n_sampled_goals=4,
            distance_type="timestep",
        ),
    ),
    "gc_unipaired_ma": generate_agent_config(
        agent_name="gc_unipaired_ma",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_uncertainty_agent_config(
                n_heads=2,
                uncertainty_aggregation="paired",
                min_aggregation=True,
            ),
            goal_generator_fn=generate_goal_generator_fn(
                generator_name="uncertainty", use_uncertainty_difference=False
            ),
            n_sampled_goals=4,
            distance_type="l2_cluster",
        ),
    ),
    "gc_unipaired_na": generate_agent_config(
        agent_name="gc_unipaired_na",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_uncertainty_agent_config(
                n_heads=2,
                uncertainty_aggregation="paired",
                min_aggregation=False,
            ),
            goal_generator_fn=generate_goal_generator_fn(
                generator_name="uncertainty", use_uncertainty_difference=False
            ),
            n_sampled_goals=4,
            distance_type="l2_cluster",
        ),
    ),
    "gc_unc_bs2_ud_bgs": generate_agent_config(
        agent_name="gc_unc_bs2_ud_bgs",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_uncertainty_agent_config(
                n_heads=10,
                uncertainty_aggregation="bootstrap",
                min_aggregation=False,
            ),
            goal_generator_fn=generate_goal_generator_fn(
                generator_name="uncertainty", use_uncertainty_difference=True
            ),
            goal_switcher="BasicGoalSwitcher",
            n_sampled_goals=4,
            distance_type="l2_cluster",
        ),
    ),
    "gc_unc_bs2_ud_ugs": generate_agent_config(
        agent_name="gc_unc_bs2_ud_ugs",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_uncertainty_agent_config(
                n_heads=10,
                uncertainty_aggregation="bootstrap",
                min_aggregation=False,
            ),
            goal_generator_fn=generate_goal_generator_fn(
                generator_name="uncertainty", use_uncertainty_difference=True
            ),
            goal_switcher="UncertaintyGoalSwitcher",
            n_sampled_goals=4,
            distance_type="l2_cluster",
        ),
    ),
    "gc_unc_bs2_us_bgs": generate_agent_config(
        agent_name="gc_unc_bs2_us_bgs",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_uncertainty_agent_config(
                n_heads=10,
                uncertainty_aggregation="bootstrap",
                min_aggregation=False,
            ),
            goal_generator_fn=generate_goal_generator_fn(
                generator_name="uncertainty", use_uncertainty_difference=False
            ),
            goal_switcher="BasicGoalSwitcher",
            n_sampled_goals=4,
            distance_type="l2_cluster",
        ),
    ),
    "gc_unc_bs2_us_ugs": generate_agent_config(
        agent_name="gc_unc_bs2_us_ugs",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_uncertainty_agent_config(
                n_heads=10,
                uncertainty_aggregation="bootstrap",
                min_aggregation=False,
            ),
            goal_generator_fn=generate_goal_generator_fn(
                generator_name="uncertainty", use_uncertainty_difference=False
            ),
            goal_switcher="UncertaintyGoalSwitcher",
            n_sampled_goals=4,
            distance_type="l2_cluster",
        ),
    ),
    "buggy_gc_fb_sac": generate_agent_config(
        agent_name="buggy_gc_fb_sac",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(trunc_as_terminal=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        ),
    ),
    "gc_fb_no_term_sac": generate_agent_config(
        agent_name="gc_fb_no_term_sac",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            use_early_terminal=False,
        ),
    ),
    "gc_fb_sac_gs": generate_agent_config(
        agent_name="gc_fb_sac_gs",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
        ),
    ),
    "gc_fb_sac_gs_r": generate_agent_config(
        agent_name="gc_fb_sac_gs_r",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=False),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="RandomGoalSwitcher",
        ),
    ),
    "gc_fb_sac_gs_sched": generate_agent_config(
        agent_name="gc_fb_sac_gs_sched",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=False),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="ScheduleGoalSwitcher",
        ),
    ),
    "gc_fb_sac_gs_minitaur1": generate_agent_config(
        agent_name="gc_fb_sac_gs1",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.0,
                "minimum_steps": 750,
                "trajectory_proportion": 0.2,
            },
        ),
    ),
    "gc_fb_sac_gs_minitaur2": generate_agent_config(
        agent_name="gc_fb_sac_gs2",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.0,
                "minimum_steps": 0,
                "trajectory_proportion": 0.5,
            },
        ),
    ),
    "gc_fb_sac_gs_minitaur3": generate_agent_config(
        agent_name="gc_fb_sac_gs4",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.95,
                "minimum_steps": 500,
                "trajectory_proportion": 0.2,
            },
        ),
    ),
    "gc_fb_sac_gs_minitaur3_log": generate_agent_config(
        agent_name="gc_fb_sac_gs4_log",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.95,
                "minimum_steps": 500,
                "trajectory_proportion": 0.2,
            },
            log_success=True,
        ),
    ),
    "gc_fb_sac_gs_nt_minitaur3": generate_agent_config(
        agent_name="gc_fb_sac_nt_gs4",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.95,
                "minimum_steps": 500,
                "trajectory_proportion": 0.2,
            },
            use_termination_signal=False,
        ),
    ),
    "gc_fb_sac_gs_sd1": generate_agent_config(
        agent_name="gc_fb_sac_gs1",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.0,
                "minimum_steps": 0,
                "trajectory_proportion": 0.5,
            },
        ),
    ),
    "gc_fb_sac_gs_sd2": generate_agent_config(
        agent_name="gc_fb_sac_gs2",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.0,
                "minimum_steps": 150,
                "trajectory_proportion": 1.0,
            },
        ),
    ),
    "gc_fb_sac_gs_sd3": generate_agent_config(
        agent_name="gc_fb_sac_gs4",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.95,
                "minimum_steps": 225,
                "trajectory_proportion": 1.0,
            },
        ),
    ),
    "gc_fb_sac_gs_sd3_log": generate_agent_config(
        agent_name="gc_fb_sac_gs4_log",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.95,
                "minimum_steps": 225,
                "trajectory_proportion": 1.0,
            },
            log_success=True,
        ),
    ),
    "gc_fb_sac_gs_sp1": generate_agent_config(
        agent_name="gc_fb_sac_gs1",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.0,
                "minimum_steps": 100,
                "trajectory_proportion": 0.2,
            },
        ),
    ),
    "gc_fb_sac_gs_sp2": generate_agent_config(
        agent_name="gc_fb_sac_gs2",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.0,
                "minimum_steps": 150,
                "trajectory_proportion": 0.5,
            },
        ),
    ),
    "gc_fb_sac_gs_sp3": generate_agent_config(
        agent_name="gc_fb_sac_gs4",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.95,
                "minimum_steps": 100,
                "trajectory_proportion": 0.2,
            },
        ),
    ),
    "gc_fb_sac_gs_sp3_log": generate_agent_config(
        agent_name="gc_fb_sac_gs4_log",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.95,
                "minimum_steps": 100,
                "trajectory_proportion": 0.2,
            },
            log_success=True,
        ),
    ),
    "gc_fb_sac_gs_tm1": generate_agent_config(
        agent_name="gc_fb_sac_gs1",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.0,
                "minimum_steps": 50,
                "trajectory_proportion": 0.8,
            },
        ),
    ),
    "gc_fb_sac_gs_tm2": generate_agent_config(
        agent_name="gc_fb_sac_gs2",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.0,
                "minimum_steps": 100,
                "trajectory_proportion": 0.8,
            },
        ),
    ),
    "gc_fb_sac_gs_tm3": generate_agent_config(
        agent_name="gc_fb_sac_gs4",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.95,
                "minimum_steps": 100,
                "trajectory_proportion": 0.5,
            },
        ),
    ),
    "gc_fb_sac_gs_tm3_log": generate_agent_config(
        agent_name="gc_fb_sac_gs4_log",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(compute_success_probability=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            goal_switcher="SuccessProbabilityGoalSwitcher",
            goal_switcher_kwargs={
                "conservative_factor": 0.95,
                "minimum_steps": 100,
                "trajectory_proportion": 0.5,
            },
            log_success=True,
        ),
    ),
    "gc_fb_dqn_lc": generate_agent_config(
        agent_name="gc_fb_dqn_lc",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(only_add_low_confidence=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            separate_agents=True,
        ),
    ),
    "gc_fb_dqn_lcrc": generate_agent_config(
        agent_name="gc_fb_dqn_lcrc",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(only_add_low_confidence=True),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="ReverseCurriculumGoalSwitcher",
            separate_agents=True,
        ),
    ),
    "gc_fb_dqn_rc": generate_agent_config(
        agent_name="gc_fb_dqn_rc",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="ReverseCurriculumGoalSwitcher",
            separate_agents=True,
        ),
    ),
    "gc_fb_dqn_rco": generate_agent_config(
        agent_name="gc_fb_dqn_rco",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_dqn_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            replay_capacity=100000,
            goal_switcher="ReverseCurriculumGoalSwitcher",
            separate_agents=True,
            oracle=True,
        ),
    ),
    "gc_fb_sac_notrunc": generate_agent_config(
        agent_name="gc_fb_sac_notrunc",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
            never_truncate=True,
        ),
    ),
    "gc_fb_sac_stable": generate_agent_config(
        agent_name="gc_fb_sac",
        agent="gc",
        base_config_fn=generate_gc_agent_config(
            base_agent_fn=generate_sac_agent_config(),
            goal_generator_fn=generate_goal_generator_fn(generator_name="stable"),
        ),
    ),
}

for bootstrap_strategy in [0, 1, 2]:
    for switching_strategy in [0, 1, 2]:
        for separate_agents in [False, True]:
            for keep_consistent_goal in [False, True]:
                agent_config_fns[
                    f"is_bs{bootstrap_strategy}_ss{switching_strategy}_sa{separate_agents}_kg{keep_consistent_goal}"
                ] = generate_agent_config(
                    agent_name=f"is-bs_{bootstrap_strategy}_ss_{switching_strategy}_sa_{separate_agents}_kg_{keep_consistent_goal}",
                    agent="gc",
                    base_config_fn=generate_is_agent_config(
                        base_agent_fn=generate_sac_agent_config(),
                        bootstrap_strategy=bootstrap_strategy,
                        switching_strategy=switching_strategy,
                        separate_agents=separate_agents,
                        keep_consistent_goal=keep_consistent_goal,
                    ),
                )


for min_aggregation in [True, False]:
    for use_uncertainty_difference in [True, False]:
        for uncertainty_method in ["bs", "mc"]:
            for uncertainty_aggregation in ["std", "range"]:
                agent_name = (
                    f"gc_unc_{uncertainty_method}_{uncertainty_aggregation}"
                    + f"{'_ma' if min_aggregation else '_na'}"
                    + f"{'_ud' if use_uncertainty_difference else '_us'}"
                    + "_sac"
                )
                agent_config_fns[agent_name] = generate_agent_config(
                    agent_name=agent_name,
                    agent="gc",
                    base_config_fn=generate_gc_agent_config(
                        base_agent_fn=generate_sac_uncertainty_agent_config(
                            n_heads=10 if uncertainty_method == "bs" else 1,
                            mc_dropout=uncertainty_method == "mc",
                            mc_dropout_p=0.5,
                            mc_samples=100 if uncertainty_method == "mc" else 1,
                            uncertainty_aggregation=uncertainty_aggregation,
                            min_aggregation=min_aggregation,
                        ),
                        goal_generator_fn=generate_goal_generator_fn(
                            generator_name="uncertainty",
                            use_uncertainty_difference=use_uncertainty_difference,
                        ),
                        n_sampled_goals=4,
                        distance_type="l2_cluster",
                    ),
                )


agent_config_fns["gc_es_sac_trunk_sp"] = generate_agent_config(
    agent_name="gc_es_sac_sp",
    agent="gc",
    base_config_fn=generate_gc_agent_config(
        base_agent_fn=generate_sac_agent_config(
            compute_success_probability=True, add_trunk_architecture=True
        ),
        goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        goal_switcher="SuccessProbabilityGoalSwitcher",
        goal_switcher_kwargs={
            "conservative_factor": 0.95,
            "minimum_steps": 100,
            "trajectory_proportion": 0.2,
        },
    ),
)
agent_config_fns["gc_fb_sac_trunk"] = generate_agent_config(
    agent_name="gc_fb_sac",
    agent="gc",
    base_config_fn=generate_gc_agent_config(
        base_agent_fn=generate_sac_agent_config(add_trunk_architecture=True),
        goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
    ),
)


def build_configs_from_names(
    runner_name,
    env_name,
    agent_name,
    logger_name="wandb",
    tags=[],
    notes="",
    experiment_name="",
    num_seeds=5,
):
    return build_seeded_configs(
        runner_config_fn=runner_config_fns[runner_name],
        agent_config_fn=agent_config_fns[agent_name],
        env_config_fn=env_config_fns[env_name],
        logger_config_fn=logger_config_fns[logger_name],
        tags=tags,
        notes=notes,
        num_seeds=num_seeds,
        experiment_name=experiment_name,
    )


full_configs = {
    "debug": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["paired_gc_dqn"],
            env_names=["four_rooms_rf"],
            logger_names=["wandb"],
            tags=["debug"],
            num_seeds=3,
        ),
    ],
    "v8": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["episodic_dqn", "fb_dqn", "paired_dqn", "paired_gc_dqn"],
            env_names=["four_rooms_rf", "empty_rf"],
            logger_names=["wandb"],
            tags=["v8", "rfrl"],
        ),
        *generate_all_configs_from_names(
            runner_names=["epi"],
            agent_names=["episodic_dqn"],
            env_names=["four_rooms_epi", "empty_epi"],
            logger_names=["wandb"],
            tags=["v8", "rfrl"],
        ),
    ],
    "v9": [
        *generate_all_configs_from_names(
            runner_names=["epi"],
            agent_names=["episodic_sac"],
            env_names=[
                "tabletop_manipulation_epi",
                "sawyer_door_epi",
                "sawyer_peg_epi",
            ],
            logger_names=["wandb"],
            tags=["v9", "rfrl", "earl"],
        )
    ],
    "v11": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            # agent_names=["fb_sac"],
            agent_names=["paired_gc_sac_replay"],
            env_names=[
                "tabletop_manipulation_rf",
                "sawyer_door_rf",
                "sawyer_peg_rf",
            ],
            logger_names=["wandb"],
            tags=["v11", "rfrl", "earl"],
        )
    ],
    "v14": [
        *generate_all_configs_from_names(
            runner_names=["epi"],
            agent_names=["episodic_sac"],
            env_names=[
                "tabletop_manipulation_epi",
                "sawyer_door_epi",
                "sawyer_peg_epi",
            ],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "gold_base"],
            experiment_name="v14",
        ),
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["fb_sac", "gc_fb_sac", "gc_vaprl_sac"],
            # agent_names=["gc_vaprl_sac"],
            env_names=["tabletop_manipulation_rf", "sawyer_door_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "gold_base"],
            experiment_name="v14",
        ),
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["paired_gc_sac_demos", "paired_gc_sac_replay"],
            env_names=["tabletop_manipulation_rf", "sawyer_door_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl"],
            experiment_name="v14",
        ),
    ],
    "v16": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_unipaired_ma", "gc_unipaired_na"],
            env_names=["tabletop_manipulation_rf", "sawyer_door_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "v16"],
        ),
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                agent_name
                for agent_name in agent_config_fns.keys()
                if agent_name.startswith("gc_unc")
            ],
            env_names=["tabletop_manipulation_rf", "sawyer_door_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "v16"],
        ),
    ],
    "v17": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                "gc_unc_bs2_ud_ugs",
                "gc_unc_bs2_ud_bgs",
                "gc_unc_bs2_us_ugs",
                "gc_unc_bs2_us_bgs",
                "gc_unipaired_na",
                "gc_unc_bs_range_na_us_sac",
                "gc_unc_bs_std_na_us_sac",
            ],
            env_names=["tabletop_manipulation_rf", "sawyer_door_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "v16"],
        ),
    ],
    "fb_baselines": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac", "fb_sac"],
            env_names=["sawyer_door_rf", "tabletop_manipulation_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "baselines"],
        )
    ],
    "fb_bl2": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac2"],
            env_names=["sawyer_door_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "baselines"],
        )
    ],
    "v18": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                "gc_fb_sac",
                "gc_unipaired_ma",
                "gc_unc_bs_range_ma_ud_sac",
                "gc_unc_bs_std_ma_ud_sac",
                "gc_unc_bs_range_ma_us_sac",
                "gc_unc_bs_std_ma_us_sac",
                "gc_vaprl_sac",
            ],
            env_names=["sawyer_door_rf", "tabletop_manipulation_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "v18"],
        )
    ],
    "buggy_fb_test": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["buggy_gc_fb_sac"],
            env_names=["sawyer_door_rf", "tabletop_manipulation_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "buggy_fb_test"],
        )
    ],
    "fk_runs": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            # agent_names=["gc_fb_sac", "gc_unipaired_ma", "gc_unc_bs_range_ma_ud_sac"],
            agent_names=["gc_fb_no_term_sac"],
            env_names=["kitchen_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fb_base"],
        )
    ],
    "minitaur_runs": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            # agent_names=["gc_fb_sac", "gc_unipaired_ma", "gc_unc_bs_range_ma_ud_sac"],
            agent_names=["gc_fb_no_term_sac"],
            env_names=["minitaur_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fb_base"],
        )
    ],
    "no_term_runs": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_no_term_sac"],
            env_names=["sawyer_door_rf", "tabletop_manipulation_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fb_base"],
        )
    ],
    "sp_sac": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac"],
            env_names=["sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fb_base"],
        )
    ],
    "4r_switching_v1": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_dqn_gs", "gc_fb_dqn"],
            env_names=["four_rooms_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "minigrid", "switching_v1"],
        )
    ],
    "4r_switching_v2": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_dqn_gs_pri", "gc_fb_dqn_pri"],
            env_names=["four_rooms_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "minigrid", "switching_v2"],
        )
    ],
    "4r_switching_v3": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_dqn_gs", "gc_fb_dqn_gsr"],
            env_names=["four_rooms_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "minigrid", "switching_v3"],
        )
    ],
    "4r_switching_v4": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_dqn", "gc_fb_dqn_gs", "gc_fb_dqn_gsr"],
            # agent_names=["gc_fb_dqn_gs_ow"],
            env_names=["four_rooms_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "minigrid", "switching_v4"],
        )
    ],
    "4r_switching_v5": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_dqn", "gc_fb_dqn_gs", "gc_fb_dqn_gsr"],
            env_names=["four_rooms_rf_every"],
            logger_names=["wandb"],
            tags=["rfrl", "minigrid", "switching_v5"],
            notes="Running 20 seeds, adding every state evaluation",
            num_seeds=20,
        )
    ],
    "earl_sparse_switching": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_gs"],
            env_names=["sawyer_door_rf", "tabletop_manipulation_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "sparse_switching"],
        )
    ],
    "earl_vaprl": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_vaprl_sac"],
            env_names=["sawyer_door_rf", "tabletop_manipulation_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "vaprl"],
        )
    ],
    "minitaur_switching": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_gs"],
            env_names=["minitaur_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "sparse_switching"],
        )
    ],
    "fk_switching": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_gs"],
            env_names=["kitchen_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "sparse_switching"],
        )
    ],
    "4r_switching_v6": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                "gc_fb_dqn",
                "gc_fb_dqn_gs",
                "gc_fb_dqn_gsr",
                "gc_fb_dqn_gs_sf",
                "gc_fb_dqn_gs_sb",
                "gc_fb_dqn_gsr_sf",
                "gc_fb_dqn_gsr_sb",
            ],
            env_names=["four_rooms_rf_every"],
            logger_names=["wandb"],
            tags=["rfrl", "minigrid", "switching_v7"],
            notes="Running 20 seeds, every state evaluation, one way switching",
            num_seeds=20,
        )
    ],
    "4r_switching_v7": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                "gc_fb_dqn_gs_sched",
            ],
            env_names=["four_rooms_rf_every"],
            logger_names=["wandb"],
            tags=["rfrl", "minigrid", "switching_v7"],
            notes="Running 20 seeds, trying schedule of episode length",
            num_seeds=20,
        )
    ],
    "fk_switching_v2": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_gs_sched", "gc_fb_sac_gs_r"],
            env_names=["kitchen_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "earl_switching_v2"],
        )
    ],
    "earl_switching_v2": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_gs_sched", "gc_fb_sac_gs_r"],
            env_names=["sawyer_door_rf", "tabletop_manipulation_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "earl_switching_v2"],
        )
    ],
    "minitaur_switching_v2": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_gs_sched", "gc_fb_sac_gs_r"],
            env_names=["minitaur_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "earl_switching_v2"],
        )
    ],
    "fk_switching_v2": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_gs_sched", "gc_fb_sac_gs_r"],
            env_names=["kitchen_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "earl_switching_v2"],
        )
    ],
    "minitaur_fb_debug": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac", "gc_fb_sac_sc"],
            env_names=["minitaur_rf_debug"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "minitaur_fb_debug"],
        )
    ],
    "full_minitaur_final": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                # "gc_fb_sac_gs_minitaur1",
                # "gc_fb_sac_gs_minitaur2",
                "gc_fb_sac",
                "buggy_gc_fb_sac",
            ],
            env_names=["minitaur_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "paper_full"],
        )
    ],
    "full_sd_final": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            # agent_names=["gc_fb_sac_gs_sd", "gc_fb_sac", "buggy_gc_fb_sac"],
            agent_names=["gc_fb_sac", "buggy_gc_fb_sac"],
            # agent_names=["gc_fb_sac_gs_sd1", "gc_fb_sac_gs_sd2"],
            env_names=["sawyer_door_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "paper_full"],
        )
    ],
    "full_sp_final": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            # agent_names=["gc_fb_sac_gs_sp", "gc_fb_sac", "buggy_gc_fb_sac"],
            agent_names=["gc_fb_sac", "buggy_gc_fb_sac"],
            # agent_names=["gc_fb_sac_gs_sp1", "gc_fb_sac_gs_sp2"],
            env_names=["sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "paper_full"],
        )
    ],
    "full_tm_final": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac", "buggy_gc_fb_sac"],
            # agent_names=["gc_fb_sac_gs_tm1", "gc_fb_sac_gs_tm2"],
            env_names=["tabletop_manipulation_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "paper_full"],
        )
    ],
    "minitaur_epi": [
        *generate_all_configs_from_names(
            runner_names=["epi"],
            agent_names=["gc_fb_sac"],
            env_names=["minitaur_epi"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "paper_full", "oracle"],
        )
    ],
    "cos_debug_tm": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                "gc_fb_sac_gs_tm3",
                # "gc_fb_sac_gs_sd3",
                # "gc_fb_sac_gs_sp3",
                # "gc_fb_sac_gs_minitaur3",
            ],
            env_names=["tabletop_manipulation_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "cos_debug"],
        )
    ],
    "cos_debug_sd": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                # "gc_fb_sac_gs_tm3",
                "gc_fb_sac_gs_sd3",
                # "gc_fb_sac_gs_sp3",
                # "gc_fb_sac_gs_minitaur3",
            ],
            env_names=["sawyer_door_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "cos_debug"],
        )
    ],
    "cos_debug_sp": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                # "gc_fb_sac_gs_tm3",
                # "gc_fb_sac_gs_sd3",
                "gc_fb_sac_gs_sp3",
                # "gc_fb_sac_gs_minitaur3",
            ],
            env_names=["sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "cos_debug"],
        )
    ],
    "cos_debug_minitaur": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                # "gc_fb_sac_gs_tm3",
                # "gc_fb_sac_gs_sd3",
                # "gc_fb_sac_gs_sp3",
                "gc_fb_sac_gs_minitaur3",
            ],
            env_names=["minitaur_rf"],
            logger_names=["wandb"],
            tags=[
                "rfrl",
                "earl",
                "cos_debug",
            ],
        )
    ],
    "fk_vpn1": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac", "gc_fb_sac_rf1"],
            env_names=["kitchen_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_vpn1"],
        ),
        *generate_all_configs_from_names(
            runner_names=["epi"],
            agent_names=["episodic_sac", "episodic_sac_rf1"],
            env_names=["kitchen_epi"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_vpn1"],
        ),
    ],
    "fk_vpn2": [
        *generate_all_configs_from_names(
            runner_names=["epi_no_term"],
            agent_names=["episodic_sac_tat"],
            env_names=["kitchen_epi"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_vpn2"],
        ),
    ],
    "fk_vpn3": [
        *generate_all_configs_from_names(
            runner_names=["epi_time_trunc", "epi_success_trunc"],
            agent_names=["episodic_sac"],
            env_names=["kitchen_epi"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_vpn2"],
        ),
    ],
    "fk_fb_rev": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_to", "gc_fb_sac"],
            env_names=["kitchen_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_fb_rev"],
        ),
    ],
    "fk_fb_rev2": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_to_sep", "gc_fb_sac_sep"],
            env_names=["kitchen_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_fb_rev2"],
        ),
    ],
    "fk_fb": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_to"],
            env_names=["kitchen_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_fb"],
        ),
    ],
    "fk_fb2": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_to_sep"],
            env_names=["kitchen_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_fb"],
        ),
    ],
    "minitaur_nt": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_to", "gc_fb_sac_gs_nt_minitaur3"],
            env_names=["minitaur_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "minitaur_nt"],
        ),
    ],
    "rebuttal_tm": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                "gc_fb_sac_log",
                "gc_fb_sac_gs_tm3_log",
            ],
            env_names=["tabletop_manipulation_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "rebuttal_logging3"],
        )
    ],
    "rebuttal_sd": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                "gc_fb_sac_log",
                "gc_fb_sac_gs_sd3_log",
            ],
            env_names=["sawyer_door_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "rebuttal_logging3"],
        )
    ],
    "rebuttal_sp": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                "gc_fb_sac_log",
                "gc_fb_sac_gs_sp3_log",
            ],
            env_names=["sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "rebuttal_logging3"],
        )
    ],
    "rebuttal_minitaur": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=[
                "gc_fb_sac_log",
                "gc_fb_sac_gs_minitaur3_log",
                # "gc_fb_sac_to",
                # "gc_fb_sac_gs_nt_minitaur3",
            ],
            env_names=["minitaur_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "rebuttal_logging3"],
        )
    ],
    "rebuttal_vaprl": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_vaprl_bug_sac", "gc_vaprl_sac"],
            # agent_names=["gc_vaprl_sac"],
            env_names=["tabletop_manipulation_rf", "sawyer_door_rf", "sawyer_peg_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "rebuttal", "vaprl"],
        ),
    ],
    "fk_rev3": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_notrunc", "gc_fb_sac_stable"],
            env_names=["kitchen_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_rev3"],
        ),
    ],
    "fk_easy": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac"],
            env_names=["microwave_rf", "light_switch_rf", "micro_light_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_easy"],
        ),
    ],
    "fk_easy2": [
        *generate_all_configs_from_names(
            runner_names=["rf"],
            agent_names=["gc_fb_sac_to"],
            env_names=["microwave_rf", "light_switch_rf", "micro_light_rf"],
            logger_names=["wandb"],
            tags=["rfrl", "earl", "fk_easy2"],
        ),
    ],
}

fk_is_fb_configs = []

for (
    kitchen_task,
    separate_agents,
    switching_strategy,
    bootstrap_strategy,
) in itertools.product(
    [
        "microwave",
        "light_switch",
        "slide_cabinet",
        "hinge_cabinet",
        "micro_slide",
        "micro_light",
        "light_slide",
        "light_hinge",
        "slide_hinge",
    ],
    [False, True],
    [0, 1, 2],
    [0, 1, 2],
):
    fk_is_fb_configs.extend(
        [
            *generate_all_configs_from_names(
                runner_names=["rf"],
                agent_names=[
                    f"is_bs{bootstrap_strategy}_ss{switching_strategy}_sa{separate_agents}_kg{False}"
                ],
                env_names=[f"{kitchen_task}_rf"],
                logger_names=["wandb"],
                tags=["rfrl", "earl", "fk_is_fb"],
            )
        ]
    )


full_configs["fk_is_fb"] = fk_is_fb_configs
full_configs["trunk_sp"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=[
            "gc_es_sac_trunk_sp",
            "gc_fb_sac_trunk",
        ],
        env_names=["sawyer_peg_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "trunk"],
    )
]
# full_configs["fb_trunk_sp"] = [
#     *generate_all_configs_from_names(
#         runner_names=["rf"],
#         agent_names=["gc_fb_sac_trunk"],
#         env_names=["sawyer_peg_rf"],
#         logger_names=["wandb"],
#         tags=["rfrl", "earl", "iclr_fb"],
#     )
# ]
full_configs["fb_trunk_sd"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=["gc_fb_sac_trunk"],
        env_names=["sawyer_door_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "iclr_fb"],
    )
]
full_configs["fb_trunk_tm2"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=["gc_fb_sac_trunk"],
        env_names=["tabletop_manipulation_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "iclr_fb"],
    )
]
full_configs["fb_trunk_minitaur"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=["gc_fb_sac_trunk"],
        env_names=["minitaur_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "iclr_fb"],
    )
]


agent_config_fns["gc_fb_sac_term_is_sp"] = generate_agent_config(
    agent_name="gc_fb_sac_term_is_sp",
    agent="gc",
    base_config_fn=generate_gc_agent_config(
        base_agent_fn=generate_sac_agent_config(
            compute_success_probability=True,
            add_trunk_architecture=True,
            trunc_as_terminal=True,
        ),
        goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        goal_switcher="SuccessProbabilityGoalSwitcher",
        goal_switcher_kwargs={
            "conservative_factor": 0.9,
            "minimum_steps": 100,
            "trajectory_proportion": 0.5,
        },
    ),
)

agent_config_fns["gc_fb_sac_term_is_sd"] = generate_agent_config(
    agent_name="gc_fb_sac_term_is_sd",
    agent="gc",
    base_config_fn=generate_gc_agent_config(
        base_agent_fn=generate_sac_agent_config(
            compute_success_probability=True,
            add_trunk_architecture=True,
            trunc_as_terminal=True,
        ),
        goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        goal_switcher="SuccessProbabilityGoalSwitcher",
        goal_switcher_kwargs={
            "conservative_factor": 0.9,
            "minimum_steps": 225,
            "trajectory_proportion": 1.0,
        },
    ),
)

agent_config_fns["gc_fb_sac_term_is_tm"] = generate_agent_config(
    agent_name="gc_fb_sac_term_is_tm",
    agent="gc",
    base_config_fn=generate_gc_agent_config(
        base_agent_fn=generate_sac_agent_config(
            compute_success_probability=True,
            add_trunk_architecture=True,
            trunc_as_terminal=True,
        ),
        goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        goal_switcher="SuccessProbabilityGoalSwitcher",
        goal_switcher_kwargs={
            "conservative_factor": 0.9,
            "minimum_steps": 100,
            "trajectory_proportion": 1.0,
        },
    ),
)
agent_config_fns["gc_fb_sac_term_is_sp"] = generate_agent_config(
    agent_name="gc_fb_sac_term_is_sp",
    agent="gc",
    base_config_fn=generate_gc_agent_config(
        base_agent_fn=generate_sac_agent_config(
            compute_success_probability=True,
            add_trunk_architecture=True,
            trunc_as_terminal=True,
        ),
        goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        goal_switcher="SuccessProbabilityGoalSwitcher",
        goal_switcher_kwargs={
            "conservative_factor": 0.9,
            "minimum_steps": 100,
            "trajectory_proportion": 0.5,
        },
    ),
)

agent_config_fns["gc_fb_sac_term_is_minitaur"] = generate_agent_config(
    agent_name="gc_fb_sac_term_is_sp",
    agent="gc",
    base_config_fn=generate_gc_agent_config(
        base_agent_fn=generate_sac_agent_config(
            compute_success_probability=True,
            add_trunk_architecture=True,
            trunc_as_terminal=True,
        ),
        goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        goal_switcher="SuccessProbabilityGoalSwitcher",
        goal_switcher_kwargs={
            "conservative_factor": 0.95,
            "minimum_steps": 250,
            "trajectory_proportion": 0.25,
        },
    ),
)
agent_config_fns["gc_fb_sac_nodemo_is_sd"] = generate_agent_config(
    agent_name="gc_fb_sac_nodemo_is_sd",
    agent="gc",
    base_config_fn=generate_gc_agent_config(
        base_agent_fn=generate_sac_agent_config(
            compute_success_probability=True,
            add_trunk_architecture=True,
        ),
        goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        goal_switcher="SuccessProbabilityGoalSwitcher",
        goal_switcher_kwargs={
            "conservative_factor": 0.9,
            "minimum_steps": 225,
            "trajectory_proportion": 1.0,
        },
        use_demo=False,
    ),
)

agent_config_fns["gc_fb_sac_nodemo_is_tm"] = generate_agent_config(
    agent_name="gc_fb_sac_nodemo_is_tm",
    agent="gc",
    base_config_fn=generate_gc_agent_config(
        base_agent_fn=generate_sac_agent_config(
            compute_success_probability=True,
            add_trunk_architecture=True,
        ),
        goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        goal_switcher="SuccessProbabilityGoalSwitcher",
        goal_switcher_kwargs={
            "conservative_factor": 0.9,
            "minimum_steps": 100,
            "trajectory_proportion": 1.0,
        },
        use_demo=False,
    ),
)
agent_config_fns["gc_fb_sac_nodemo_is_sp"] = generate_agent_config(
    agent_name="gc_fb_sac_nodemo_is_sp",
    agent="gc",
    base_config_fn=generate_gc_agent_config(
        base_agent_fn=generate_sac_agent_config(
            compute_success_probability=True,
            add_trunk_architecture=True,
        ),
        goal_generator_fn=generate_goal_generator_fn(generator_name="fb"),
        goal_switcher="SuccessProbabilityGoalSwitcher",
        goal_switcher_kwargs={
            "conservative_factor": 0.9,
            "minimum_steps": 100,
            "trajectory_proportion": 0.5,
        },
        use_demo=False,
    ),
)

full_configs["is_term_sp"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=["gc_fb_sac_term_is_sp"],
        env_names=["sawyer_peg_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "iclr_term"],
    )
]

full_configs["is_term_sd"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=["gc_fb_sac_term_is_sd"],
        env_names=["sawyer_door_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "iclr_term"],
    )
]

full_configs["is_term_tm"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=["gc_fb_sac_term_is_tm"],
        env_names=["tabletop_manipulation_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "iclr_term"],
    )
]

full_configs["is_term_minitaur"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=["gc_fb_sac_term_is_minitaur"],
        env_names=["minitaur_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "iclr_term"],
    )
]

full_configs["is_nodemo_sd"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=["gc_fb_sac_nodemo_is_sd"],
        env_names=["sawyer_door_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "iclr_nodemo"],
    )
]

full_configs["is_nodemo_tm"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=["gc_fb_sac_nodemo_is_tm"],
        env_names=["tabletop_manipulation_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "iclr_nodemo"],
    )
]

full_configs["is_nodemo_sp"] = [
    *generate_all_configs_from_names(
        runner_names=["rf"],
        agent_names=["gc_fb_sac_nodemo_is_sp"],
        env_names=["sawyer_peg_rf"],
        logger_names=["wandb"],
        tags=["rfrl", "earl", "iclr_nodemo"],
    )
]
