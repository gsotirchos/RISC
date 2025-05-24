import copy
from enum import Enum

import earl_benchmark
import gymnasium as gym
import numpy as np
from envs.reset_free_envs import ResetFreeEnv
from envs.types import UpdateInfo
from hive.envs import GymEnv

from functools import partial


class DistanceType(str, Enum):
    l2 = "l2"
    l2_cluster = "l2_cluster"
    timestep = "timestep"


class EARLWrapper(gym.ObservationWrapper):
    """Wrapper for EARL environments that adds a desired goal to the observation."""

    def __init__(self, env, goal_size):
        super().__init__(env)
        self.goal_size = goal_size
        observation_low = env.observation_space.low[:-goal_size]
        observation_high = env.observation_space.high[:-goal_size]
        goal_low = env.observation_space.low[-goal_size:]
        goal_high = env.observation_space.high[-goal_size:]
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=observation_low, high=observation_high, dtype=np.float32
                ),
                "desired_goal": gym.spaces.Box(
                    low=goal_low, high=goal_high, shape=(goal_size,), dtype=np.float32
                ),
            }
        )

    def observation(self, observation):
        goal = observation[-self.goal_size :].astype(np.float32)
        observation = observation[: -self.goal_size].astype(np.float32)
        return {"observation": observation, "desired_goal": goal}


def convert_to_gc_obs(obs):
    state = obs["observation"]
    goal = obs["desired_goal"]
    return np.concatenate([state, goal], axis=-1)


def replace_goal_fn(obs, goal):
    obs = copy.deepcopy(obs)
    obs["desired_goal"] = goal
    return obs


class EARLEnv(GymEnv):
    """Adapts the EARL environment to the GymEnv interface."""

    def create_env(
        self,
        env_name,
        env_wrappers=None,
        env=None,
        goal_size=None,
        reset_free=True,
        **kwargs
    ):
        env = gym.make("GymV21Environment-v0", env=env)
        self._reset_free = reset_free
        self._env = EARLWrapper(env, goal_size=goal_size)

    def is_successful(self, gc_obs):
        return bool(self._env.is_successful(gc_obs))

    def compute_reward(self, obs):
        return self._env.compute_reward(obs)

    def step(self, action):
        observation, reward, terminated, truncated, self._turn, info = super().step(
            action
        )
        truncated = bool(terminated)
        terminated = (not self._reset_free) and bool(
            self._env.is_successful(convert_to_gc_obs(observation))
        )
        return observation, reward, terminated, truncated, self._turn, info

    def register_logger(self, logger):
        self.logger = logger


def process_demos(demos, goal_shape):
    """Processes the demonstrations to be used by the agents."""
    if demos is None:
        return demos
    return {
        "observation": demos["observations"][:, :-goal_shape],
        "desired_goal": demos["observations"][:, -goal_shape:],
        "action": demos["actions"],
        "reward": demos["rewards"][:, 0],
        "terminated": demos["terminals"][:, 0],
        "truncated": np.zeros(demos["terminals"].shape[0], dtype=bool),
        "next_observation": demos["next_observations"][:, :-goal_shape],
        "info": demos["infos"],
        "achieved_goal": np.copy(demos["next_observations"][:, :-goal_shape]),
    }


def get_goals(env_name, env_loader, env):
    if env_name == "minitaur":
        return np.array(env.env._goal_locations)
    else:
        return env_loader.get_goal_states()


def get_distance_fn(
    distance_type: DistanceType,
    initial_states: np.ndarray,
    forward_demos,
    backward_demos,
):
    """Returns a distance function that computes the distance between the
    observation and the initial states."""
    initial_states = initial_states.reshape((initial_states.shape[0], -1))  # N x D
    initial_states_mean = np.mean(initial_states, axis=0)  # D

    def l2_distance_fn(x):  # expected shape is M x ...
        return np.linalg.norm(x.reshape((x.shape[0], -1)) - initial_states_mean, axis=1)

    def l2_cluster_distance_fn(x):
        single_state = x.ndim < initial_states.ndim
        if single_state:
            x = x[np.newaxis, ...]  # 1 x D x ...
        x = x[:, -initial_states.shape[-1] :]  # M x D x ...
        states = initial_states[np.newaxis, ...]  # 1 x N x D
        x = x.reshape((x.shape[0], 1, -1))  # M x 1 x D
        dists = np.amin(np.linalg.norm(x - states, axis=-1), axis=1)  # M
        if single_state:
            return dists[0]
        else:
            return dists

    if forward_demos is not None:
        timesteps = []
        timestep = 0
        for terminal in forward_demos["terminated"]:
            timesteps.append(timestep)
            timestep += 1
            if terminal:
                timestep = 0
        if backward_demos is not None:
            timestep = 0
            backward_timesteps = []
            for terminal in backward_demos["terminated"][::-1]:
                backward_timesteps.append(timestep)
                timestep += 1
                if terminal:
                    timestep = 0
            timesteps += backward_timesteps[::-1]
        timesteps = np.array(timesteps)

    def timestep_distance(x):
        """Returns the timestep index of the observation in the
        demonstration."""
        return timesteps

    if distance_type == "l2":
        return l2_distance_fn
    elif distance_type == "l2_cluster":
        return l2_cluster_distance_fn
    elif distance_type == "timestep":
        if forward_demos is None:
            raise ValueError("Timestep distance requires forward_demos to be provided.")
        return timestep_distance
    else:
        raise NotImplementedError


def create_success_fn(env):
    def success_fn(observation, goal=None):
        state = observation["observation"]
        if goal is None:
            goal = observation["desired_goal"]
        gc_observation = np.concatenate([state, goal], axis=-1)
        return env.is_successful(gc_observation)

    return success_fn


def create_reward_fn(env):
    def reward_fn(observation, goal=None):
        state = observation["observation"]
        if goal is None:
            goal = observation["desired_goal"]
        gc_observation = np.concatenate([state, goal], axis=-1)
        reward = env.compute_reward(gc_observation)
        if isinstance(reward, list):
            reward = reward[0]
        return reward

    return reward_fn


def get_earl_envs(
    env_name,
    reward_type: str = "sparse",
    reset_train_env_at_goal: bool = False,
    setup_as_lifelong_learning: bool = False,
    reset_free: bool = True,
    **kwargs
    # Arguments that can be passed from the config file
    # train_horizon,
    # eval_horizon,
    # num_initial_state_samples,
    # goal_change_frequency,
    # wide_init_distr,
    # kitchen_task,
):
    env_loader = earl_benchmark.EARLEnvs(
        env_name=env_name,
        reward_type=reward_type,
        reset_train_env_at_goal=reset_train_env_at_goal,
        setup_as_lifelong_learning=setup_as_lifelong_learning,
        **kwargs
    )
    train_env, eval_env = env_loader.get_envs()
    if not reset_free:
        train_env = env_loader.get_eval_env()
    forward_demos, reverse_demos = env_loader.get_demonstrations()
    goals = get_goals(env_name, env_loader, train_env)
    initial_states = env_loader.get_initial_states()
    initial_states = initial_states[:, -goals.shape[-1] :]
    forward_demos = process_demos(forward_demos, goals.shape[-1])
    backward_demos = process_demos(reverse_demos, goals.shape[-1])
    train_env = EARLEnv(
        env_name=env_name,
        env=train_env,
        goal_size=goals[0].shape[0],
        reset_free=reset_free,
    )
    eval_env = EARLEnv(
        env_name=env_name,
        env=eval_env,
        goal_size=goals[0].shape[0],
        reset_free=False
    )
    return ResetFreeEnv(
        train_env=lambda: train_env,
        eval_env=lambda: eval_env,
        success_fn=create_success_fn(train_env),
        reward_fn=create_reward_fn(train_env),
        vis_fn=None,
        replace_goal_fn=replace_goal_fn,
        get_distance_fn=partial(
            get_distance_fn,
            distance_type="l2_cluster",
            initial_states=initial_states,
            forward_demos=forward_demos,
            backward_demos=backward_demos,
        ),
        goal_states=goals,
        initial_states=initial_states,
        forward_demos=forward_demos,
        backward_demos=backward_demos,
        eval_every=False,
    )
