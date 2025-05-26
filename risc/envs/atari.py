import copy
from enum import Enum
from functools import partial

import numpy as np
import gymnasium as gym
from hive.envs import GymEnv
from envs.reset_free_envs import ResetFreeEnv
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from hive.envs.gym.gym_wrappers import PermuteImageWrapper

from functools import partial


class DistanceType(str, Enum):
    l2 = "l2"
    l2_cluster = "l2_cluster"
    timestep = "timestep"


class GCObsWrapper(gym.ObservationWrapper):
    """Wrapper for Atari environments that adds a desired goal to the observation."""

    def __init__(self, env
                 #, goal_size
                 ):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "desired_goal": env.observation_space,
            }
        )
        self.goal = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype)

    def observation(self, observation):
        return {"observation": observation, "desired_goal": self.goal}


class AtariEnv(GymEnv):
    """Expands the GymEnv interface."""
    def __init__(
        self,
        *args,
        seed=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.seed(seed=seed)

    # def step(self, action):
    #     observation, reward, terminated, truncated, self._turn, info = super().step(action)
    #     truncated = bool(terminated)
    #     terminated = self.is_successful(concat_to_gc_obs(observation))
    #     return observation, reward, terminated, truncated, self._turn, info

    # def reset(self):
    #     return super().reset()

    def teleport(self, _):
        observation, _ = super().reset()
        return observation

    def register_logger(self, logger):
        self.logger = logger

    def save(self, _):
        pass


def success_fn(observation, goal=None):
    obs = observation["observation"]
    if goal is None:
        goal = observation["desired_goal"]
    return np.allclose(obs[0], goal[0])


def reward_fn(observation, goal=None, env_reward=-1, **kwargs):
    bonus = -10 + float(success_fn(observation, goal))
    return env_reward + bonus


def replace_goal_fn(obs, goal):
    obs = copy.deepcopy(obs)
    obs["desired_goal"] = goal
    return obs


def get_distance_fn(
    distance_type: DistanceType,
    initial_states: np.ndarray,
    # forward_demos,
    # backward_demos,
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

    # if forward_demos is not None:
    #     timesteps = []
    #     timestep = 0
    #     for terminal in forward_demos["terminated"]:
    #         timesteps.append(timestep)
    #         timestep += 1
    #         if terminal:
    #             timestep = 0
    #     if backward_demos is not None:
    #         timestep = 0
    #         backward_timesteps = []
    #         for terminal in backward_demos["terminated"][::-1]:
    #             backward_timesteps.append(timestep)
    #             timestep += 1
    #             if terminal:
    #                 timestep = 0
    #         timesteps += backward_timesteps[::-1]
    #     timesteps = np.array(timesteps)

    # def timestep_distance(x):
    #     """Returns the timestep index of the observation in the
    #     demonstration."""
    #     return timesteps

    if distance_type == "l2":
        return l2_distance_fn
    elif distance_type == "l2_cluster":
        return l2_cluster_distance_fn
    # elif distance_type == "timestep":
    #     if forward_demos is None:
    #         raise ValueError("Timestep distance requires forward_demos to be provided.")
    #     return timestep_distance
    else:
        raise NotImplementedError


def get_atari_envs(
        env_name,
        repeat_action_probability,
        frame_skip,
        screen_size,
        grayscale_newaxis,
        eval_every=False,
        seed=None,
        **kwargs
):
    train_env = AtariEnv(
        env_name,
        repeat_action_probability=repeat_action_probability,
        frameskip=1,
        env_wrappers=[
            partial(
                AtariPreprocessing,
                frame_skip=frame_skip,
                screen_size=screen_size,
                grayscale_newaxis=grayscale_newaxis,
            ),
            PermuteImageWrapper,
            GCObsWrapper,
        ],
        seed=seed,
        **kwargs
    )
    eval_env = copy.deepcopy(train_env)
    obs, _ =  train_env.reset()
    initial_states = np.expand_dims(obs['observation'], axis=0)
    goal_states = np.expand_dims(obs['desired_goal'], axis=0)
    forward_demos = None
    backward_demos = None
    return ResetFreeEnv(
        train_env=lambda: train_env,
        eval_env=lambda: eval_env,
        success_fn=success_fn,
        reward_fn=reward_fn,
        replace_goal_fn=replace_goal_fn,
        all_states_fn=None,
        vis_fn=None,
        get_distance_fn=lambda distance_type: lambda x: 0,
        # partial(
        #     get_distance_fn,
        #     distance_type="l2_cluster",
        #     initial_states=initial_states,
        #     # forward_demos=forward_demos,
        #     # backward_demos=backward_demos,
        # ),
        goal_states=goal_states,
        initial_states=initial_states,
        forward_demos=forward_demos,
        backward_demos=backward_demos,
        eval_every=eval_every,
    )
