import os
import time
import copy
import random
# from functools import partial
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
import minihack
from hive.envs import GymEnv
from envs.reset_free_envs import ResetFreeEnv
from gymnasium.spaces import Box


# fix legacy error
# gym.wrappers.common.PassiveEnvChecker.reset = lambda self, **kwargs: self.env.reset(**kwargs)


@dataclass(frozen=True)
class CharValues():
    WALL: int = 35
    FLOOR: int = 46
    AGENT: int = 64
    BOX: int = 96
    GOAL: int = 123


class ReseedWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.unwrapped._goal_pos_set = {}

    def get_lvl_gen(self, lvl_index=None):
        env = self.env.unwrapped
        if lvl_index is None:
            level = random.choice(env._levels)
        else:
            level = env._levels[lvl_index]
        level = level.split("\n")
        map, info = env.get_env_map(level)
        flags = list(env._flags)
        flags.append("noteleport")
        flags.append("premapped")
        lvl_gen = minihack.LevelGenerator(map=map, lit=True, flags=flags, solidfill=" ")
        for b in info["boulders"]:
            lvl_gen.add_boulder(b)
        for f in info["fountains"]:
            lvl_gen.add_fountain(f)
        lvl_gen.set_start_pos(info["player"])
        return lvl_gen

    def reset(self, level=0, **kwargs):
        env = self.env.unwrapped
        des_file = self.get_lvl_gen(level).get_des()
        env.update(des_file)
        initial_obs = super(minihack.MiniHackNavigation, env).reset(**kwargs)
        env._goal_pos_set = env._object_positions(env.last_observation, "{")
        env.seed(0, 0, reseed=False)
        return initial_obs


class GCObsWrapper(gym.ObservationWrapper):
    """Wrapper for MiniHack environments that adds a desired goal to the observation."""

    def __init__(self, env):
        super().__init__(env)
        observation_space = Box(
            low=0,
            high=255,
            shape=(1, 10, 10),
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "observation": observation_space,
                "desired_goal": observation_space,
            }
        )

    def observation(self, observation):
        obs = observation["chars"][7:-4, 34:-35]
        obs = np.expand_dims(obs, axis=0)
        self.goal = copy.deepcopy(obs)
        self.goal[self.goal == CharValues.AGENT] = CharValues.FLOOR
        self.goal[self.goal == CharValues.BOX] = CharValues.FLOOR
        self.goal[self.goal == CharValues.GOAL] = CharValues.BOX
        return {"observation": obs, "desired_goal": self.goal}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class MiniHackEnv(GymEnv):
    """Expands the GymEnv interface."""
    def __init__(
        self,
        *args,
        seed=None,
        level=0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._env.set_wrapper_attr("_elapsed_steps", 0)
        self.seed(seed=seed)
        self._level = level
        self._random_goals = level is None

    def step(self, action, **kwargs):
        if self._env.render_mode == "human":
            os.system('cls' if os.name == 'nt' else 'clear')
            self.render()
            time.sleep(0.25)
        return super().step(action, **kwargs)
    #     observation, reward, terminated, truncated, self._turn, info = super().step(action)
    #     truncated = bool(terminated)
    #     terminated = self.is_successful(concat_to_gc_obs(observation))
    #     return observation, reward, terminated, truncated, self._turn, info

    def reset(self):
        level = None if self._random_goals else self._level
        return self._env.reset(level=level)

    def close(self):
        pass

    def teleport(self, _):
        observation, _ = super().reset()
        return observation

    def randomize_goal(self):
        self._random_goals = True

    def reset_goal(self):
        self._random_goals = False

    def place_subgoal(self, pos):
        # TODO
        pass

    def remove_subgoal(self):
        # TODO
        pass

    def register_logger(self, logger):
        self.logger = logger

    def save(self, _):
        pass


def success_fn(observation, goal=None):
    obs = observation["observation"]
    if goal is None:
        goal = observation["desired_goal"]
    if not np.any(obs == CharValues.GOAL):
        # success if all boxes are moved to the goal tiles
        return True
    if np.allclose(obs, goal):
        # success if the current goal state is reached
        return True
    return False


def reward_fn(observation, goal=None, env_reward=0, **kwargs):
    bonus = -1 + float(success_fn(observation, goal))
    return env_reward + bonus


def replace_goal_fn(obs, goal):
    obs = copy.deepcopy(obs)
    obs["desired_goal"] = goal
    return obs


def get_minihack_envs(
    env_name,
    seed=None,
    eval_every=False,
    train_max_steps=100,
    eval_max_steps=100,
    **kwargs
):
    train_env = MiniHackEnv(
        env_name,
        observation_keys=("chars",),  # or "glyphs", "chars", "colors", "pixel"
        env_wrappers=[
            ReseedWrapper,
            GCObsWrapper,
        ],
        seed=seed,
        max_episode_steps=train_max_steps,
        **kwargs
    )
    eval_env = MiniHackEnv(
        env_name,
        observation_keys=("chars",),  # or "glyphs", "chars", "colors", "pixel"
        env_wrappers=[
            ReseedWrapper,
            GCObsWrapper,
        ],
        seed=seed,
        max_episode_steps=eval_max_steps,
        **kwargs
    )
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
        goal_states=goal_states,
        initial_states=initial_states,
        forward_demos=forward_demos,
        backward_demos=backward_demos,
        eval_every=eval_every,
    )
