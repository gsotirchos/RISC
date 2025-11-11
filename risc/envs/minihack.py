import os
import time
import copy
import random
# from functools import partial
from dataclasses import dataclass, fields
from functools import partial
# from inspect import getsource

import numpy as np
import gymnasium as gym
import minihack
from hive.envs import GymEnv
from envs.reset_free_envs import ResetFreeEnv
from gymnasium.spaces import Box


@dataclass(frozen=True)
class DefaultCharValues():
    WALL: int = 35
    FLOOR: int = 46
    AGENT: int = 64
    START: int = 60
    BOULDER: int = 96
    FOUNTAIN: int = 123


@dataclass(frozen=True)
class CharValues():
    WALL: int = 255
    FLOOR: int = 0
    AGENT: int = 128
    START: int = 1
    BOULDER: int = 192
    FOUNTAIN: int = 64


class ReseedWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.unwrapped._goal_pos_set = {}
        self._default_level = 0
        self._random_goals = False
        self._find_and_patch_reset(self.env, gym.wrappers.PassiveEnvChecker)

    def _find_and_patch_reset(self, env, wrapper):
        """ Patch `reset()` method in requested wrapper to ignore the `seed` keyword argument"""
        def pop_seed_and_call(function, **kwargs):
            if "seed" in kwargs:
                kwargs.pop("seed")
            return function(**kwargs)

        current_env = env
        while hasattr(current_env, "env"):
            if isinstance(current_env, wrapper):
                current_env.reset = partial(pop_seed_and_call, function=current_env.env.reset)
                return
            current_env = current_env.env

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
        # lvl_gen.add_boulder(info["boulders"][1])
        for b in info["boulders"]:
            lvl_gen.add_boulder(b)
        # lvl_gen.add_fountain((7, 3))
        # lvl_gen.add_fountain(info["fountains"][0])
        # lvl_gen.add_fountain(info["fountains"][1])
        # lvl_gen.add_fountain(info["fountains"][2])
        for f in info["fountains"]:
            lvl_gen.add_fountain(f)
        lvl_gen.set_start_pos(info["player"])
        # lvl_gen.fill_terrain("fillrect", " ", 0, 4, 10, 9)
        return lvl_gen

    def reset(self, level=None, **kwargs):
        super().reset()
        if level is None and not self._random_goals:
            level = self._default_level
        env = self.env.unwrapped
        des_file = self.get_lvl_gen(level).get_des()
        env.update(des_file)
        if "seed" in kwargs:
            kwargs.pop("seed")
        obs, info = super(minihack.MiniHackNavigation, env).reset(**kwargs)
        env._goal_pos_set = env._object_positions(env.last_observation, "{")
        env.seed(0, 0, reseed=False)
        return obs, info


class ObsWrapper(gym.ObservationWrapper):
    """Wrapper for MiniHack environments that adds a desired goal to the observation."""

    def __init__(self, env):
        super().__init__(env)
        observation_space = Box(
            low=0,
            high=255,
            shape=(2, 10, 10),
            dtype=np.uint8,
        )
        goal_space = Box(
            low=0,
            high=255,
            shape=(1, 10, 10),
            dtype=np.uint8,
        )
        self._obs_translator_lut = ObsWrapper.build_lookup_table(DefaultCharValues(), CharValues())
        self.observation_space = gym.spaces.Dict(
            {
                "observation": observation_space,
                "desired_goal": goal_space,
            }
        )
        self.goal = None

    @staticmethod
    def build_lookup_table(default_vals, custom_vals):
        lut = np.full(256, custom_vals.FLOOR, dtype=np.uint8)

        # Populate the LUT from the dataclasses
        for field in fields(default_vals):
            attr_name = field.name
            # Check if the custom map also has this object
            if hasattr(custom_vals, attr_name):
                old_value = getattr(default_vals, attr_name)
                new_value = getattr(custom_vals, attr_name)

                # Set the mapping: lut[old_value] = new_value
                lut[old_value] = new_value

        return lut

    def _update_goal(self, observation):
        self.goal = np.zeros_like(observation["observation"][:1])
        self.goal[0][observation["observation"][1] == CharValues.FOUNTAIN] = CharValues.BOULDER
        observation["desired_goal"] = self.goal
        return observation

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        obs = self._update_goal(obs)
        return obs, info

    def observation(self, observation):
        agent_obs = copy.deepcopy(observation["chars"][7:-4, 34:-35])
        agent_obs = self._obs_translator_lut[agent_obs]
        walls_obs = np.zeros_like(agent_obs)
        walls_obs[agent_obs == CharValues.WALL] = CharValues.WALL
        walls_obs[agent_obs == CharValues.FOUNTAIN] = CharValues.FOUNTAIN
        agent_obs[agent_obs == CharValues.WALL] = CharValues.FLOOR
        agent_obs[agent_obs == CharValues.FOUNTAIN] = CharValues.FLOOR
        agent_obs[agent_obs == CharValues.START] = CharValues.FLOOR
        return {"observation": np.array([agent_obs, walls_obs]), "desired_goal": self.goal}


class MiniHackEnv(GymEnv):
    """Expands the GymEnv interface.
    Actions:
      0: North
      1: East
      2: South
      3: West
    """
    def __init__(
        self,
        *args,
        seed=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._env.set_wrapper_attr("_elapsed_steps", 0)
        self.seed(seed=seed)

    @staticmethod
    def success_fn(observation, goal=None):
        obs = observation["observation"]
        if goal is None:
            goal = observation["desired_goal"]
        if CharValues.AGENT in goal:
            return np.allclose(obs, goal)
        def get_positions_set(state, value):
            return set(map(tuple, np.argwhere(state.squeeze() == value)))
        boulders_pos_set = get_positions_set(obs, CharValues.BOULDER)
        goal_boulders_pos_set = get_positions_set(goal, CharValues.BOULDER)
        return boulders_pos_set <= goal_boulders_pos_set

    def step(self, action):
        observation, reward, terminated, truncated, self._turn, info = super().step(action)
        terminated = self.success_fn(observation)
        # truncated = bool(terminated)
        if self._env.render_mode == "human":
            os.system("cls" if os.name == "nt" else "clear")
            self.render()
            time.sleep(0.25)
        return observation, reward, terminated, truncated, self._turn, info

    def close(self):
        # TODO check if any wrapper should be closed
        pass

    def randomize_goal(self):
        self._env.env._random_goals = True  # TODO

    def reset_goal(self):
        self._env.env._random_goals = False  # TODO

    def teleport(self, _):
        observation, _ = super().reset()
        return observation

    def place_subgoal(self, pos):
        pass

    def remove_subgoal(self):
        pass

    def register_logger(self, logger):
        self.logger = logger

    def save(self, _):
        pass


def reward_fn(observation, goal=None, env_reward=0, **kwargs):
    bonus = -1 + float(MiniHackEnv.success_fn(observation, goal))
    return bonus # + env_reward


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
            ObsWrapper,
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
            ObsWrapper,
        ],
        seed=seed,
        max_episode_steps=eval_max_steps,
        **kwargs
    )
    obs, _ =  train_env.reset()
    initial_states = np.expand_dims(obs["observation"][:1], axis=0)
    goal_states = np.expand_dims(obs["desired_goal"], axis=0)
    forward_demos = None
    backward_demos = None
    return ResetFreeEnv(
        train_env=lambda: train_env,
        eval_env=lambda: eval_env,
        success_fn=MiniHackEnv.success_fn,
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
