from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from hive.envs import GymEnv
from hive.utils.schedule import PeriodicSchedule, ConstantSchedule
from minigrid.wrappers import (
    ReseedWrapper as _ReseedWrapper,
    ImgObsWrapper,
    ObservationWrapper,
)
from envs.utils import heatmap
import wandb
import matplotlib.ticker as tkr

from gymnasium.vector.sync_vector_env import SyncVectorEnv
from envs import minigrid_envs
from itertools import product
from functools import partial
from envs.reset_free_envs import ResetFreeEnv
from envs.types import UpdateInfo
import copy
from collections import defaultdict
import pickle
import os

try:
    import matplotlib
    matplotlib.use("TkAgg")
except:
    pass


class ReseedWrapper(_ReseedWrapper):
    def reset(self, seed=None, **kwargs):
        return super().reset(**kwargs)


class RGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, self.env.height * tile_size, self.env.width * tile_size),
            dtype="uint8",
        )

        self.observation_space = gym.spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=False, tile_size=self.tile_size)
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        return {**obs, "image": rgb_img}


def create_every_eval_env(env, env_fn):
    obs, _ = env.reset()
    grid = env.grid

    # Iterate through positions in grid and see which ones are potential
    # starting positions
    locations, env_fns = zip(
        *[
            ((x, y), partial(env_fn, starting_pos=(x, y)))
            for (x, y) in product(range(grid.width), range(grid.height))
            if grid.get(x, y) is None or grid.get(x, y).can_overlap()
        ]
    )
    locations = np.array(locations)
    states = np.tile(obs["observation"], (len(locations), 1, 1, 1))
    states[:, 0] = 0
    states[np.arange(len(locations)), 0, locations[:, 0], locations[:, 1]] = 255
    return SyncVectorEnv(env_fns), list(states)


def create_env_fn(env_name, seed, symbolic, **kwargs):
    def create_env(starting_pos=None):
        if starting_pos is not None:
            if "Empty" in env_name:
                kwargs["agent_start_pos"] = starting_pos
            else:
                kwargs["agent_pos"] = starting_pos

        env = gym.make(env_name, symbolic=symbolic, **kwargs)
        env = ReseedWrapper(env, seeds=[seed])
        if not symbolic:
            env = ImgObsWrapper(RGBImgObsWrapper(env, tile_size=5))
            env.action_space = gym.spaces.Discrete(3)
        return env

    return create_env


class MiniGridEnv(GymEnv):
    """An adapter for the MiniGrid environment to work with RLHive. Also provides for
    reset free training."""

    def __init__(
        self,
        env_name,
        num_players=1,
        seed=0,
        reset_free=False,
        eval=False,
        eval_every=False,
        local_vis_period=500,
        vis_frequency=5000,
        video_period=-1,
        symbolic=True,
        train_video=False,
        video_length=100,
        train_video_period=20,
        id="env",
        render_mode=None,
        **kwargs,
    ):
        self._reset_free = reset_free
        self._eval = eval
        self._eval_every = eval_every
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        # if self._eval:
        #     render_mode = "rgb_array_list" if not eval_every else None
        # else:
        #     render_mode = "rgb_array_list" if train_video else None
        # render_mode = None
        self.render_mode = render_mode
        self._train_video = (not self._eval) and train_video
        self._video_reset_schedule = PeriodicSchedule(False, True, video_length)
        self._video_write_schedule = PeriodicSchedule(False, True, train_video_period)

        super().__init__(
            env_name,
            num_players=num_players,
            seed=seed,
            symbolic=symbolic,
            render_mode=self.render_mode,
            highlight=False,
            **kwargs,
        )
        self._visitation_counts = np.zeros((self._height, self._width))
        self._local_visitation_counts = np.zeros((self._height, self._width))
        self._local_vis_period = local_vis_period
        self._local_states = deque()
        self._vis_period = PeriodicSchedule(False, True, vis_frequency)
        self._id = id
        self._num_episodes = 0
        self._has_reset = False
        if video_period == -1:
            self._video_schedule = ConstantSchedule(False)
        else:
            self._video_schedule = PeriodicSchedule(False, True, video_period)

    def create_env(self, env_name, env_wrappers, seed, symbolic, **kwargs):
        env_fn = create_env_fn(env_name, seed, symbolic, **kwargs)
        env = env_fn()
        self._width, self._height = env.width, env.height
        if self._eval_every:
            self._env, self._initial_states = create_every_eval_env(env, env_fn)
        else:
            self._env = env

    def gen_all_obs(self):
        if not self._has_reset:
            self._env.reset()
        return self._env.gen_all_obs()

    def visualize(self, prefix):
        fig1, ax1 = plt.subplots()
        if not os.path.exists("npdir"):
            os.makedirs("npdir")
        with open(os.path.join("npdir", "visitation_counts.pkl"), "wb") as f:
            pickle.dump(
                {
                    "global": self._visitation_counts,
                    "local": self._local_visitation_counts,
                },
                f,
            )
        heatmap(
            self._visitation_counts,
            np.arange(self._height),
            np.arange(self._width),
            ax1,
            logscale=True,
        )
        fig1.tight_layout()

        fig2, ax2 = plt.subplots()
        heatmap(
            self._local_visitation_counts,
            np.arange(self._height),
            np.arange(self._width),
            ax2,
            logscale=True,
            vmin=1,
            vmax=self._local_vis_period,
        )
        fig2.tight_layout()
        self._logger.log_metrics(
            {
                "global_visitation": wandb.Image(fig1),
                "local_visitation": wandb.Image(fig2),
            },
            prefix,
        )
        plt.close("all")

    def register_logger(self, logger):
        self._logger = logger

    def step(self, action):
        observation, reward, terminated, truncated, self._turn, info = super().step(
            action
        )
        if self._reset_free:
            truncated, terminated = False, False
        reward = reward != 0
        if not self._eval_every:
            pos_x, pos_y = self._env.agent_pos

            self._visitation_counts[pos_y][pos_x] += 1
            self._local_visitation_counts[pos_y][pos_x] += 1
            self._local_states.append((pos_x, pos_y))
            if len(self._local_states) > self._local_vis_period:
                old_pos_x, old_pos_y = self._local_states.popleft()
                self._local_visitation_counts[old_pos_y][old_pos_x] -= 1
            if self._vis_period.update():
                self.visualize(self._id)

        #print(f"=== self._env.agent_pos: {self._env.agent_pos}")
        #plt.pause(1)
        if self._train_video and self._video_reset_schedule.update():
            if self.render_mode == "rgb_array_list":
                frames = np.array(self._env.render())
                if self._video_write_schedule.update():
                    frames = frames.transpose(0, 3, 1, 2)
                    self._logger.log_scalar("video", wandb.Video(frames), self._id)
        return observation, reward, terminated, truncated, self._turn, info

    def reset(self):
        self._has_reset = True
        if not self._eval_every and self._video_schedule.update():
            if self.render_mode == "rgb_array_list":
                frames = np.array(self._env.render())
                frames = frames.transpose(0, 3, 1, 2)
                self._logger.log_scalar("video", wandb.Video(frames), self._id)
        return super().reset()

    def teleport(self, state):
        pos = np.flip(np.argwhere(state[0] == 255)[..., -2:].squeeze()).tolist()
        #print(f"=== teleporting to: {pos}")
        return self._env.teleport(pos)

    def save(self, folder_name):
        pass


def create_vis_fn(env_name):
    if env_name == "MiniGrid-Empty-16x16-v1":
        width, height = 16, 16
    elif env_name == "MiniGrid-Empty-8x8-v1":
        width, height = 8, 8
    elif env_name == "MiniGrid-FourRooms-v1":
        width, height = 19, 19
    elif env_name == "MiniGrid-TwoRooms-v1":
        width, height = 19, 10
    else:
        raise ValueError(f"{env_name} not supported")
    storage = defaultdict(list)

    def vis_fn(
        locations,
        log_fn,
        width,
        height,
        totals=1,
        max_count=None,
        fmt=".0f",
        already_counts=False,
        name=None,
    ):
        if not already_counts:
            locations = np.array(locations)
            _, y, x = np.nonzero(locations[:, 0])
            counts = np.zeros((height, width))
            np.add.at(counts, (np.array(y), np.array(x)), totals)
        else:
            counts = locations
        if name is not None:
            storage[name].append(counts)
            pickle.dump(storage, open("npdir/storage.pkl", "wb"))

        fig, ax = plt.subplots()
        formatter = tkr.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        heatmap(
            counts,
            np.arange(height),
            np.arange(width),
            ax,
            mask=counts == 0,
            annot=True,
            fmt=fmt,
            vmin=0,
            vmax=max_count,
        )
        fig.tight_layout()
        image = wandb.Image(fig)
        plt.close("all")
        return image, counts

    return partial(vis_fn, width=width, height=height)


def get_goals(env_name, all_obs):
    goals = {
        "MiniGrid-Empty-16x16-v1": (14, 14),
        "MiniGrid-Empty-8x8-v1": (6, 6),
        "MiniGrid-FourRooms-v1": (17, 17),
        "MiniGrid-TwoRooms-v1": (17, 8),
    }
    goal = goals[env_name]
    return [g for g in all_obs["desired_goal"] if g[0, goal[0], goal[1]] == 255]


def get_distance_calculator(distance_type, initial_state):
    initial_state_loc = np.concatenate(np.nonzero(initial_state[0]))

    def distance_calculator(obs: np.ndarray):  # , goal):
        agent_loc = np.concatenate(np.nonzero(obs[0]))

        return np.abs(agent_loc - initial_state_loc).sum()

    return distance_calculator


def success_fn(observation, goal=None):
    obs = observation["observation"]
    if goal is None:
        goal = observation["desired_goal"]
    return np.allclose(obs[0], goal[0])


def reward_fn(observation, goal=None):
    return float(success_fn(observation, goal))


def replace_goal_fn(obs, goal):
    obs = copy.deepcopy(obs)
    obs["desired_goal"] = goal
    return obs


def get_minigrid_envs(
    env_name,
    symbolic: bool = True,
    seed: int = 0,
    eval_every=True,
    reset_free=True,
    video_period=-1,
    train_max_steps=100,
    eval_max_steps=100,
    **kwargs,
):
    """Create a reset-free environment for MiniGrid.

    Args:
        env_name: The name of the MiniGrid environment.
        symbolic: Whether to use symbolic observations.
        seed: The seed for the environment.
        eval_every: Whether to evaluate the agent at every single possible location.
        reset_free: Whether to use reset-free training.
        video_period: The period at which to save videos.
        train_max_steps: The maximum number of steps in the training environment.
        eval_max_steps: The maximum number of steps in the evaluation environment.
        kwargs: Additional arguments to pass to the environment.
    """
    env = MiniGridEnv(
        env_name=env_name,
        seed=seed,
        reset_free=reset_free,
        eval=False,
        symbolic=symbolic,
        id="train_env",
        eval_every=False,
        video_period=-1,
        max_steps=train_max_steps,
        **kwargs,
    )
    initial_obs, _ = env.reset()
    all_obs = env.gen_all_obs()

    return ResetFreeEnv(
        train_env=partial(
            MiniGridEnv,
            env_name=env_name,
            seed=seed,
            reset_free=reset_free,
            eval=False,
            symbolic=symbolic,
            id="train_env",
            eval_every=False,
            video_period=-1,
            max_steps=train_max_steps,
            **kwargs,
        ),
        eval_env=partial(
            MiniGridEnv,
            env_name=env_name,
            seed=seed,
            reset_free=False,
            eval=True,
            symbolic=symbolic,
            id="eval_env",
            eval_every=eval_every,
            video_period=video_period,
            max_steps=eval_max_steps,
            **kwargs,
        ),
        success_fn=success_fn,
        reward_fn=reward_fn,
        replace_goal_fn=replace_goal_fn,
        all_states_fn=lambda *args: all_obs,
        vis_fn=create_vis_fn(env_name),
        get_distance_fn=partial(
            get_distance_calculator, initial_state=initial_obs["observation"]
        ),
        goal_states=get_goals(env_name, all_obs),
        initial_states=[initial_obs["observation"][0:1]],
        forward_demos=None,
        backward_demos=None,
        eval_every=eval_every,
    )
