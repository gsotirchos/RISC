from __future__ import annotations

from enum import IntEnum

import gymnasium as gym
import minigrid
import numpy as np
from gymnasium.envs.registration import register
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.envs.empty import EmptyEnv as _EmptyEnv
from minigrid.envs.fourrooms import FourRoomsEnv as _FourRoomsEnv


class MiniGridEnv(minigrid.minigrid_env.MiniGridEnv):
    """Base class for MiniGrid environments. This class provides a representation of the
    state with one layer representing the grid, one layer representing the agent's
    position, and one layer representing the goal position."""

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        right = 0
        down = 1
        left = 2
        up = 3

    def __init__(
        self,
        mission_space: MissionSpace,
        grid_size: int = None,
        width: int = None,
        height: int = None,
        max_steps: int = 100,
        symbolic: bool = True,
        **kwargs,
    ):
        minigrid.minigrid_env.MiniGridEnv.__init__(
            self,
            mission_space,
            grid_size,
            width,
            height,
            max_steps,
            **kwargs,
        )
        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.symbolic = symbolic
        if self.symbolic:
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(
                        0, 255, (2, self.height, self.width), dtype=np.uint8
                    ),
                    "desired_goal": gym.spaces.Box(
                        0, 255, (1, self.height, self.width), dtype=np.uint8
                    ),
                }
            )
            self.grid_obs = np.zeros((2, self.height, self.width), dtype=np.uint8)
            self.goal_obs = np.zeros((1, self.height, self.width), dtype=np.uint8)
            for i in range(self.width):
                for j in range(self.height):
                    cell = self.grid.get(i, j)
                    if cell:
                        if cell.type == "wall":
                            self.grid_obs[1, j, i] = 255
                        elif cell.type == "goal":
                            self.goal_obs[0, j, i] = 255

    def reset(self, *, seed=None, options=None):
        x = super().reset(seed=seed, options=options)
        self.grid_obs = np.zeros((2, self.height, self.width), dtype=np.uint8)
        self.goal_obs = np.zeros((1, self.height, self.width), dtype=np.uint8)

        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell:
                    if cell.type == "wall":
                        self.grid_obs[1, j, i] = 255
                    elif cell.type == "goal":
                        self.goal_obs[0, j, i] = 255
        return self.gen_obs(), {}

    def gen_obs(self):
        if not self.symbolic:
            return super().gen_obs()
        obs = np.copy(self.grid_obs)
        obs[0, self.agent_pos[1], self.agent_pos[0]] = 255
        return {"observation": obs, "desired_goal": self.goal_obs}

    def gen_all_obs(self):
        obs = np.tile(
            self.grid_obs,
            (self.height * self.width, *((1,) * len(self.grid_obs.shape))),
        )
        idx = np.arange(self.height * self.width)
        obs[idx, 0, idx // self.width, idx % self.width] = 255
        valid_idxs = obs[idx, 1, idx // self.width, idx % self.width] < 255
        obs = obs[valid_idxs]
        goals = np.tile(
            self.goal_obs,
            (len(obs), *((1,) * len(self.goal_obs.shape))),
        )

        return {
            "observation": obs,
            "desired_goal": goals,
        }

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        try:
            fwd_pos = self.agent_pos + DIR_TO_VEC[action]
        except IndexError:
            raise ValueError(f"Unknown action: {action}")

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = tuple(fwd_pos)
            self.agent_dir = action
        if fwd_cell is not None and fwd_cell.type == "goal":
            terminated = True
            reward = 1.0  # self._reward()
        if fwd_cell is not None and fwd_cell.type == "lava":
            terminated = True
        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def render(self):
        return super().render()


class FourRoomsEnv(MiniGridEnv, _FourRoomsEnv):
    """Four rooms environment."""

    def __init__(self, agent_pos=None, goal_pos=None, max_steps=100, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.size = 19
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )


class TwoRoomsEnv(MiniGridEnv):
    """Two rooms environment."""

    def __init__(
        self,
        agent_pos=None,
        goal_pos=None,
        max_steps=100,
        width=19,
        height=10,
        **kwargs,
    ):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.width = width
        self.height = height
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        self.grid.vert_wall(room_w, 0)
        pos = (room_w, self._rand_int(1, height))
        self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())


class EmptyEnv(MiniGridEnv, _EmptyEnv):
    """Empty environment."""

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )


register(id="MiniGrid-TwoRooms-v1", entry_point="envs.minigrid_envs:TwoRoomsEnv")
register(id="MiniGrid-FourRooms-v1", entry_point="envs.minigrid_envs:FourRoomsEnv")
register(
    id="MiniGrid-Empty-Random-5x5-v1",
    entry_point="envs.minigrid_envs:EmptyEnv",
    kwargs={"size": 5, "agent_start_pos": None},
)

register(
    id="MiniGrid-Empty-6x6-v1",
    entry_point="envs.minigrid_envs:EmptyEnv",
    kwargs={"size": 6},
)

register(
    id="MiniGrid-Empty-Random-6x6-v1",
    entry_point="envs.minigrid_envs:EmptyEnv",
    kwargs={"size": 6, "agent_start_pos": None},
)

register(
    id="MiniGrid-Empty-8x8-v1",
    entry_point="envs.minigrid_envs:EmptyEnv",
)

register(
    id="MiniGrid-Empty-16x16-v1",
    entry_point="envs.minigrid_envs:EmptyEnv",
    kwargs={"size": 16},
)
