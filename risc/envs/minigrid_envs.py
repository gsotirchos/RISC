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
from minigrid.core.roomgrid import RoomGrid as _RoomGrid
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
        if fwd_cell is not None and fwd_cell.can_pickup():
            if self.carrying is None:
                self.carrying = fwd_cell
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(fwd_pos[0], fwd_pos[1], None)
            self.agent_pos = tuple(fwd_pos)
            self.agent_dir = action
        if fwd_cell is not None and fwd_cell.type == "door":
            if fwd_cell.is_locked:
                fwd_cell.toggle(self, fwd_pos)
            if fwd_cell.is_open:
                self.agent_pos = tuple(fwd_pos)
                self.agent_dir = action
        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def teleport(self, pos):
        self.agent_pos = pos
        return self.gen_obs()

    def render(self):
        return super().render()


class FourRoomsEnv(MiniGridEnv, _FourRoomsEnv):
    """Four rooms environment."""

    def __init__(self, agent_pos=(1, 1), goal_pos=(17, 17), max_steps=100, **kwargs):
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
        agent_pos=(1, 1),
        goal_pos=(17, 8),
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


class BugTrapEnv(MiniGridEnv):
    """Bug trap environment."""

    def __init__(self, agent_pos=(5, 12), goal_pos=(17, 17), max_steps=100, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.width = self.height = 19
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

        # top and bottom horizontal wall of the bugtrap
        self.grid.horz_wall(3, 3, length=width - 6)
        self.grid.horz_wall(3, height - 4, length=width - 6)

        # middle entrance of the bugtrap
        self.grid.horz_wall(3, height // 2 - 1, length=width // 2 - 3)
        self.grid.horz_wall(3, height // 2 + 1, length=width // 2 - 3)

        # left and right vertical walls of the bugtrap
        self.grid.vert_wall(3, 3, length=height // 2 - 3)
        self.grid.vert_wall(3, height // 2 + 1, length=height // 2 - 4)
        self.grid.vert_wall(width - 4, 3, length=height - 6)

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


class LockedDoorEnv(MiniGridEnv, _RoomGrid):
    """Single locked door environment."""

    def __init__(self, agent_pos=(1, 1), goal_pos=(13, 6), max_steps=50, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        room_size = 8
        mission_space = MissionSpace(mission_func=self._gen_mission)

        _RoomGrid.__init__(
            self,
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
        )

        MiniGridEnv.__init__(
            self,
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
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        self.door = door

        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

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
        agent_pos=(1, 1),
        agent_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_pos
        self.agent_start_dir = agent_dir

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
register(id="MiniGrid-BugTrap-v1", entry_point="envs.minigrid_envs:BugTrapEnv")
register(id="MiniGrid-LockedDoor-v1", entry_point="envs.minigrid_envs:LockedDoorEnv")

for size in range(4, 20, 2):
    register(
        id=f"MiniGrid-Empty-{size}x{size}-v1",
        entry_point="envs.minigrid_envs:EmptyEnv",
        kwargs={"size": size},
    )

register(
    id="MiniGrid-Empty-Random-6x6-v1",
    entry_point="envs.minigrid_envs:EmptyEnv",
    kwargs={"size": 6, "agent_pos": None},
)
