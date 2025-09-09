from __future__ import annotations

from enum import IntEnum

import gymnasium as gym
import minigrid
import numpy as np
from gymnasium.envs.registration import register
from minigrid.core.constants import DIR_TO_VEC, COLORS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, WorldObj, Floor
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_triangle
from minigrid.core.roomgrid import RoomGrid as _RoomGrid
from minigrid.envs.empty import EmptyEnv as _EmptyEnv
# from minigrid.envs.fourrooms import FourRoomsEnv as _FourRoomsEnv


class Floor(WorldObj):
    def __init__(self, color: str = "grey"):
        super().__init__("floor", color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), (0, 0, 0))


class Subgoal(WorldObj):
    def __init__(self):
        super().__init__("floor", "blue")

    def can_overlap(self):
        return True

    def can_pickup(self):
        return False

    def can_contain(self):
        return False

    def see_behind(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Slide(WorldObj):
    def __init__(self, action, color):
        self.action = action
        super().__init__(type="wall", color=color)

    def can_overlap(self):
        return True

    def can_pickup(self):
        return False

    def can_contain(self):
        return False

    def see_behind(self):
        return True

    def _draw_arrows(self, action, img, color):
        tri1_pts = np.array([
            [0.55, 0.5],
            [ 0.1, 0.1],
            [ 0.1, 0.9],
        ])

        tri2_pts = np.array([
            [0.95, 0.5],
            [0.5,  0.1],
            [0.5,  0.9],
        ])

        rotations = {
            0: [[-1,  0], [ 0, -1]],
            1: [[ 0,  1], [-1,  0]],
            2: [[ 1,  0], [ 0,  1]],
            3: [[ 0, -1], [ 1,  0]],
        }

        rot_mat = rotations[action]

        # Apply the rotation to the vertices and add 0.5 to center them on the tile
        center_offset = np.array([0.5, 0.5])
        tri1_rotated = np.dot(tri1_pts - center_offset, rot_mat) + center_offset
        tri2_rotated = np.dot(tri2_pts - center_offset, rot_mat) + center_offset

        # Draw the triangles
        fill_coords(img, point_in_triangle(*tri1_rotated), color)
        fill_coords(img, point_in_triangle(*tri2_rotated), color)

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS["purple"])
        self._draw_arrows(self.action, img, COLORS["purple"] / 2)


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
            self.gen_grid_obs()
            self.gen_goal_obs()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.gen_grid_obs()
        self.gen_goal_obs()
        #print("=== resetting")
        return self.gen_obs(), {}

    def gen_grid_obs(self):
        self.grid_obs = np.zeros((2, self.height, self.width), dtype=np.uint8)
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell:
                    if cell.type == "wall":
                        self.grid_obs[1, j, i] = 255

    def gen_goal_obs(self):
        self.goal_obs = np.zeros((1, self.height, self.width), dtype=np.uint8)
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell:
                    if cell.type == "goal":
                        self.goal_obs[0, j, i] = 255

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
        return {"observation": obs, "desired_goal": goals}

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        #print(f"     State: {np.flip(np.argwhere(self.gen_obs()['observation'][0] == 255)[..., -2:].squeeze(), axis=-1).tolist()}")
        #print(f"    Action: {action}")

        # Get the position in front of the agent
        try:
            fwd_pos = self.agent_pos + DIR_TO_VEC[action]
        except IndexError as err:
            raise ValueError(f"Unknown action: {action}") from err

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = tuple(fwd_pos)
            self.agent_dir = action
        if fwd_cell is not None and fwd_cell.color == "purple":
            slide_action = fwd_cell.action
            self.agent_pos += DIR_TO_VEC[slide_action]
            self.agent_dir = slide_action
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
            if not fwd_cell.is_open:
                fwd_cell.toggle(self, fwd_pos)
            if fwd_cell.is_open:
                self.agent_pos = tuple(fwd_pos)
                self.agent_dir = action
        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()
        #print(f"Next state: {np.flip(np.argwhere(obs['observation'][0] == 255)[..., -2:].squeeze(), axis=-1).tolist()}")
        #breakpoint()

        return obs, reward, terminated, truncated, {}

    def place_agent(self, pos=None):
        if pos is None:
            super().place_agent()  # random position and orientation
        else:
            self.agent_pos = pos
            self.grid.set(*pos, None)
            self.agent_dir = self._rand_int(0, 4)  # random direction

    def place_goal(self, pos=None):
        if pos is None:
            self.place_obj(Goal())  # random position
        else:
            self.put_obj(Goal(), *pos)

    def place_subgoal(self, pos):
        self.put_obj(Subgoal(), *pos)

    def place_floor(self, pos):
        self.put_obj(Floor(), *pos)

    def teleport(self, agent_pos=None):
        #print(f"=== teleporting to: {agent_pos}")
        self.place_agent(agent_pos)
        return self.gen_obs()

    def render(self):
        return super().render()

    def close(self):
        self.window = None
        super().close()


class FourRoomsEnv(MiniGridEnv):  #, _FourRoomsEnv):
    """Four rooms environment."""

    def __init__(self, agent_pos=(1, 1), goal_pos=(17, 17), max_steps=100, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.width = 19
        self.height = 19
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
        room_h = height // 2

        self.grid.vert_wall(room_w, 0, height)
        self.grid.horz_wall(0, room_h, width)

        doors_pos = (
                (room_w, 7),
                (room_w, 12),
                (6, room_h),
                (14, room_h),
        )

        for door_pos in doors_pos:
            self.grid.set(*door_pos, None)

        self.place_agent(self._agent_default_pos)
        self.place_goal(self._goal_default_pos)


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
        self.agent_pos = agent_pos
        self.goal_pos = goal_pos
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

        self.place_agent(self._agent_default_pos)
        self.place_goal(self._goal_default_pos)


class TinyRoomEnv(MiniGridEnv):
    """Tiny room with an obstacle environment."""

    def __init__(self, agent_pos=(1, 1), goal_pos=(5, 6), max_steps=100, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.width = 7
        self.height = 8
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

        # center-top and -bottom obstacles
        self.grid.wall_rect(x=3, y=1, w=1, h=1)
        self.grid.wall_rect(x=3, y=height - 2, w=1, h=1)

        # center obstacle
        self.grid.horz_wall(x=2, y=height - 4, length=width - 4)
        self.grid.vert_wall(x=width // 2, y=3, length=height - 6)

        self.place_agent(self._agent_default_pos)
        self.place_goal(self._goal_default_pos)


class BugTrapEnv(MiniGridEnv):
    """Bug trap environment."""

    def __init__(self, agent_pos=(6, 9), goal_pos=(9, 9), max_steps=100, **kwargs):
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

        # Generate bug trap walls
        self.grid.vert_wall(int(width * 0.45), int(height * 0.35), length=int(height * 0.35))
        self.grid.horz_wall(int(width * 0.45), int(height * 0.35), length=int(width * 0.35))
        self.grid.horz_wall(int(width * 0.45), int(height * 0.65), length=int(width * 0.35))

        ### Old bugtrap
        # # top and bottom horizontal wall of the bugtrap
        # self.grid.horz_wall(3, 3, length=width - 6)
        # self.grid.horz_wall(3, height - 4, length=width - 6)
        #
        # # middle entrance of the bugtrap
        # self.grid.horz_wall(3, height // 2 - 1, length=width // 2 - 3)
        # self.grid.horz_wall(3, height // 2 + 1, length=width // 2 - 3)
        #
        # # left and right vertical walls of the bugtrap
        # self.grid.vert_wall(3, 3, length=height // 2 - 3)
        # self.grid.vert_wall(3, height // 2 + 1, length=height // 2 - 4)
        # self.grid.vert_wall(width - 4, 3, length=height - 6)

        self.place_agent(self._agent_default_pos)
        self.place_goal(self._goal_default_pos)


class HallwayEnv(MiniGridEnv):
    """Hallway environment."""

    def __init__(
        self,
        room_size=19,
        goal_pos=(9, 9),
        agent_pos=(16, 9),
        hallway_length=5,
        num_hallways=3,
        max_steps=100,
        **kwargs
    ):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self._hallway_length = hallway_length
        self._num_hallways = num_hallways
        self._hallway_start_x = goal_pos[0] - hallway_length
        self._hallway_end_x = goal_pos[0]
        self._hallway_y = goal_pos[1]

        if isinstance(room_size, tuple):
            self.width, self.height = room_size
        else:
            self.width = self.height = room_size
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

        # Generate main and decoy hallway walls
        for i in range(self._num_hallways):
            offset = int(np.ceil(i / 2) * np.ceil(self.height / 4)) * (- 1 + int(i == 0)) ** i
            for x in range(self._hallway_start_x, self._hallway_end_x):
                self.put_obj(Slide(self.actions.up, "purple"), x, self._hallway_y + offset - 1)
                self.put_obj(Slide(self.actions.down, "red"), x, self._hallway_y + offset + 1)
            try:
                self.grid.vert_wall(self._hallway_end_x + 1, self._hallway_y + offset - 1, 3)
            except Exception:
                pass
            self.grid.horz_wall(self._hallway_end_x, self._hallway_y + offset - 1, 1)
            self.grid.horz_wall(self._hallway_end_x, self._hallway_y + offset + 1, 1)

        self.place_agent(self._agent_default_pos)
        self.place_goal(self._goal_default_pos)


class LockedDoorEnv(MiniGridEnv, _RoomGrid):
    """Single locked door environment."""

    def __init__(self, agent_pos=(7, 1), goal_pos=(14, 4), max_steps=50, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        room_size = 6
        mission_space = MissionSpace(mission_func=self._gen_mission)

        _RoomGrid.__init__(
            self,
            mission_space=mission_space,
            num_rows=1,
            num_cols=3,
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

        # Make sure the rooms are directly connected by a locked door
        door, _ = self.add_door(1, 0, 0, locked=True)
        _, _ = self.add_door(0, 0, 0, locked=False)

        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(self._agent_default_pos)
        self.place_goal(self._goal_default_pos)


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
register(id="MiniGrid-TinyRoom-v1", entry_point="envs.minigrid_envs:TinyRoomEnv")
register(id="MiniGrid-BugTrap-v1", entry_point="envs.minigrid_envs:BugTrapEnv")
register(id="MiniGrid-Hallway-v1", entry_point="envs.minigrid_envs:HallwayEnv")
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
