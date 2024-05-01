from hive.utils.registry import registry
from hive.envs import BaseEnv
from envs.minigrid_rf import get_minigrid_envs, MiniGridEnv

from envs.earl import get_earl_envs
import envs.minigrid_envs
from envs.reset_free_envs import ResetFreeEnv

registry.register("minigrid_envs", get_minigrid_envs, ResetFreeEnv)
registry.register("earl_envs", get_earl_envs, ResetFreeEnv)
registry.register("MiniGridEnv", MiniGridEnv, BaseEnv)
