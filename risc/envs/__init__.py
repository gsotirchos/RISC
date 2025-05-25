from hive.utils.registry import registry
from envs.atari import get_atari_envs
from envs.minigrid_rf import get_minigrid_envs
from envs.earl import get_earl_envs
from envs.reset_free_envs import ResetFreeEnv

registry.register_all(
    ResetFreeEnv,
    {
        "atari_envs": get_atari_envs,
        "minigrid_envs": get_minigrid_envs,
        "earl_envs": get_earl_envs,
    },
)
