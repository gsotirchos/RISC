from hive.utils.registry import Registrable
from hive.envs import BaseEnv
from dataclasses import dataclass
from typing import Callable
from envs.types import UpdateInfo
import numpy as np
from typing import Any


def do_nothing(*args, **kwargs):
    pass


@dataclass
class ResetFreeEnv(Registrable):
    """Base class for reset free environments."""

    train_env: BaseEnv
    eval_env: BaseEnv
    success_fn: Callable[[UpdateInfo], bool]
    reward_fn: Callable[[UpdateInfo], float]
    replace_goal_fn: Callable[[Any, np.ndarray], Any] = lambda x, y: x
    all_states_fn: Callable[[], np.ndarray] = do_nothing
    vis_fn: Callable = do_nothing
    get_distance_fn: Callable[[Any], Callable[[Any], float]] = do_nothing
    goal_states: np.ndarray = None
    initial_states: np.ndarray = None
    forward_demos: dict = None
    backward_demos: dict = None
    eval_every: bool = False

    @classmethod
    def type_name(cls):
        return "reset_free_env"
