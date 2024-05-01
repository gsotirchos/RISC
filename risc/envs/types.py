from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UpdateInfo:
    observation: Any
    next_observation: Any
    reward: Any
    action: Any
    terminated: Any
    truncated: Any
    info: Any
