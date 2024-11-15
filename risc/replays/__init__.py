from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.registry import registry

from replays.circular_replay import CircularReplayBuffer
from replays.her_replay import HERReplayBuffer

registry.register_all(
    BaseReplayBuffer,
    {
        "CircularReplayBuffer": CircularReplayBuffer,
        "HERReplayBuffer": HERReplayBuffer,
    }
)
