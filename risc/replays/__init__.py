from hive.utils.registry import registry

from replays.circular_replay import CircularReplayBuffer

registry.register_all(
    CircularReplayBuffer, {"CircularReplayBuffer": CircularReplayBuffer}
)
