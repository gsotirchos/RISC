from collections.abc import Iterable

import numpy as np

from replays.circular_replay import CircularReplayBuffer


class HashStorage(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    @staticmethod
    def to_hashable(obj):
        def _to_hashable(obj):
            if isinstance(obj, Iterable):
                return tuple(_to_hashable(sub_obj) for sub_obj in obj)
            else:
                return obj
        return _to_hashable(obj)

    @staticmethod
    def hashable_key(method):
        def wrapper(self, key, *args, **kwargs):
            key = self.to_hashable(key)
            return method(self, key, *args, **kwargs)
        return wrapper

    @hashable_key
    def __getitem__(self, *args, **kwargs):
        return super().__getitem__(*args, **kwargs)

    @hashable_key
    def __setitem__(self, *args, **kwargs):
        super().__setitem__(*args, **kwargs)

    @hashable_key
    def __delitem__(self, *args, **kwargs):
        super().__delitem__(*args, **kwargs)

    @hashable_key
    def __contains__(self, *args, **kwargs):
        return super().__contains__(*args, **kwargs)

    @hashable_key
    def get(self, *args, **kwargs):
        return super().get(*args, **kwargs)

    @hashable_key
    def pop(self, *args, **kwargs):
        return super().pop(*args, **kwargs)

    def update(self, other=None, **kwargs):
        if other is not None:
            if hasattr(other, 'items'):
                for k, v in other.items():
                    self.__setitem__(k, v)
            else:
                for k, v in other:
                    self.__setitem__(k, v)
        for k, v in kwargs.items():
            self.__setitem__(k, v)


class SymmetricHashStorage(HashStorage):
    def to_hashable(self, *args, **kwargs):
        return tuple(sorted(super().to_hashable(*args, **kwargs)))


def distance(state1, state2):
    """Compute distance for both symbolic and continuous states."""
    state1, state2 = np.squeeze(state1), np.squeeze(state2)
    if state1.ndim == state2.ndim == 3:
        state1 = np.argwhere(state1[0] == 255).flatten()
        state2 = np.argwhere(state2[0] == 255).flatten()
    return np.linalg.norm(state1 - state2)


class CountsReplayBuffer(CircularReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = HashStorage()
        self.distances = SymmetricHashStorage()

    def _update_distances(self, state):
        # print(f"updating distances for : {np.argwhere(state[0] == 255).flatten()}")
        if self.counts[state] > 1:
            # print("state already there")
            # breakpoint()
            return
        for other_state in self.counts.keys():
            # print(f"key: ({np.argwhere(state[0] == 255).flatten()}, {np.argwhere(np.array(other_state)[0] == 255).flatten()})...")
            if (state, other_state) in self.distances:
                # print("already in self.distances")
                # breakpoint()
                continue
            # print("calculating distance")
            # breakpoint()
            self.distances[state, other_state] = distance(state, other_state)

    def _update_metadata(self, new_state):
        """ If a transition will be overwritten decrement its state's count and if it has
        no counts left remove its entry. Then increment the count for the new state. """
        overwritten_state = self._storage["observation"][self._cursor]
        if not np.all(overwritten_state == 0):
            self.counts[overwritten_state] -= 1
            if self.counts[overwritten_state] == 0:
                # for other_state in self.counts.keys():
                #     del self.distances[overwritten_state, other_state]
                del self.counts[overwritten_state]
        if not np.all(new_state == 0):
            self.counts[new_state] = self.counts.get(new_state, 0) + 1
            self._update_distances(new_state)

    def _add_transition(self, **transition):
        self._update_metadata(transition["observation"])
        super()._add_transition(**transition)
