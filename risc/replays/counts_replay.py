from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from replays.circular_replay import CircularReplayBuffer


class HashStorage(defaultdict):
    def __init__(self, other=None, *args, **kwargs):
        super().__init__(lambda: HashStorage(), *args, **kwargs)
        if isinstance(other, Iterable):
            super().update(other)

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
        if not self.__contains__(*args, **kwargs):
            return  # all good
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
        super().update(other=other, **kwargs)


class SymmetricHashStorage(HashStorage):
    def to_hashable(self, *args, **kwargs):
        return tuple(sorted(super().to_hashable(*args, **kwargs)))


class SymmetricMatrix:
    def __init__(self):
        self.views = HashStorage(HashStorage)

    @staticmethod
    def process_key(method):
        def wrapper(self, key, *args, **kwargs):
            if isinstance(key, np.ndarray):
                key = (key, None)
            elif isinstance(key, tuple):
                if len(key) != 2:
                    raise KeyError("The keys must be a pair of NumPy arrays")
                if (not isinstance(key[0], np.ndarray)
                        or not (isinstance(key[1], np.ndarray) or key[1] is None)):
                    raise KeyError("All keys nust must be NumPy arrays")
            else:
                raise KeyError("All keys nust must be NumPy arrays")
            return method(self, key, *args, **kwargs)
        return wrapper

    @process_key
    def __setitem__(self, key, value: HashStorage):
        i, j = key
        if j is None:
            self.views[i] = value
        else:
            self.views[i][j] = value
            self.views[j][i] = value

    @process_key
    def __getitem__(self, key):
        i, j = key
        if not self.__contains__(key):
            raise KeyError(key)
        if j is None:  # whole row access [i]
            return HashStorage(self.views[i])
            # return self.views[i].items()  # O(1) return a generator for lazy access
            # return self.views[i].copy()  # O(k) return a copy for safe modifications
        else:  # direct access: [i, j]
            return self.views[i][j] or self.views[j][i]

    @process_key
    def __delitem__(self, key):
        for i, j in key, reversed(key):
            if j is None:
                del self.views[i]
                for k in np.array(list(self.views.keys())):
                    del self.views[k][i]
                    if not self.views[k]:
                        del self.views[k]
                break
            del self.views[i][j]
            if not self.views[i]:
                del self.views[i]

    @process_key
    def __contains__(self, key):
        i, j = key
        if j is None:
            return i in self.views
        else:
            return j in self.views[i] or i in self.views[j]

    def get(self, key, default=None):
        return self.views.get(key, default)

    def pop(self, key, default=None):
        return self.views.pop(key, default)

    def update(self, other=None, **kwargs):
        return self.views.update(other=None, **kwargs)

    def keys(self):
        return self.views.keys()

    def values(self):
        return self.views.values()

    def items(self):
        return self.views.items()

    def __repr__(self):
        return self.views.__repr__()

    def __len__(self):
        return self.views.__len__()

    def __clear__(self):
        return self.views.__clear__()


def distance(state1, state2):
    """Compute distance for both symbolic and continuous states."""
    # return np.random.randint(32)
    state1, state2 = np.squeeze(state1), np.squeeze(state2)
    if state1.ndim == state2.ndim == 3:
        state1 = np.argwhere(state1[0] == 255).flatten()
        state2 = np.argwhere(state2[0] == 255).flatten()
    return np.linalg.norm(state1 - state2)


class CountsReplayBuffer(CircularReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = HashStorage()
        self.distances = SymmetricMatrix()

    def _remove_distances(self, state):
        for other_state in self.counts.keys():
            if (state, other_state) in self.distances:
                del self.distances[state, other_state]

    def _update_distances(self, state):
        if self.counts[state] > 1:
            return
        for other_state in np.array(list(self.counts.keys())):
            if (state, other_state) not in self.distances:
                self.distances[state, other_state] = distance(state, other_state)

    def _update_metadata(self, new_state):
        """ If a state will be overwritten decrement its count, and if it has no counts
        left remove its entry. Then increment the count for the new state. """
        overwritten_state = self._storage["observation"][self._cursor]
        if not np.all(overwritten_state == 0):
            self.counts[overwritten_state] -= 1
            if self.counts[overwritten_state] == 0:
                # self._remove_distances(overwritten_state)
                del self.distances[overwritten_state]
                del self.counts[overwritten_state]
        if not np.all(new_state == 0):
            self.counts[new_state] = self.counts.get(new_state, 0) + 1
            self._update_distances(new_state)

    def _add_transition(self, **transition):
        self._update_metadata(transition["observation"])
        super()._add_transition(**transition)
