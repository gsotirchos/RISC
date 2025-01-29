from collections import defaultdict
from collections.abc import Iterable, Mapping, Callable

import numpy as np

from replays.circular_replay import CircularReplayBuffer


class HashableKeyDict(defaultdict):
    def __init__(self, default_factory=None, other=None, **kwargs):
        if isinstance(default_factory, type):
            def default_factory(_type=default_factory):
                return _type()
        super().__init__(default_factory, **kwargs)
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


class SymmetricHashableKeyDict(HashableKeyDict):
    def to_hashable(self, *args, **kwargs):
        return tuple(sorted(super().to_hashable(*args, **kwargs)))


class SymmetricMatrix(Mapping):
    def __init__(self):
        self._views = HashableKeyDict(HashableKeyDict)

    @staticmethod
    def process_key(method):
        def wrapper(self, key, *args, **kwargs):
            key_error = KeyError("All keys must be NumPy arrays")
            if isinstance(key, np.ndarray):
                key = (key, None)
            elif isinstance(key, tuple):
                if len(key) != 2:
                    raise key_error
                if not isinstance(key[0], np.ndarray):
                    raise key_error
                if not (isinstance(key[1], np.ndarray) or key[1] is None):
                    raise key_error
            else:
                raise key_error
            return method(self, key, *args, **kwargs)
        return wrapper

    @process_key
    def __setitem__(self, key, value: HashableKeyDict):
        i, j = key
        if j is None:
            if not isinstance(value, HashableKeyDict):
                raise TypeError(value)
            self._views[i] = value
            for j in value:
                self._views[j][i] = value[j]
        else:
            self._views[i][j] = value
            self._views[j][i] = value

    @process_key
    def __getitem__(self, key):
        i, j = key
        if not self.__contains__(key):
            raise KeyError(key)
        if j is None:
            return dict(self._views[i])  # O(k)
            #return self._views[i].items()  # O(1) return a generator for lazy access
            #return self._views[i].copy()  # O(k) return a copy for safe modifications
        else:
            return self._views[i][j] or self._views[j][i]

    @process_key
    def __delitem__(self, key):
        for i, j in key, reversed(key):
            if j is None:
                del self._views[i]
                for k in np.array(list(self._views.keys())):
                    del self._views[k][i]
                    if not self._views[k]:
                        del self._views[k]
                break
            del self._views[i][j]
            if not self._views[i]:
                del self._views[i]

    @process_key
    def __contains__(self, key):
        i, j = key
        if j is None:
            return i in self._views
        else:
            return j in self._views[i] or i in self._views[j]

    @process_key
    def get(self, key, default=None):
        return self._views.get(key, default)

    @process_key
    def pop(self, key, default=None):
        return self._views.pop(key, default)

    def update(self, other=None, **kwargs):
        return self._views.update(other=None, **kwargs)

    def keys(self):
        return self._views.keys()

    def values(self):
        items = dict()
        for i in self._views:
            items[i] = list(self._views[i].values())
        return items.values()

    def items(self):
        items = dict()
        for i in self._views:
            items[i] = list(self._views[i].values())
        return items.items()

    def __iter__(self):
        return self._views.__iter__()

    def __len__(self):
        return self._views.__len__()

    def __repr__(self):
        return self._views.__repr__()

    def __clear__(self):
        return self._views.__clear__()


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
        self.counts = HashableKeyDict()
        self.distances = SymmetricMatrix()

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
                #del self.distances[overwritten_state]  # TODO
                del self.counts[overwritten_state]
        if not np.all(new_state == 0):
            self.counts[new_state] = self.counts.get(new_state, 0) + 1
            #self._update_distances(new_state)  # TODO

    def _add_transition(self, **transition):
        self._update_metadata(transition["observation"])
        super()._add_transition(**transition)
