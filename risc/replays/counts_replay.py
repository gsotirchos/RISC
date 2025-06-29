from collections import defaultdict
from collections.abc import Iterable, Mapping, KeysView, ItemsView
from functools import wraps
import os
import pickle

import numpy as np

from replays.circular_replay import CircularReplayBuffer


class UnhashingKeysView(KeysView):
    def __init__(self, mapping):
        self._mapping = mapping
        self._mapping_keys = super(type(self._mapping), self._mapping).keys

    def __len__(self):
        return len(self._mapping_keys())

    def __iter__(self):
        for key in self._mapping_keys():
            yield self._mapping.to_unhashable(key)

    def __contains__(self, key):
        return key in self._mapping_keys()

    def __repr__(self):
        return str(self._mapping_keys())


class UnhashingItemsView(ItemsView):
    def __init__(self, mapping):
        self._mapping = mapping
        self._mapping_items = super(type(self._mapping), self._mapping).items

    def __len__(self):
        return len(self._mapping)

    def __iter__(self):
        for key, value in self._mapping_items():
            yield (self._mapping.to_unhashable(key), value)

    def __contains__(self, item):
        key, value = item
        try:
            return self._mapping[key] == value
        except KeyError:
            return False

    def __repr__(self):
        return str(self._mapping_items())


class HashableKeyDict(defaultdict):
    def __init__(self, default=None, **kwargs):
        # if isinstance(default, type):
        #     super().__init__(lambda: default(), **kwargs)
        if isinstance(default, Iterable):
            super().__init__(**kwargs)
            self.update(other=default)
        else:
            super().__init__(default, **kwargs)

    #@staticmethod
    def to_unhashable(self, obj):
        def _to_hashable(obj):
            if isinstance(obj, bytes):
                return np.reshape(
                    np.frombuffer(obj, dtype=self._key_dtype),
                    newshape=self._key_shape
                )
            elif isinstance(obj, (np.number, str, int, float, bool)) or obj is None:
                return obj
            elif isinstance(obj, Iterable):
                return tuple(_to_hashable(sub_obj) for sub_obj in obj)
            else:
                raise KeyError(f"Key {obj} cannot be converted to hashable type")
        return _to_hashable(obj)

    #@staticmethod
    def to_hashable(self, obj):
        def _to_hashable(obj):
            if isinstance(obj, np.ndarray):
                self._key_dtype = obj.dtype
                self._key_shape = obj.shape
                return obj.tobytes()
            elif isinstance(obj, (np.number, str, int, float, bool, bytes)) or obj is None:
                return obj
            elif isinstance(obj, Iterable):
                return tuple(_to_hashable(sub_obj) for sub_obj in obj)
            else:
                raise KeyError(f"Key {obj} cannot be converted to hashable type")
        return _to_hashable(obj)

    #@staticmethod
    def hashable_key(method):
        @wraps(method)
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
        if other is not None:
            if hasattr(other, "items"):
                for k, v in other.items():
                    self.__setitem__(k, v)
            else:
                for k, v in other:
                    self.__setitem__(k, v)
        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def keys(self):
        if hasattr(self, "_key_shape"):
            return UnhashingKeysView(self)
        else:
            return super().keys()

    def items(self):
        if hasattr(self, "_key_shape"):
            return UnhashingItemsView(self)
        else:
            return super().items()


class SymmetricHashableKeyDict(HashableKeyDict):
    def to_hashable(self, *args, **kwargs):
        return tuple(sorted(super().to_hashable(*args, **kwargs)))


class SymmetricMatrix(Mapping):
    def __init__(self):
        self._views = HashableKeyDict(HashableKeyDict)

    #@staticmethod
    def process_key(method):
        @wraps(method)
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

    def update(self, **kwargs):
        return self._views.update(**kwargs)

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


# def distance(state1, state2):
#     """Compute distance for both symbolic and continuous states."""
#     state1, state2 = np.squeeze(state1), np.squeeze(state2)
#     if state1.ndim == state2.ndim == 3:
#         state1 = np.argwhere(state1[0] == 255).flatten()
#         state2 = np.argwhere(state2[0] == 255).flatten()
#     return np.linalg.norm(state1 - state2)


class CountsReplayBuffer(CircularReplayBuffer):
    def __init__(self, *args, action_n: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_n = action_n
        self.state_counts = HashableKeyDict(int)
        self.action_counts = HashableKeyDict(int)
        self.action_familiarities = HashableKeyDict(int)
        # self.distances = SymmetricMatrix()
        self._trajectory_familiarity = 1.0
        self._is_new_phase = False
        self._prev_next_state = None

    # def _update_distances(self, state):
    #     if self.action_counts[state] > 1:
    #         return
    #     for other_state in np.array(list(self.action_counts.keys())):
    #         if (state, other_state) not in self.distances:
    #             self.distances[state, other_state] = distance(state, other_state)

    def _familiarity(self, state, action):
        counts = self.action_counts.get((state, action), 1)
        return 1 - 1 / (1 + counts)

    def _increment_transition_metadata(self, transition):
        new_state = transition["observation"]
        new_action = transition["action"]
        new_next_state = transition["next_observation"]
        if not np.all(new_state == 0):
            self.action_counts[new_state, new_action] += 1
            self.state_counts[new_next_state] += 1
            # insert 0-count state-count entries for each action in the observed next state
            if self.state_counts[new_next_state] == 1:
                for action in range(self._action_n):
                    self.action_counts[new_next_state, action] += 0
            # self._update_distances(new_state)
            if self._is_new_phase:
                if not np.all(new_state == self._prev_next_state):
                    print("=== AND IS NOT CONTINUING ===")
                    breakpoint()
                    self._trajectory_familiarity = 1.0
                    self._prev_next_state = None
                else:
                    print("=== BUT IS CONTINUING ===")
                    breakpoint()
                self._is_new_phase = False
            self._trajectory_familiarity *= self._familiarity(new_state, new_action)
            print(f"{self._trajectory_familiarity=}")
            # breakpoint()
            self.action_familiarities[new_state, new_action] = self._trajectory_familiarity

    def _decrement_overwritten_transition_metadata(self):
        overwritten_state = self._storage["observation"][self._cursor]
        overwritten_action = self._storage["action"][self._cursor]
        overwritten_next_state = self._storage["next_observation"][self._cursor]
        if not np.all(overwritten_state == 0):
            self._decrement_state_metadata(overwritten_state, overwritten_action)
            self._decrement_state_metadata(overwritten_next_state)

    def _decrement_state_metadata(self, state, action=None):
        if action is not None:
            self.action_counts[state, action] -= 1
            if self.action_counts.get((state, action), 0) == 0:
                # keep 0-count state-action entries unless their state entries have been deleted
                if state not in self.state_counts:
                    del self.action_counts[state, action]
                    del self.action_familiarities[state, action]
        else:
            self.state_counts[state] -= 1
            if self.state_counts.get(state, 0) == 0:
                del self.state_counts[state]
                for action in range(self._action_n):
                    # delete any 0-count state-action entries for newly deleted (next) state entries
                    if self.action_counts.get((state, action), 0) == 0:
                        del self.action_counts[state, action]
                        del self.action_familiarities[state, action]
                # del self.distances[state]

    def _update_metadata(self, **transition):
        """ Decrement the overwritten observation's metadata, increment those of the new one. """
        self._decrement_overwritten_transition_metadata()
        self._increment_transition_metadata(transition)
        if transition["terminated"] or transition["done"]:
            print("=== NEW PHASE ===")
            self._is_new_phase = True
            self._prev_next_state = transition["next_observation"]

    def _add_transition(self, **transition):
        self._update_metadata(**transition)
        super()._add_transition(**transition)

    def save(self, dname):
        super().save(dname)
        with open(os.path.join(dname, "action_counts.pkl"), "wb") as f:
            pickle.dump(self.action_counts, f)
        with open(os.path.join(dname, "state_counts.pkl"), "wb") as f:
            pickle.dump(self.state_counts, f)

    def load(self, dname):
        super().load(dname)
        with open(os.path.join(dname, "action_counts.pkl"), "rb") as f:
            self.action_counts = pickle.load(f)
        with open(os.path.join(dname, "state_counts.pkl"), "rb") as f:
            self.state_counts = pickle.load(f)


def main():
    counts = HashableKeyDict(int)
    counts[np.array([[1, 1]])] = 1
    counts[(np.array([[1, 1]]), 0)] = 2
    print(counts.keys())
    for key in counts.keys():
        print(key)
    print(counts.items())
    for item in counts.items():
        print(item)
    print(counts.values())
    for value in counts.values():
        print(value)

if __name__ == "__main__":
    main()
