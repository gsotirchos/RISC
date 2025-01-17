import numpy as np

from replays.circular_replay import CircularReplayBuffer


class CountsReplayBuffer(CircularReplayBuffer):
    def __init__(self, *args, limited_counts=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.limited_counts = limited_counts
        self.counts = {}
        for key in self._specs:
            self.counts[key] = {}

    def _to_tuple(self, arr):
        if isinstance(arr, np.ndarray):
            return tuple(self._to_tuple(sub_arr) for sub_arr in arr)
        else:
            return arr

    def _update_counts(self, key, **transition):
        """ If a transition will be overwritten decrement its item's count and if it has no counts left remove its entry. Then increment the count for the new item. """
        if self.limited_counts:
            overwritten_item = self._to_tuple(self._storage[key][self._cursor])
            if not (np.array(overwritten_item) == 0).all():
                self.counts[key][overwritten_item] -= 1
                if self.counts[key][overwritten_item] == 0:
                    del self.counts[key][overwritten_item]
        new_item = self._to_tuple(transition[key])
        if not (np.array(new_item) == 0).all():
            self.counts[key][new_item] = self.counts[key].get(new_item, 0) + 1

    def _add_transition(self, **transition):
        self._update_counts("observation", **transition)
        super()._add_transition(**transition)
