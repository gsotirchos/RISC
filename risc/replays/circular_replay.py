from collections import defaultdict
import os
import pickle

import numpy as np

from hive.replays.prioritized_replay import (
    PrioritizedReplayBuffer as _PrioritizedReplayBuffer,
)
from hive.replays.circular_replay import (
    str_to_dtype,
    CircularReplayBuffer as _CircularReplayBuffer,
)
from hive.utils.utils import create_folder, seeder
from typing import Tuple, Dict


class CircularReplayBuffer(_CircularReplayBuffer):
    """An efficient version of a circular replay buffer that only stores each observation
    once.
    """

    def __init__(
        self,
        capacity: int = 10000,
        stack_size: int = 1,
        n_step: int = 1,
        gamma: float = 0.99,
        observation_shape=(),
        observation_dtype=np.uint8,
        action_shape=(),
        action_dtype=np.uint8,
        reward_shape=(),
        reward_dtype=np.float32,
        extra_storage_types=None,
        commit_at_done: bool = False,
        **kwargs,
    ):
        """Constructor for CircularReplayBuffer.

        Args:
            capacity (int): Total number of observations that can be stored in the
                buffer. Note, this is not the same as the number of transitions that
                can be stored in the buffer.
            stack_size (int): The number of frames to stack to create an observation.
            n_step (int): Horizon used to compute n-step return reward
            gamma (float): Discounting factor used to compute n-step return reward
            observation_shape: Shape of observations that will be stored in the buffer.
            observation_dtype: Type of observations that will be stored in the buffer.
                This can either be the type itself or string representation of the
                type. The type can be either a native python type or a numpy type. If
                a numpy type, a string of the form np.uint8 or numpy.uint8 is
                acceptable.
            action_shape: Shape of actions that will be stored in the buffer.
            action_dtype: Type of actions that will be stored in the buffer. Format is
                described in the description of observation_dtype.
            action_shape: Shape of actions that will be stored in the buffer.
            action_dtype: Type of actions that will be stored in the buffer. Format is
                described in the description of observation_dtype.
            reward_shape: Shape of rewards that will be stored in the buffer.
            reward_dtype: Type of rewards that will be stored in the buffer. Format is
                described in the description of observation_dtype.
            extra_storage_types (dict): A dictionary describing extra items to store
                in the buffer. The mapping should be from the name of the item to a
                (type, shape) tuple.
            num_players_sharing_buffer (int): Number of agents that share their
                buffers. It is used for self-play.
        """
        self._capacity = capacity
        self._specs = {
            "observation": (observation_dtype, observation_shape),
            "next_observation": (observation_dtype, observation_shape),
            "done": (np.uint8, ()),
            "terminated": (np.uint8, ()),
            "action": (action_dtype, action_shape),
            "reward": (reward_dtype, reward_shape),
        }
        if extra_storage_types is not None:
            self._specs.update(extra_storage_types)
        self._storage = self._create_storage(capacity, self._specs)
        self._stack_size = stack_size
        self._n_step = n_step
        self._gamma = gamma
        self._discount = np.asarray(
            [self._gamma**i for i in range(self._n_step)],
            dtype=self._specs["reward"][0],
        )
        self._cursor = 0
        self._num_added = 0
        self._rng = np.random.default_rng(seed=seeder.get_new_seed("replay"))
        self._commit_at_done = commit_at_done
        if commit_at_done:
            self._episode_storage = defaultdict(list)
        self._episode_start = True

    def size(self):
        """Returns the number of transitions stored in the buffer."""
        return max(
            min(self._num_added, self._capacity) - self._stack_size - self._n_step + 1,
            0,
        )

    def _create_storage(self, capacity, specs):
        """Creates the storage buffer for each type of item in the buffer.
        Args:
            capacity: The capacity of the buffer.
            specs: A dictionary mapping item name to a tuple (type, shape) describing
                the items to be stored in the buffer.
        """
        storage = {}
        for key in specs:
            dtype, shape = specs[key]
            dtype = str_to_dtype(dtype)
            specs[key] = dtype, shape
            shape = (capacity,) + tuple(shape)
            storage[key] = np.zeros(shape, dtype=dtype)
        return storage

    def _add_transition(self, **transition):
        """Internal method to add a transition to the buffer."""
        for key in transition:
            if key in self._storage:
                self._storage[key][self._cursor] = transition[key]
        self._num_added += 1
        self._cursor = (self._cursor + 1) % self._capacity

    def _pad_buffer(self, pad_length):
        """Adds padding to the buffer. Used when stack_size > 1, and padding needs to
        be added to the beginning of the episode.
        """
        for _ in range(pad_length):
            transition = {
                key: np.zeros_like(self._storage[key][0]) for key in self._storage
            }
            self._add_transition(**transition)

    def add_batch(self, batch):
        terminated = batch["terminated"]
        truncated = batch["truncated"]
        done = np.logical_or(terminated, truncated)
        for i in range(len(terminated)):
            transition = {k: batch[k][i] for k in batch if len(batch[k]) > i}
            transition["done"] = done[i]
            self.add(**transition)

    def add(
        self,
        observation,
        next_observation,
        action,
        reward,
        done,
        terminated,
        source=None,
        **kwargs,
    ):
        """Adds a transition to the buffer.
        The required components of a transition are given as positional arguments. The
        user can pass additional components to store in the buffer as kwargs as long as
        they were defined in the specification in the constructor.
        """

        transition = {
            "observation": observation,
            "next_observation": next_observation,
            "action": action,
            "reward": reward,
            "done": done,
            "terminated": terminated,
        }
        transition.update(kwargs)
        for key in self._specs:
            obj_type = (
                transition[key].dtype
                if hasattr(transition[key], "dtype")
                else type(transition[key])
            )
            if not np.can_cast(obj_type, self._specs[key][0], casting="same_kind"):
                raise ValueError(
                    f"Key {key} has wrong dtype. Expected {self._specs[key][0]}, "
                    f"received {obj_type}."
                )
        if not self._commit_at_done:
            if self._episode_start:
                self._pad_buffer(self._stack_size - 1)
                self._episode_start = False
            self._add_transition(**transition)
            if done:
                self._episode_start = True
        else:
            self._episode_storage[source].append(transition)
            if done:
                self._pad_buffer(self._stack_size - 1)
                for transition in self._episode_storage[source]:
                    self._add_transition(**transition)
                self._episode_storage[source] = []

    def _get_from_array(self, array, indices, num_to_access=1):
        """Retrieves consecutive elements in the array, wrapping around if necessary.
        If more than 1 element is being accessed, the elements are concatenated along
        the first dimension.
        Args:
            array: array to access from
            indices: starts of ranges to access from
            num_to_access: how many consecutive elements to access
        """
        full_indices = np.indices((indices.shape[0], num_to_access))[1]
        full_indices = (full_indices + np.expand_dims(indices, axis=1)) % (
            self.size() + self._stack_size + self._n_step - 1
        )
        elements = array[full_indices]
        elements = elements.reshape(indices.shape[0], -1, *elements.shape[3:])
        return elements

    def _get_from_storage(self, key, indices, num_to_access=1):
        """Gets values from storage.
        Args:
            key: The name of the component to retrieve.
            indices: This can be a single int or a 1D numpyp array. The indices are
                adjusted to fall within the current bounds of the buffer.
            num_to_access: how many consecutive elements to access
        """
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices])
        if num_to_access == 0:
            return np.array([])
        elif num_to_access == 1:
            return self._storage[key][
                indices % (self.size() + self._stack_size + self._n_step - 1)
            ]
        else:
            return self._get_from_array(
                self._storage[key], indices, num_to_access=num_to_access
            )

    def _sample_indices(self, batch_size):
        """Samples valid indices that can be used by the replay."""
        indices = np.array([], dtype=np.int32)
        while len(indices) < batch_size:
            start_index = (
                self._rng.integers(self.size(), size=batch_size - len(indices))
                + self._cursor
            )
            start_index = self._filter_transitions(start_index)
            indices = np.concatenate([indices, start_index])
        return (indices + self._stack_size - 1) % (
            self.size() + self._stack_size + self._n_step - 1
        )

    def _filter_transitions(self, indices):
        """Filters invalid indices."""
        if self._stack_size == 1:
            return indices
        done = self._get_from_storage("done", indices, self._stack_size - 1)
        done = done.astype(bool)
        if self._stack_size == 2:
            indices = indices[~done]
        else:
            indices = indices[~done.any(axis=1)]
        return indices

    def sample(self, batch_size):
        """Sample transitions from the buffer. For a given transition, if it's
        done is True, the next_observation value should not be taken to have any
        meaning.

        Args:
            batch_size (int): Number of transitions to sample.
        """
        if self._num_added < self._stack_size + self._n_step:
            raise ValueError("Not enough transitions added to the buffer to sample")
        indices = self._sample_indices(batch_size)
        batch = {}
        batch["indices"] = indices
        terminals = self._get_from_storage("done", indices, self._n_step)

        if self._n_step == 1:
            is_terminal = terminals
            trajectory_lengths = np.ones(terminals.shape[0])
        else:
            is_terminal = terminals.any(axis=1).astype(int)
            trajectory_lengths = (
                np.argmax(terminals.astype(bool), axis=1) + 1
            ) * is_terminal + self._n_step * (1 - is_terminal)
        trajectory_lengths = trajectory_lengths.astype(np.int64)

        for key in self._specs:
            if key == "observation":
                batch[key] = self._get_from_storage(
                    "observation",
                    indices - self._stack_size + 1,
                    num_to_access=self._stack_size,
                )
            elif key == "next_observation":
                batch[key] = self._get_from_storage(
                    "next_observation",
                    indices - self._stack_size + 1,
                    num_to_access=self._stack_size,
                )
            elif key == "done":
                batch["done"] = is_terminal
            elif key == "reward":
                rewards = self._get_from_storage("reward", indices, self._n_step)
                if self._n_step == 1:
                    rewards = np.expand_dims(rewards, 1)
                rewards = rewards * np.expand_dims(self._discount, axis=0)

                # Mask out rewards past trajectory length
                mask = np.expand_dims(trajectory_lengths, 1) > np.arange(self._n_step)
                rewards = np.sum(rewards * mask, axis=1)
                batch["reward"] = rewards
            else:
                batch[key] = self._get_from_storage(key, indices)

        batch["trajectory_lengths"] = trajectory_lengths
        return batch

    def save(self, dname):
        """Save the replay buffer.

        Args:
            dname (str): directory where to save buffer. Should already have been
                created.
        """
        storage_path = os.path.join(dname, "storage")
        create_folder(storage_path)
        for key in self._specs:
            np.save(
                os.path.join(storage_path, f"{key}"),
                self._storage[key],
                allow_pickle=False,
            )
        state = {
            "episode_start": self._episode_start,
            "cursor": self._cursor,
            "num_added": self._num_added,
            "rng": self._rng,
        }
        if self._commit_at_done:
            state["episode_storage"] = self._episode_storage
        with open(os.path.join(dname, "replay.pkl"), "wb") as f:
            pickle.dump(state, f)

    def load(self, dname):
        """Load the replay buffer.

        Args:
            dname (str): directory where to load buffer from.
        """
        storage_path = os.path.join(dname, "storage")
        for key in self._specs:
            self._storage[key] = np.load(
                os.path.join(storage_path, f"{key}.npy"), allow_pickle=False
            )
        with open(os.path.join(dname, "replay.pkl"), "rb") as f:
            state = pickle.load(f)
        self._episode_start = state["episode_start"]
        self._cursor = state["cursor"]
        self._num_added = state["num_added"]
        self._rng = state["rng"]
        if self._commit_at_done:
            self._episode_storage = state["episode_storage"]

    def _sample_indices_random(self, batch_size):
        """Samples valid indices that can be used by the replay."""
        indices = np.array([], dtype=np.int32)
        while len(indices) < batch_size:
            start_index = (
                self._rng.integers(self.size(), size=batch_size - len(indices))
                + self._cursor
            )
            start_index = self._filter_transitions(start_index)
            indices = np.concatenate([indices, start_index])
        indices = indices + self._stack_size - 1
        return indices % (self.size() + self._stack_size + self._n_step - 1)

    def sample_random(self, batch_size):
        if self._num_added < self._stack_size + self._n_step:
            raise ValueError("Not enough transitions added to the buffer to sample")
        indices = self._sample_indices_random(batch_size)
        batch = {}
        batch["indices"] = indices
        terminals = self._get_from_storage("done", indices, self._n_step)

        if self._n_step == 1:
            is_terminal = terminals
            trajectory_lengths = np.ones(batch_size)
        else:
            is_terminal = terminals.any(axis=1).astype(int)
            trajectory_lengths = (
                np.argmax(terminals.astype(bool), axis=1) + 1
            ) * is_terminal + self._n_step * (1 - is_terminal)
        trajectory_lengths = trajectory_lengths.astype(np.int64)

        for key in self._specs:
            if key == "observation":
                batch[key] = self._get_from_storage(
                    "observation",
                    indices - self._stack_size + 1,
                    num_to_access=self._stack_size,
                )
            elif key == "next_observation":
                batch[key] = self._get_from_storage(
                    "next_observation",
                    indices - self._stack_size + 1,
                    num_to_access=self._stack_size,
                )
            elif key == "done":
                batch["done"] = is_terminal
            elif key == "reward":
                rewards = self._get_from_storage("reward", indices, self._n_step)
                if self._n_step == 1:
                    rewards = np.expand_dims(rewards, 1)
                rewards = rewards * np.expand_dims(self._discount, axis=0)

                # Mask out rewards past trajectory length
                mask = np.expand_dims(trajectory_lengths, 1) > np.arange(self._n_step)
                rewards = np.sum(rewards * mask, axis=1)
                batch["reward"] = rewards
            else:
                batch[key] = self._get_from_storage(key, indices)

        batch["trajectory_lengths"] = trajectory_lengths
        return batch
