import numpy as np

from replays.counts_replay import CountsReplayBuffer


class HERReplayBuffer(CountsReplayBuffer):
    """Replay buffer with Hindsight Experience Replay (HER)."""
    def __init__(self, *args, her_ratio=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._her_ratio = her_ratio

    def sample(self, batch_size):
        # Get a batch from the parent class's sampler
        batch = super().sample(batch_size)

        indices = batch["indices"]
        num_her_samples = int(batch_size * self._her_ratio)
        her_indices_in_batch = self._rng.choice(
            batch_size, num_her_samples, replace=False
        )

        for i in her_indices_in_batch:
            # The hindsight goal will be the final observation of the current trajectory
            hindsight_goal_index = indices[i]
            tries_left = self.size()
            while tries_left > 0:
                hindsight_goal_index = (hindsight_goal_index + 1) % (
                    self.size() + self._stack_size + self._n_step - 1
                )
                if self._get_from_storage("done", hindsight_goal_index):
                    break
                tries_left -= 1
            hindsight_goal = self._get_from_storage("next_observation", hindsight_goal_index)[0]
            hindsight_goal = hindsight_goal[
                tuple(slice(size) for size in self._specs["desired_goal"][1])
            ]

            # Update the transition with the new goal and reward
            batch["desired_goal"][i] = hindsight_goal
            batch["reward"][i] = self._compute_her_reward(
                batch["next_observation"][i], hindsight_goal
            )

        return batch

    def _compute_her_reward(self, achieved_goal, desired_goal):
        # Custom reward function based on goal achievement
        return 0.0 if np.array_equal(achieved_goal, desired_goal) else -1.0
