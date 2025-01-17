import numpy as np

from replays.counts_replay import CountsReplayBuffer


class HERReplayBuffer(CountsReplayBuffer):
    """Replay buffer with Hindsight Experience Replay (HER)."""
    def __init__(self, *args, her_batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.her_batch_size = her_batch_size

    def sample(self, batch_size):
        # Get the regular batch
        batch = super().sample(batch_size)

        # Sample HER trajectories from the buffer
        her_batch = self._sample_with_her(batch_size)

        # Mix HER samples with the original batch
        mixed_batch = self._mix_batches(batch, her_batch)

        return mixed_batch

    def _mix_batches(self, batch, her_batch):
        mixed_batch = {}

        for key in batch:
            # Combine regular transitions with HER ones
            mixed_batch[key] = np.concatenate([batch[key], her_batch[key]], axis=0)

        return mixed_batch

    def _sample_with_her(self, batch_size):
        batch = super().sample(batch_size)

        # Select random indices for HER in the batch
        her_indices = np.random.choice(
            batch_size, self.her_batch_size, replace=False
        )

        for i in her_indices:
            # Select a future state from the episode as hindsight goal
            future_goal_index = np.random.randint(i, self.size())
            batch["desired_goal"][i] = self._get_from_storage(
                "next_observation", future_goal_index
            ).squeeze()[0]
            # Update reward based on the new goal
            batch["reward"][i] = self._compute_her_reward(
                batch["observation"][i], batch["desired_goal"][i]
            )

        return batch

    def _compute_her_reward(self, achieved_goal, desired_goal):
        # Custom reward function based on goal achievement
        return 1.0 if np.array_equal(achieved_goal, desired_goal) else 0.0
