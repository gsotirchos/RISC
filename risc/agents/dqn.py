import copy
from dataclasses import asdict
from functools import partial
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
import torch
from hive.agents import DQNAgent as _DQNAgent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.qnet_heads import DQNNetwork
from hive.agents.qnets.utils import InitializationFn, calculate_output_dim
from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.loggers import Logger
from hive.utils.schedule import Schedule
from hive.utils.utils import LossFn, OptimizerFn
from enum import IntEnum


class SuccessNet(DQNNetwork):
    """Network that computes the probability of success of an action."""
    def __init__(self, *args, correction=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._correction = correction

    def forward(self, x):
        x = super().forward(x)
        return torch.sigmoid(x - self._correction)


class FourRoomsOracle():
    def __init__(self, discount_rate=1, step_reward=0, goal_reward=1, observation=None):
        self._discount_rate = discount_rate
        self._step_reward = step_reward
        self._goal_reward = goal_reward
        self._walls = None
        self._process_obs(observation)

    class Actions(IntEnum):
        right = 0
        down = 1
        left = 2
        up = 3

        @classmethod
        def move(cls, action, cell=(0, 0)):
            movements = {
                cls.right: (1, 0),
                cls.down: (0, 1),
                cls.left: (-1, 0),
                cls.up: (0, -1),
            }
            return cell[0] + movements[action][0], cell[1] + movements[action][1]

    def _process_obs(self, *args):
        if not args:
            return
        observation = args[0]
        if observation is None:
            return
        goal = None
        if len(args) >= 2:
            goal = args[1]
        agent_obs, walls_obs = observation.squeeze()[:2]
        goal_obs = goal.squeeze() if goal is not None else observation.squeeze()[2]
        self._agent_cell = tuple(np.flip(np.argwhere(agent_obs != 0)).flatten())
        self._goal_cell = tuple(np.flip(np.argwhere(goal_obs != 0)).flatten())
        if not np.array_equal(self._walls, walls_obs):
            self._walls = walls_obs
            self._valid_cells = [(x, y) for y, x in np.argwhere(walls_obs == 0)]
            self._distances = self._compute_distances()
        self._agent2goal_distances =  \
            np.array(list(self._distances[self._agent_cell, self._goal_cell].values()))

    @staticmethod
    def process_obs(method):
        def wrapper(self, *args, **kwargs):
            self._process_obs(*args)
            return method(self, *args, **kwargs)
        return wrapper

    def _bfs(self, goal):
        dist = defaultdict(lambda: -1)  # -1 = unreachable
        dist[goal] = 0
        queue = deque([goal])
        while queue:
            cell = queue.popleft()
            for action in self.Actions:
                next_cell = self.Actions.move(action, cell)
                if next_cell in self._valid_cells and dist[next_cell] == -1:
                    dist[next_cell] = dist[cell] + 1
                    queue.append(next_cell)
        return dist

    def _compute_distances(self):
        goal_dist = dict()
        for goal_cell in self._valid_cells:
            goal_dist[goal_cell] = self._bfs(goal_cell)
        distances = defaultdict(lambda: dict())
        for cell in self._valid_cells:
            for goal_cell in self._valid_cells:
                current_dist = goal_dist[goal_cell][cell]
                for action in self.Actions:
                    next_cell = self.Actions.move(action, cell)
                    if next_cell in self._valid_cells:
                        next_dist = goal_dist[goal_cell][next_cell]
                    else:
                        next_dist = current_dist

                    distances[cell, goal_cell][action] = next_dist
        return distances

    def _return(self, distances, discount_rate=None, step_reward=None, goal_reward=None):
        discount_rate = discount_rate or self._discount_rate
        step_reward = step_reward or self._step_reward
        goal_reward = goal_reward or self._goal_reward
        distances = np.asarray(distances)
        # Σ^(N-1)_(n=0) γ ^ n = (1 - γ ^ Ν) / (1 - γ)
        if discount_rate == 1:
            geom_series_sum = distances
        else:
            geom_series_sum = (1 - discount_rate ** distances) / (1 - discount_rate)
        return step_reward * geom_series_sum + goal_reward * (discount_rate ** distances)

    def _randargmax(self, x, **kwargs):
        return np.argmax(np.random.random(x.shape) * (x == x.max()), **kwargs)

    @process_obs
    def action(self, observation, **kwargs):
        return self._randargmax(self._return(self._agent2goal_distances, **kwargs))

    @process_obs
    def value(self, observation, step_reward=None, goal_reward=None, **kwargs):
        return np.max(self._return(1 + self._agent2goal_distances, **kwargs), keepdims=True)

    @process_obs
    def compute_success_prob(self, observation, goal):
        return 1 / np.min(1 + self._agent2goal_distances)


class DQNAgent(_DQNAgent):
    """Adapts observation format to RLHive's observation format. Also takes in a
    batch of observations during act instead of single observation.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        representation_net: FunctionApproximator,
        stack_size: int = 1,
        id=0,
        optimizer_fn: OptimizerFn = None,
        loss_fn: LossFn = None,
        init_fn: InitializationFn = None,
        replay_buffer: BaseReplayBuffer = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: float = None,
        reward_clip: float = None,
        update_period_schedule: Schedule = None,
        target_net_soft_update: bool = False,
        target_net_update_fraction: float = 0.05,
        target_net_update_schedule: Schedule = None,
        epsilon_schedule: Schedule = None,
        test_epsilon: float = 0.001,
        min_replay_history: int = 5000,
        batch_size: int = 32,
        device="cpu",
        logger: Logger = None,
        log_frequency: int = 100,
        compute_success_probability: bool = True,
        td_log_frequency: int = 20,
        only_add_low_confidence: bool = False,
        success_prob_threshold: float = 0.1,
        oracle: bool = False,
        **kwargs,
    ):
        #device = "cuda"
        self._compute_success_probability = compute_success_probability
        self._only_add_low_confidence = only_add_low_confidence
        self._success_prob_threshold = success_prob_threshold

        if loss_fn is None:
            loss_fn = torch.nn.MSELoss
        replay_buffer = partial(
            replay_buffer,
            action_shape=action_space.shape,
            action_dtype=action_space.dtype,
        )
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            representation_net=representation_net,
            stack_size=stack_size,
            optimizer_fn=optimizer_fn,
            loss_fn=loss_fn,
            init_fn=init_fn,
            id=id,
            replay_buffer=replay_buffer,
            discount_rate=discount_rate,
            n_step=n_step,
            grad_clip=grad_clip,
            reward_clip=reward_clip,
            update_period_schedule=update_period_schedule,
            target_net_soft_update=target_net_soft_update,
            target_net_update_fraction=target_net_update_fraction,
            target_net_update_schedule=target_net_update_schedule,
            epsilon_schedule=epsilon_schedule,
            test_epsilon=test_epsilon,
            min_replay_history=min_replay_history,
            batch_size=batch_size,
            device=device,
            logger=logger,
            log_frequency=log_frequency,
        )
        if self._compute_success_probability:
            self._success_net_loss_fn = torch.nn.BCELoss(reduction="none")
            if optimizer_fn is None:
                optimizer_fn = torch.optim.Adam
            self._success_net_optimizer = optimizer_fn(self._success_net.parameters())
        self._td_losses = []
        self._success_probs = []
        self._qval_error = []
        self._optimal_pds = []
        self._td_log_frequency = td_log_frequency
        self._use_oracle = oracle
        self._oracle = FourRoomsOracle(discount_rate, step_reward=-1, goal_reward=0)
        self._td_error = 0
        self._steps = 0

    def create_q_networks(self, representation_net):
        super().create_q_networks(representation_net)
        if self._compute_success_probability:
            network = representation_net(self._state_size)
            network_output_dim = np.prod(
                calculate_output_dim(network, self._state_size)
            )
            self._success_net = SuccessNet(
                network, network_output_dim, self._action_space.n
            ).to(self._device)
            self._success_net.apply(self._init_fn)
            self._target_success_net = copy.deepcopy(self._success_net).requires_grad_(
                False
            )

    def preprocess_update_info(self, update_info):
        if not isinstance(update_info, dict):
            update_info = asdict(update_info)
        if self._reward_clip is not None:
            update_info["reward"] = np.clip(
                update_info["reward"], -self._reward_clip, self._reward_clip
            )
        preprocessed_update_info = {
            "observation": update_info["observation"],
            "action": update_info["action"],
            "reward": update_info["reward"],
            "done": update_info["terminated"] or update_info["truncated"],
            "terminated": update_info["terminated"],
        }
        if "agent_id" in update_info:
            preprocessed_update_info["agent_id"] = int(update_info["agent_id"])

        return preprocessed_update_info

    @torch.no_grad()
    def act(self, observation, agent_traj_state=None):
        if self._training:
            if not self._learn_schedule.get_value():
                epsilon = 1.0
            else:
                epsilon = self._epsilon_schedule.update()
            if self._logger.update_step(self._timescale):
                self._logger.log_scalar("epsilon", epsilon, self._timescale)
        else:
            epsilon = self._test_epsilon
        observation = torch.tensor(observation, device=self._device).float()
        if len(observation.shape) == len(self._observation_space.shape):
            observation = observation.unsqueeze(0)
            batch_size, unsqueezed = 1, True
        else:
            batch_size, unsqueezed = len(observation), False

        qvals = self._qnet(observation)

        random_actions = self._rng.integers(self._action_space.n, size=batch_size)
        if self._use_oracle:
            greedy_actions = [self._oracle.action(observation.cpu().numpy())]
        else:
            greedy_actions = torch.argmax(qvals, dim=1).cpu().numpy()

        actions = np.where(
            self._rng.random(batch_size) < epsilon, random_actions, greedy_actions
        )
        if unsqueezed:
            actions = actions[0]

        if (
            self._training
            and self._logger.should_log(self._timescale)
            and agent_traj_state is None
        ):
            metrics = {"train_qval": torch.max(qvals).item()}
            self._logger.log_metrics(metrics, self._timescale)
        agent_traj_state = {} if agent_traj_state is None else agent_traj_state

        return actions, agent_traj_state

    def get_loss(self, transition):
        observation = torch.as_tensor(
            transition["observation"], device=self._device, dtype=torch.float32
        ).unsqueeze(0)
        desired_goal = torch.as_tensor(
            transition["desired_goal"], device=self._device, dtype=torch.float32
        ).unsqueeze(0)
        observation = torch.cat((observation, desired_goal), dim=1)
        next_observation = torch.as_tensor(
            transition["next_observation"], device=self._device, dtype=torch.float32
        ).unsqueeze(0)
        next_observation = torch.cat((next_observation, desired_goal), dim=1)
        optimal_qval = self._oracle.value(observation.cpu().numpy())
        qval = self._qnet(observation)
        next_qval = self._target_qnet(next_observation)
        error = (
            qval[0, transition["action"]]
            - (
                transition["reward"]
                + (1 - transition["terminated"]) * self._discount_rate * next_qval.max()
            )
        ) ** 2
        optimal_difference = np.abs((qval.amax() - optimal_qval).cpu().item())
        optimal_pd = optimal_difference / (optimal_qval + np.finfo(float).eps)
        return (
            error.item(),
            np.abs((qval.amax() - optimal_qval).cpu().item()),
            optimal_pd,
        )

    def preprocess_update_batch(self, batch):
        """Preprocess the batch sampled from the replay buffer.

        Args:
            batch: Batch sampled from the replay buffer for the current update.

        Returns:
            (tuple):
                - (tuple) Inputs used to calculate current state values.
                - (tuple) Inputs used to calculate next state values
                - Preprocessed batch.
        """
        for key in batch:
            batch[key] = torch.tensor(batch[key], device=self._device)
        return (batch["observation"],), (batch["next_observation"],), batch

    def update(self, update_info, agent_traj_state=None):
        """
        Updates the DQN agent.

        Args:
            update_info: dictionary containing all the necessary information
                from the environment to update the agent. Should contain a full
                transition, with keys for "observation", "action", "reward",
                "next_observation", and "done".
            agent_traj_state: Contains necessary state information for the agent
                to process current trajectory. This should be updated and returned.

        Returns:
            - action
            - agent trajectory state
        """
        if not self._training:
            return

        # Add the most recent transition to the replay buffer.
        transition = self.preprocess_update_info(update_info)
        metrics = {}
        with torch.no_grad():
            if self._compute_success_probability and self._only_add_low_confidence:
                success_prob = self.compute_success_prob(
                    transition["next_observation"], transition["desired_goal"]
                )
                if (
                    not self._learn_schedule.get_value()
                    or success_prob < self._success_prob_threshold
                ):
                    self._replay_buffer.add(**transition)
                self._success_probs.append(success_prob)
                if len(self._success_probs) > self._td_log_frequency:
                    metrics["trajectory_success_prob"] = np.mean(self._success_probs)
                    metrics["trajectory_success_prob_min"] = np.amin(
                        self._success_probs
                    )
                    metrics["trajectory_success_prob_max"] = np.amax(
                        self._success_probs
                    )
                    metrics["buffer_size"] = self._replay_buffer.size()
                    self._success_probs = []
            else:
                self._replay_buffer.add(**transition)
            td_loss, optimal_qval_error, optimal_pd = self.get_loss(transition)
            self._td_losses.append(td_loss)
            self._qval_error.append(optimal_qval_error)
            self._optimal_pds.append(optimal_pd)
            if len(self._td_losses) > self._td_log_frequency:
                metrics["trajectory_td_loss"] = np.mean(self._td_losses)
                metrics["optimal_qval_difference"] = np.mean(self._qval_error)
                metrics["optimal_pd"] = np.mean(self._optimal_pds)
                metrics["steps"] = self._steps
                self._logger.log_metrics(metrics, self._timescale)
                self._td_losses = []
                self._qval_error = []
                self._optimal_pds = []
        if (
            self._learn_schedule.update()
            and self._replay_buffer.size() > 0
            and self._update_period_schedule.update()
        ):
            batch = self._replay_buffer.sample(batch_size=self._batch_size)
            (
                current_state_inputs,
                next_state_inputs,
                batch,
            ) = self.preprocess_update_batch(batch)

            # Compute predicted Q values
            self.update_on_batch(batch, current_state_inputs, next_state_inputs)

        # Update target network
        if self._target_net_update_schedule.update():
            self._update_target()
        return agent_traj_state

    def update_on_batch(self, batch, current_state_inputs, next_state_inputs):
        metrics = {}
        metrics["train_loss"] = self.update_network(
            batch,
            current_state_inputs,
            next_state_inputs,
            batch["reward"],
            self._discount_rate,
            self._qnet,
            self._target_qnet,
            self._loss_fn,
            self._optimizer,
            True,
        )
        self._td_error += metrics["train_loss"].cpu().item()
        self._steps += 1
        metrics["cumulative_td_error"] = self._td_error
        metrics["steps"] = self._steps
        metrics["average_cumulative_td_error"] = self._td_error / self._steps
        if self._compute_success_probability:
            metrics["success_loss"] = self.update_network(
                batch,
                current_state_inputs,
                next_state_inputs,
                batch["terminated"].float(),  # * 2 - 1,
                self._discount_rate,
                self._success_net,
                self._target_success_net,
                self._success_net_loss_fn,
                self._success_net_optimizer,
                False,
            )

        if self._logger.should_log(self._timescale):
            self._logger.log_metrics(metrics, self._timescale)

    def update_network(
        self,
        batch,
        current_state_inputs,
        next_state_inputs,
        rewards,
        discount_rate,
        network,
        target_network,
        loss_fn,
        optimizer,
        update_priorities=False,
    ):
        pred_qvals = network(*current_state_inputs)
        actions = batch["action"].long()
        pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

        # Compute 1-step Q targets
        next_qvals = target_network(*next_state_inputs)
        next_qvals, _ = torch.max(next_qvals, dim=1)

        q_targets = rewards + discount_rate * next_qvals * (1 - batch["terminated"])

        loss = loss_fn(pred_qvals, q_targets)  # .mean()
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_value_(network.parameters(), self._grad_clip)
        optimizer.step()
        return loss

    def _update_target(self):
        super()._update_target()
        if not self._compute_success_probability:
            return
        if self._target_net_soft_update:
            target_params = self._target_success_net.state_dict()
            current_params = self._success_net.state_dict()
            for key in list(target_params.keys()):
                target_params[key] = (
                    1 - self._target_net_update_fraction
                ) * target_params[
                    key
                ] + self._target_net_update_fraction * current_params[
                    key
                ]
            self._target_success_net.load_state_dict(target_params)
        else:
            self._target_success_net.load_state_dict(self._success_net.state_dict())

    def compute_success_prob(self, observation):
        observation = torch.as_tensor(
            observation, device=self._device, dtype=torch.float32
        ).unsqueeze(0)
        return torch.max(self._success_net(observation)).detach().cpu().item()

    def get_stats(self, obs, goals):
        with torch.no_grad():
            obs = torch.tensor(obs, device=self._device)
            goals = torch.tensor(goals, device=self._device)
            goals = goals.unsqueeze(0)
            goals = goals.expand((obs.shape[0], -1, -1, -1))
            state = torch.cat([obs, goals], dim=1)
            metrics = {}
            if self._compute_success_probability:
                success_prob = self._success_net(state).amax(dim=1)  # > 0.25
                metrics["success_prob"] = success_prob.cpu().numpy()
            qvals = self._qnet(state).amax(dim=1)
            metrics["qvals"] = qvals.cpu().numpy()
            return metrics
