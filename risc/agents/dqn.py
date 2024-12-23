import copy
from dataclasses import asdict
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from hive.agents import DQNAgent as _DQNAgent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import InitializationFn, calculate_output_dim
from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.loggers import Logger
from hive.utils.schedule import Schedule
from hive.utils.utils import LossFn, OptimizerFn
from enum import IntEnum


class SuccessNet(torch.nn.Module):
    """Network that computes the probability of success of an action."""

    def __init__(
        self,
        base_network: torch.nn.Module,
        hidden_dim: int,
        out_dim: int,
        linear_fn: torch.nn.Module = None,
        correction=0,
    ):
        """
        Args:
            base_network (torch.nn.Module): Backbone network that computes the
                representations that are used to compute action values.
            hidden_dim (int): Dimension of the output of the :obj:`network`.
            out_dim (int): Output dimension of the DQN. Should be equal to the
                number of actions that you are computing values for.
            linear_fn (torch.nn.Module): Function that will create the
                :py:class:`torch.nn.Module` that will take the output of
                :obj:`network` and produce the final action values. If
                :obj:`None`, a :py:class:`torch.nn.Linear` layer will be used.
        """
        super().__init__()
        self.base_network = base_network
        self._linear_fn = linear_fn if linear_fn is not None else torch.nn.Linear
        self.output_layer = self._linear_fn(hidden_dim, out_dim)
        self._correction = correction

    def forward(self, x):
        x = self.base_network(x)
        x = x.flatten(start_dim=1)
        x = self.output_layer(x)
        return torch.sigmoid(x - self._correction)


class Actions(IntEnum):
    # Turn left, turn right, move forward
    right = 0
    down = 1
    left = 2
    up = 3


def oracle_policy(obs, rng: np.random.RandomState):
    """Oracle policy that computes the optimal action to take given the current
    observation for 4rooms environment.
    """
    obs = obs[0]
    [y], [x] = np.nonzero(obs[0])
    h_doors = np.where(obs[1, 9] == 0)[0]
    v_doors = np.where(obs[1, :, 9] == 0)[0]
    upper_half = y < 10
    left_half = x < 10
    targets = {
        (True, True): [(1, 1)],
        (True, False): [(9, v_doors[0])],
        (False, True): [(h_doors[0], 9)],
        (False, False): [(9, v_doors[1]), (h_doors[1], 9)],
    }
    target_locations = targets[(upper_half, left_half)]
    if len(target_locations) == 2:
        tx, ty = target_locations[0]
        dist0 = np.abs(tx - x) + np.abs(ty - y)
        tx, ty = target_locations[1]
        dist1 = np.abs(tx - x) + np.abs(ty - y)
        if dist0 < dist1:
            target_locations = [target_locations[0]]
        else:
            target_locations = [target_locations[1]]
    x_target, y_target = target_locations[0]
    if x == 9:
        return Actions.left
    elif y == 9:
        return Actions.up
    else:
        dx = x_target - x
        dy = y_target - y
        valid_directions = {
            Actions.right: True,
            Actions.left: True,
            Actions.up: True,
            Actions.down: True,
        }
        if x == 10 and y not in v_doors:
            valid_directions[Actions.left] = False
        elif y == 10 and x not in h_doors:
            valid_directions[Actions.up] = False

        dirx = Actions.right if dx > 0 else Actions.left
        diry = Actions.down if dy > 0 else Actions.up
        #  and rng.random() < 0.5:
        if dx == 0:
            return diry
        elif dy == 0:
            return dirx
        elif not valid_directions[dirx]:
            return diry
        elif not valid_directions[diry]:
            return dirx
        else:
            return dirx if rng.random() < 0.5 else diry


def oracle_q_value(obs):
    """Calculates the optimal Q value for the current observation."""
    obs = obs[0]
    [y], [x] = np.nonzero(obs[0])
    [goal_y], [goal_x] = np.nonzero(obs[2])
    if obs[0].shape[0] < 19:
        distance = np.abs(y - goal_y) + np.abs(x - goal_x)
        return 0.95**distance
    distances = {}
    h_doors = np.where(obs[1, 9] == 0)[0]  # x values of horizontal doors
    v_doors = np.where(obs[1, :, 9] == 0)[0]  # y values of vertical doors
    distances[(h_doors[1], 9)] = np.abs(h_doors[1] - goal_x) + np.abs(9 - goal_y)
    distances[(9, v_doors[1])] = np.abs(v_doors[1] - goal_y) + np.abs(9 - goal_x)
    distances[(9, v_doors[0])] = (
        np.abs(v_doors[0] - 9) + np.abs(9 - h_doors[1]) + distances[(h_doors[1], 9)]
    )
    distances[(h_doors[0], 9)] = (
        np.abs(h_doors[0] - 9) + np.abs(9 - v_doors[1]) + distances[(9, v_doors[1])]
    )

    upper_half = y < 10
    left_half = x < 10
    if not upper_half and not left_half:
        distance = (np.abs(goal_x - x) + np.abs(goal_y - y)) - 1
        distance = max(distance, 0)
    elif upper_half and not left_half:
        distance = np.abs(x - h_doors[1]) + np.abs(y - 9) + distances[(h_doors[1], 9)]
    elif not upper_half and left_half:
        distance = np.abs(x - 9) + np.abs(y - v_doors[1]) + distances[(9, v_doors[1])]
    else:
        distance1 = np.abs(x - 9) + np.abs(y - v_doors[0]) + distances[(9, v_doors[0])]
        distance2 = np.abs(x - h_doors[0]) + np.abs(y - 9) + distances[(h_doors[0], 9)]
        distance = min(distance1, distance2)
    return 0.95**distance


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
        self._oracle = oracle
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
        if self._oracle:
            greedy_actions = [oracle_policy(observation.cpu().numpy(), self._rng)]
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
        optimal_qval = oracle_q_value(observation.cpu().numpy())
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
        optimal_pd = optimal_difference / optimal_qval
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
