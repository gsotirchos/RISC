import copy
import os

import gymnasium as gym
import numpy as np
import torch

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import InitializationFn, calculate_output_dim
from hive.replays import BaseReplayBuffer, CircularReplayBuffer
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import PeriodicSchedule, SwitchSchedule, DoublePeriodicSchedule
from hive.utils.utils import LossFn, OptimizerFn, create_folder

from typing import Tuple, Union
import gymnasium as gym
import numpy as np
import torch

from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import calculate_output_dim

from dataclasses import dataclass, asdict


@dataclass
class SACAgentState: ...


MIN_LOG_STD = -20
MAX_LOG_STD = 10


class GaussianPolicyHead(torch.nn.Module):
    """A module that implements a continuous actor head. It takes in the output
    from the :obj:`actor_net`, and adds creates a normal distribution to compute
    the action distribution. It also adds a tanh layer to bound the output of
    the network to the action space. The forward method returns a sampled action
    from the distribution and the log probability of the action. It also returns
    a dummy value to keep a consistent interface with the discrete actor head.
    """

    def __init__(self, feature_dim: Tuple[int], action_space: gym.spaces.Box) -> None:
        """
        Args:
            feature dim: Expected output shape of the actor network.
            action_shape: Expected shape of actions.
        """
        super().__init__()
        self._action_shape = action_space.shape
        self._policy_mean = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, np.prod(self._action_shape))
        )
        self._policy_logstd = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, np.prod(self._action_shape))
        )
        self._distribution = torch.distributions.normal.Normal

    def forward(self, x):
        mean = self._policy_mean(x)
        log_std = self._policy_logstd(x)

        log_std = torch.clamp(log_std, MIN_LOG_STD, MAX_LOG_STD)
        std = torch.exp(log_std)
        distribution = self._distribution(mean, std)
        unsquashed_action = distribution.rsample()
        log_prob = distribution.log_prob(unsquashed_action).sum(dim=-1, keepdim=True)

        # Correct for Tanh squashing
        correction = 2 * (
            (
                np.log(2)
                - unsquashed_action
                - torch.nn.functional.softplus(-2 * unsquashed_action)
            )
        ).sum(axis=1, keepdim=True)
        log_prob = log_prob - correction
        action = torch.tanh(unsquashed_action)
        return action, log_prob, torch.tanh(mean)


class SACActorNetwork(torch.nn.Module):
    """A module that implements the SAC actor computation. It puts together the
    :obj:`representation_network` and :obj:`actor_net`, and adds a final
    :py:class:`~torch.nn.Linear` layer to compute the action."""

    def __init__(
        self,
        representation_network: torch.nn.Module,
        actor_net: FunctionApproximator,
        representation_network_output_shape: Union[int, Tuple[int]],
        action_space: gym.spaces.Box,
    ) -> None:
        """
        Args:
            representation_network (torch.nn.Module): Network that encodes the
                observations.
            actor_net (FunctionApproximator): Function that takes in the shape of the
                encoded observations and creates a network. This network takes the
                encoded observations from representation_net and outputs the
                representations used to compute the actions (ie everything except the
                last layer).
            network_output_shape: Expected output shape of the representation network.
            action_shape: Requiured shape of the output action.
        """
        super().__init__()

        self._action_shape = action_space.shape
        if actor_net is None:
            actor_network = torch.nn.Identity()
        else:
            actor_network = actor_net(representation_network_output_shape)
        feature_dim = np.prod(
            calculate_output_dim(actor_network, representation_network_output_shape)
        )
        actor_modules = [
            representation_network,
            actor_network,
            torch.nn.Flatten(),
        ]
        policy_head_fn = GaussianPolicyHead
        actor_modules.append(policy_head_fn(feature_dim, action_space))
        self.actor = torch.nn.Sequential(*actor_modules)

    def forward(self, x):
        return self.actor(x)


class SACSuccessCriticNetwork(torch.nn.Module):
    def __init__(
        self,
        representation_network: torch.nn.Module,
        critic_net: FunctionApproximator,
        network_output_shape: Union[int, Tuple[int]],
        action_space: gym.spaces.Box,
        n_critics: int = 2,
    ) -> None:
        """
        Args:
            representation_network (torch.nn.Module): Network that encodes the
                observations.
            critic_net (FunctionApproximator): Function that takes in the shape of the
                encoded observations and creates a network. This network takes two
                inputs: the encoded observations from representation_net and actions.
                It outputs the representations used to compute the values of the
                actions (ie everything except the last layer).
            network_output_shape: Expected output shape of the representation network.
            action_space: Expected shape of actions.
            n_critics: How many copies of the critic to create. They will all use the
                shared representation from the representation_network.
        """
        super().__init__()
        self.network = representation_network
        if critic_net is None:
            critic_net = lambda x: torch.nn.Identity()
        self._n_critics = n_critics
        input_shape = (np.prod(network_output_shape) + np.prod(action_space.shape),)
        critics = [critic_net(input_shape) for _ in range(n_critics)]
        feature_dim = np.prod(calculate_output_dim(critics[0], input_shape=input_shape))
        self._critics = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    critic,
                    torch.nn.Flatten(),
                    torch.nn.Linear(feature_dim, 1),
                )
                for critic in critics
            ]
        )

    def forward(self, obs, actions):
        obs = self.network(obs)
        obs = torch.flatten(obs, start_dim=1)
        actions = torch.flatten(actions, start_dim=1)
        x = torch.cat([obs, actions], dim=1)
        return [-0.5 * (torch.cos(critic(x)) - 1) for critic in self._critics]


class SACContinuousCriticNetwork(torch.nn.Module):
    def __init__(
        self,
        representation_network: torch.nn.Module,
        critic_net: FunctionApproximator,
        network_output_shape: Union[int, Tuple[int]],
        action_space: gym.spaces.Box,
        n_critics: int = 2,
    ) -> None:
        """
        Args:
            representation_network (torch.nn.Module): Network that encodes the
                observations.
            critic_net (FunctionApproximator): Function that takes in the shape of the
                encoded observations and creates a network. This network takes two
                inputs: the encoded observations from representation_net and actions.
                It outputs the representations used to compute the values of the
                actions (ie everything except the last layer).
            network_output_shape: Expected output shape of the representation network.
            action_space: Expected shape of actions.
            n_critics: How many copies of the critic to create. They will all use the
                shared representation from the representation_network.
        """
        super().__init__()
        self.network = representation_network
        if critic_net is None:
            critic_net = lambda x: torch.nn.Identity()
        self._n_critics = n_critics
        input_shape = (np.prod(network_output_shape) + np.prod(action_space.shape),)
        critics = [critic_net(input_shape) for _ in range(n_critics)]
        feature_dim = np.prod(calculate_output_dim(critics[0], input_shape=input_shape))
        self._critics = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    critic,
                    torch.nn.Flatten(),
                    torch.nn.Linear(feature_dim, 1),
                )
                for critic in critics
            ]
        )

    def forward(self, obs, actions):
        obs = self.network(obs)
        obs = torch.flatten(obs, start_dim=1)
        actions = torch.flatten(actions, start_dim=1)
        x = torch.cat([obs, actions], dim=1)
        return [critic(x) for critic in self._critics]


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        # TODO: Changed initialization to xavier_uniform_
        torch.nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class ReplayDataset(torch.utils.data.IterableDataset):
    def __init__(self, replay_buffer: CircularReplayBuffer):
        self._replay_buffer = replay_buffer
        self._batch_size = 256

    def __iter__(self):
        while True:
            yield self._replay_buffer.sample(self._batch_size)


class SACAgent(Agent):
    """An agent implementing the SAC algorithm."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        actor_trunk_net: FunctionApproximator = None,
        critic_trunk_net: FunctionApproximator = None,
        actor_net: FunctionApproximator = None,
        critic_net: FunctionApproximator = None,
        init_fn: InitializationFn = None,
        actor_optimizer_fn: OptimizerFn = None,
        critic_optimizer_fn: OptimizerFn = None,
        critic_loss_fn: LossFn = None,
        auto_alpha: bool = True,
        alpha: float = 0.2,
        alpha_optimizer_fn: OptimizerFn = None,
        target_entropy_scale: float = 0.5,
        reward_scale_factor: float = 10.0,
        n_critics: int = 2,
        stack_size: int = 1,
        replay_buffer: BaseReplayBuffer = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: float = None,
        reward_clip: float = None,
        soft_update_fraction: float = 0.005,
        batch_size: int = 64,
        logger: Logger = None,
        log_frequency: int = 100,
        update_frequency: int = 1,
        policy_update_frequency: int = 2,
        min_replay_history: int = 1000,
        target_net_update_frequency: int = 1,
        critic_loss_weight: float = 0.5,
        device="cpu",
        id=0,
        forward_demos=None,
        num_actions_value_computation: int = 5,
        pretrain_steps: int = 0,
        use_value_comp_critic: bool = False,
        greedy_actor: bool = True,
        compute_separate_targets: bool = False,
        trunc_as_terminal: bool = False,
        compute_success_probability: bool = False,
        **kwargs,
    ):
        """
        Args:
            observation_space (gym.spaces.Box): Observation space for the agent.
            action_space (gym.spaces.Box): Action space for the agent.
            representation_net (FunctionApproximator): The network that encodes the
                observations that are then fed into the actor_net and critic_net. If
                None, defaults to :py:class:`~torch.nn.Identity`.
            actor_net (FunctionApproximator): The network that takes the encoded
                observations from representation_net and outputs the representations
                used to compute the actions (ie everything except the last layer).
            critic_net (FunctionApproximator): The network that takes two inputs: the
                encoded observations from representation_net and actions. It outputs
                the representations used to compute the values of the actions (ie
                everything except the last layer).
            init_fn (InitializationFn): Initializes the weights of agent networks using
                create_init_weights_fn.
            actor_optimizer_fn (OptimizerFn): A function that takes in the list of
                parameters of the actor returns the optimizer for the actor. If None,
                defaults to :py:class:`~torch.optim.Adam`.
            critic_optimizer_fn (OptimizerFn): A function that takes in the list of
                parameters of the critic returns the optimizer for the critic. If None,
                defaults to :py:class:`~torch.optim.Adam`.
            critic_loss_fn (LossFn): The loss function used to optimize the critic. If
                None, defaults to :py:class:`~torch.nn.MSELoss`.
            n_critics (int): The number of critics used by the agent to estimate
                Q-values. The minimum Q-value is used as the value for the next state
                when calculating target Q-values for the critic. The output of the
                first critic is used when computing the loss for the actor. The
                default value for SAC is 2.
            stack_size (int): Number of observations stacked to create the state fed
                to the agent.
            replay_buffer (BaseReplayBuffer): The replay buffer that the agent will
                push observations to and sample from during learning. If None,
                defaults to
                :py:class:`~hive.replays.circular_replay.CircularReplayBuffer`.
            discount_rate (float): A number between 0 and 1 specifying how much
                future rewards are discounted by the agent.
            n_step (int): The horizon used in n-step returns to compute TD(n) targets.
            grad_clip (float): Gradients will be clipped to between
                [-grad_clip, grad_clip].
            reward_clip (float): Rewards will be clipped to between
                [-reward_clip, reward_clip].
            soft_update_fraction (float): The weight given to the target
                net parameters in a soft (polyak) update. Also known as tau.
            batch_size (int): The size of the batch sampled from the replay buffer
                during learning.
            logger (Logger): Logger used to log agent's metrics.
            log_frequency (int): How often to log the agent's metrics.
            update_frequency (int): How frequently to update the agent. A value of 1
                means the agent will be updated every time update is called.
            policy_update_frequency (int): Relative update frequency of the actor
                compared to the critic. The actor will be updated every
                policy_update_frequency times the critic is updated.
            min_replay_history (int): How many observations to fill the replay buffer
                with before starting to learn.
            device: Device on which all computations should be run.
            id: Agent identifier.
        """
        super().__init__(observation_space, action_space, id)
        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        self._state_size = (
            stack_size * self._observation_space.shape[0],
            *self._observation_space.shape[1:],
        )
        self._action_min = self._action_space.low
        self._action_max = self._action_space.high
        self._action_scaling = 0.5 * (self._action_max - self._action_min)
        self._action_min_tensor = torch.as_tensor(self._action_min, device=self._device)
        self._action_max_tensor = torch.as_tensor(self._action_max, device=self._device)
        self._init_fn = weight_init

        self._n_critics = n_critics
        self._auto_alpha = auto_alpha
        self._target_entropy_scale = target_entropy_scale
        self._use_value_comp_critic = use_value_comp_critic
        self._compute_success_probability = compute_success_probability

        self.create_networks(actor_trunk_net, critic_trunk_net, actor_net, critic_net)
        if critic_optimizer_fn is None:
            critic_optimizer_fn = torch.optim.Adam
        if actor_optimizer_fn is None:
            actor_optimizer_fn = torch.optim.Adam
        if auto_alpha and alpha_optimizer_fn is None:
            alpha_optimizer_fn = torch.optim.Adam
        self._critic_optimizer = critic_optimizer_fn(self._critic.parameters())
        self._actor_optimizer = actor_optimizer_fn(self._actor.parameters())
        if self._use_value_comp_critic:
            self._value_comp_critic_optimizer = critic_optimizer_fn(
                self._value_comp_critic.parameters()
            )
        if self._compute_success_probability:
            self._success_critic_optimizer = critic_optimizer_fn(
                self._success_critic.parameters()
            )

        if auto_alpha:
            self._alpha_optimizer = alpha_optimizer_fn([self._log_alpha])
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha
        if replay_buffer is None:
            replay_buffer = CircularReplayBuffer
        self._replay_buffer = replay_buffer(
            observation_shape=self._observation_space.shape,
            observation_dtype=self._observation_space.dtype,
            action_shape=self._action_space.shape,
            action_dtype=self._action_space.dtype,
            gamma=discount_rate,
        )
        self._discount_rate = discount_rate**n_step
        self._grad_clip = grad_clip
        self._reward_clip = reward_clip
        self._soft_update_fraction = soft_update_fraction
        if critic_loss_fn is None:
            critic_loss_fn = torch.nn.MSELoss
        self._critic_loss_fn = critic_loss_fn(reduction="mean")
        self._batch_size = batch_size
        self._logger = logger
        if self._logger is None:
            self._logger = NullLogger([])
        self._timescale = self.id
        self._logger.register_timescale(
            self._timescale, PeriodicSchedule(False, True, log_frequency)
        )
        self._update_schedule = PeriodicSchedule(False, True, update_frequency)
        self._target_net_update_schedule = PeriodicSchedule(
            False, True, target_net_update_frequency
        )
        self._policy_update_schedule = DoublePeriodicSchedule(
            True, False, policy_update_frequency, policy_update_frequency
        )
        self._learn_schedule = SwitchSchedule(False, True, min_replay_history)
        self._training = False
        self._reward_scale_factor = reward_scale_factor
        self._critic_loss_weight = critic_loss_weight

        self._pretrain_steps = pretrain_steps
        if forward_demos is not None:
            self.load_demos(forward_demos)
        self._num_actions_value_computation = num_actions_value_computation
        self._greedy_actor = greedy_actor
        self._compute_separate_targets = compute_separate_targets
        self._trunc_as_terminal = trunc_as_terminal
        self._replay_dataset = ReplayDataset(self._replay_buffer)
        self._replay_loader = None

    def create_networks(self, actor_trunk_net, critic_trunk_net, actor_net, critic_net):
        """Creates the actor and critic networks.

        Args:
            representation_net: A network that outputs the shared representations that
                will be used by the actor and critic networks to process observations.
            actor_net: The network that will be used to compute actions.
            critic_net: The network that will be used to compute values of state action
                pairs.
        """
        if actor_trunk_net is None:
            actor_trunk = torch.nn.Identity()
        else:
            actor_trunk = actor_trunk_net(self._state_size)
        if critic_trunk_net is None:
            critic_trunk = torch.nn.Identity()
        else:
            critic_trunk = critic_trunk_net(self._state_size)
        critic_trunk_output_shape = calculate_output_dim(critic_trunk, self._state_size)
        actor_trunk_output_shape = calculate_output_dim(actor_trunk, self._state_size)
        self._actor = SACActorNetwork(
            actor_trunk,
            actor_net,
            actor_trunk_output_shape,
            self._action_space,
        ).to(self._device)
        self._critic = SACContinuousCriticNetwork(
            critic_trunk,
            critic_net,
            critic_trunk_output_shape,
            self._action_space,
            self._n_critics,
        ).to(self._device)
        self._actor.apply(self._init_fn)
        self._critic.apply(self._init_fn)
        self._target_critic = copy.deepcopy(self._critic).requires_grad_(False)
        if self._use_value_comp_critic:
            self._value_comp_critic = SACContinuousCriticNetwork(
                critic_trunk,
                critic_net,
                critic_trunk_output_shape,
                self._action_space,
                self._n_critics,
            ).to(self._device)
            self._value_comp_critic.apply(self._init_fn)
            self._target_value_comp_critic = copy.deepcopy(
                self._value_comp_critic
            ).requires_grad_(False)
        if self._compute_success_probability:
            self._success_critic = SACSuccessCriticNetwork(
                critic_trunk,
                critic_net,
                critic_trunk_output_shape,
                self._action_space,
                self._n_critics,
            ).to(self._device)
            self._success_critic.apply(self._init_fn)
            self._target_success_critic = copy.deepcopy(
                self._success_critic
            ).requires_grad_(False)
        # Automatic entropy tuning
        if self._auto_alpha:
            self._target_entropy = (
                -np.prod(self._action_space.shape) * self._target_entropy_scale
            )
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._actor.train()
        self._critic.train()
        self._target_critic.train()
        if self._use_value_comp_critic:
            self._value_comp_critic.train()
            self._target_value_comp_critic.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._actor.eval()
        self._critic.eval()
        self._target_critic.eval()
        if self._use_value_comp_critic:
            self._value_comp_critic.eval()
            self._target_value_comp_critic.eval()

    def scale_action(self, actions):
        """Scales actions to [-1, 1]."""
        return ((actions - self._action_min) / self._action_scaling) - 1.0

    def unscale_actions(self, actions):
        """Unscales actions from [-1, 1] to expected scale."""
        return ((actions + 1.0) * self._action_scaling) + self._action_min

    def preprocess_update_info(self, update_info):
        """Preprocesses the :obj:`update_info` before it goes into the replay buffer.
        Scales the action to [-1, 1].

        Args:
            update_info: Contains the information from the current timestep that the
                agent should use to update itself.
        """
        if not isinstance(update_info, dict):
            update_info = asdict(update_info)
        if self._reward_clip is not None:
            update_info["reward"] = np.clip(
                update_info["reward"], -self._reward_clip, self._reward_clip
            )
        preprocessed_update_info = {
            "observation": update_info["observation"],
            "next_observation": update_info["next_observation"],
            "action": self.scale_action(update_info["action"]),
            "reward": update_info["reward"],
            "terminated": update_info["terminated"],
            "done": update_info["truncated"] or update_info["terminated"],
        }

        return preprocessed_update_info

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
            batch[key] = batch[key].to(self._device)
        return (batch["observation"],), (batch["next_observation"],), batch

    @torch.no_grad()
    def act(self, observation, agent_traj_state=None):
        """Returns the action for the agent. If in training mode, adds noise with
        standard deviation :py:obj:`self._action_noise`.

        Args:
            observation: The current observation.
            agent_traj_state: Contains necessary state information for the agent
                to process current trajectory. This should be updated and returned.

        Returns:
            - action
            - agent trajectory state
        """

        # Calculate action and add noise if training.
        if self._training and not self._learn_schedule.get_value():
            return (self._action_space.sample(), SACAgentState())

        observation = torch.tensor(
            np.expand_dims(observation, axis=0), device=self._device
        ).float()
        action, _, mean = self._actor(observation)
        if not self._training and self._greedy_actor:
            action = mean

        action = action.cpu().detach().numpy()
        action = self.unscale_actions(action)
        return action[0], agent_traj_state

    def load_demos(self, demo_batch):
        self._replay_buffer.add_batch(demo_batch)
        if self._pretrain_steps > 0:
            self.pretrain(self._pretrain_steps)

    def update(self, update_info, agent_traj_state=None):
        """
        Updates the SAC agent.

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
            return agent_traj_state
        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(**self.preprocess_update_info(update_info))
        # Update the agent based on a sample batch from the replay buffer.
        if (
            self._learn_schedule.update()
            and self._replay_buffer.size() > 0
            and self._update_schedule.update()
        ):
            if self._replay_loader is None:
                if self._device.type == "cuda":
                    self._replay_loader = torch.utils.data.DataLoader(
                        self._replay_dataset,
                        batch_size=None,
                        pin_memory=True,
                        num_workers=2,
                        prefetch_factor=2,
                    )
                else:
                    self._replay_loader = torch.utils.data.DataLoader(
                        self._replay_dataset,
                        batch_size=None,
                    )
                self._replay_iter = iter(self._replay_loader)

            batch = next(self._replay_iter)
            self.update_on_batch(batch)
        return agent_traj_state

    def pretrain(self, num_steps):
        """Pretrains the agent for a number of steps.

        Args:
            num_steps (int): Number of steps to pretrain for.
        """
        if not self._training:
            return
        for _ in range(num_steps):
            batch = self._replay_buffer.sample(batch_size=self._batch_size)
            self.update_on_batch(batch)

    def update_on_batch(self, batch):
        (
            current_state_inputs,
            next_state_inputs,
            batch,
        ) = self.preprocess_update_batch(batch)

        metrics = {}
        metrics.update(
            self._update_critics(batch, current_state_inputs, next_state_inputs)
        )
        if self._use_value_comp_critic:
            metrics.update(
                self._update_value_comp_critics(
                    batch, current_state_inputs, next_state_inputs
                )
            )
        if self._compute_success_probability:
            metrics.update(
                self._update_success_probability(
                    batch, current_state_inputs, next_state_inputs
                )
            )
        # Update policy with policy delay
        while self._policy_update_schedule.update():
            metrics.update(self._update_actor(current_state_inputs))
        if self._target_net_update_schedule.update():
            self._update_target()
        if self._logger.update_step(self._timescale):
            self._logger.log_metrics(metrics, self._timescale)

    def _update_actor(self, current_state_inputs):
        actions, log_probs, _ = self._actor(*current_state_inputs)
        action_values = self._critic(*current_state_inputs, actions)
        min_action_values = torch.min(action_values[0], action_values[1]).view(-1)
        actor_loss = torch.mean(self._alpha * log_probs - min_action_values)
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._grad_clip)
        self._actor_optimizer.step()
        metrics = {"actor_loss": actor_loss}
        if self._auto_alpha:
            metrics.update(self._update_alpha(current_state_inputs))

        return metrics

    def _update_alpha(self, current_state_inputs):
        with torch.no_grad():
            _, log_probs, _ = self._actor(*current_state_inputs)
        alpha_loss = (
            -self._log_alpha * (log_probs + self._target_entropy).detach()
        ).mean()
        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()
        self._alpha = self._log_alpha.exp().item()
        return {"alpha": self._alpha, "alpha_loss": alpha_loss}

    def _update_critics(self, batch, current_state_inputs, next_state_inputs):
        target_q_values = self._calculate_target_q_values(
            batch, next_state_inputs, self._target_critic
        )

        # Critic losses
        pred_qvals = self._critic(*current_state_inputs, batch["action"])
        critic_losses = [
            self._critic_loss_fn(qvals, target_qvals)
            for (qvals, target_qvals) in zip(pred_qvals, target_q_values)
        ]
        critic_loss = sum(critic_losses) * self._critic_loss_weight
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self._grad_clip)
        self._critic_optimizer.step()
        metrics = {f"critic_{idx}_value": x.mean() for idx, x in enumerate(pred_qvals)}
        metrics.update({f"critic_{idx}_loss": x for idx, x in enumerate(critic_losses)})
        metrics["critic_targets"] = target_q_values.mean()
        metrics["critic_loss"] = critic_loss
        return metrics

    def _update_success_probability(
        self, batch, current_state_inputs, next_state_inputs
    ):
        with torch.no_grad():
            next_actions, _, _ = self._actor(*next_state_inputs)
            next_success_probs = torch.stack(
                self._target_success_critic(*next_state_inputs, next_actions)
            )
            next_success_probs = torch.amin(
                next_success_probs, dim=0, keepdim=True
            ).broadcast_to(
                next_success_probs.shape
            )  # n_critics x batch_size x n_heads
            terminals = batch["terminated"][None, :, None]
            target_success_probs = (
                terminals + (1 - terminals) * self._discount_rate * next_success_probs
            )
        # Critic losses
        pred_success_probs = self._success_critic(
            *current_state_inputs, batch["action"]
        )
        critic_losses = [
            self._critic_loss_fn(success_probs, target_success)
            for (success_probs, target_success) in zip(
                pred_success_probs, target_success_probs
            )
        ]
        critic_loss = sum(critic_losses) * self._critic_loss_weight
        self._success_critic_optimizer.zero_grad()
        critic_loss.backward()
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self._success_critic_optimizer.parameters(), self._grad_clip
            )
        self._success_critic_optimizer.step()
        metrics = {
            f"success_critic_{idx}_value": x.mean()
            for idx, x in enumerate(pred_success_probs)
        }
        metrics.update(
            {f"success_critic_{idx}_loss": x for idx, x in enumerate(critic_losses)}
        )
        metrics["success_critic_loss"] = critic_loss
        metrics["target_success_prob"] = target_success_probs.mean()
        return metrics

    def _update_value_comp_critics(
        self, batch, current_state_inputs, next_state_inputs
    ):
        target_q_values = self._calculate_target_q_values(
            batch,
            next_state_inputs,
            self._target_value_comp_critic,
            entropy=False,
            compute_separate_targets=True,
        )

        # Critic losses
        pred_qvals = self._value_comp_critic(*current_state_inputs, batch["action"])
        critic_losses = [
            self._critic_loss_fn(qvals, target_qvals)
            for (qvals, target_qvals) in zip(pred_qvals, target_q_values)
        ]
        critic_loss = sum(critic_losses) * self._critic_loss_weight
        self._value_comp_critic_optimizer.zero_grad()
        critic_loss.backward()
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self._value_comp_critic_optimizer.parameters(), self._grad_clip
            )
        self._value_comp_critic_optimizer.step()
        metrics = {
            f"vc_critic_{idx}_value": x.mean() for idx, x in enumerate(pred_qvals)
        }
        metrics.update(
            {f"vc_critic_{idx}_loss": x for idx, x in enumerate(critic_losses)}
        )
        metrics["vc_critic_loss"] = critic_loss
        return metrics

    def _calculate_target_q_values(
        self,
        batch,
        next_state_inputs,
        target_critic,
        entropy=True,
        compute_separate_targets=False,
    ):
        with torch.no_grad():
            next_actions, next_log_prob, _ = self._actor(*next_state_inputs)
            next_q_vals = torch.stack(target_critic(*next_state_inputs, next_actions))
            if not compute_separate_targets:
                next_q_vals = torch.amin(next_q_vals, dim=0, keepdim=True).broadcast_to(
                    next_q_vals.shape
                )  # n_critics x batch_size x n_heads

            if entropy:
                next_q_vals = next_q_vals - self._alpha * next_log_prob
            if self._trunc_as_terminal:
                terminals = batch["done"][None, :, None]
            else:
                terminals = batch["terminated"][None, :, None]
            target_q_values = (
                self._reward_scale_factor * batch["reward"][None, :, None]
                + (1 - terminals) * self._discount_rate * next_q_vals
            )

        return target_q_values

    def _update_target(self):
        """Update the target network."""
        networks = [(self._critic, self._target_critic)]
        if self._use_value_comp_critic:
            networks.append((self._value_comp_critic, self._target_value_comp_critic))
        if self._compute_success_probability:
            networks.append((self._success_critic, self._target_success_critic))
        for network, target_network in networks:
            target_params = target_network.state_dict()
            current_params = network.state_dict()
            for key in list(target_params.keys()):
                target_params[key] = (1 - self._soft_update_fraction) * target_params[
                    key
                ] + self._soft_update_fraction * current_params[key]
            target_network.load_state_dict(target_params)

    def save(self, dname):
        torch.save(
            {
                "critic": self._critic.state_dict(),
                "target_critic": self._target_critic.state_dict(),
                "critic_optimizer": self._critic_optimizer.state_dict(),
                "actor": self._actor.state_dict(),
                "actor_optimizer": self._actor_optimizer.state_dict(),
                "learn_schedule": self._learn_schedule,
                "update_schedule": self._update_schedule,
                "policy_update_schedule": self._policy_update_schedule,
            },
            os.path.join(dname, "agent.pt"),
        )
        replay_dir = os.path.join(dname, "replay")
        create_folder(replay_dir)
        self._replay_buffer.save(replay_dir)

    def compute_value(self, observations):
        observations = torch.as_tensor(
            observations, dtype=torch.float32, device=self._device
        )
        tile_shape = [1] * (len(observations.shape) + 1)
        tile_shape[1] = self._num_actions_value_computation
        observations = torch.tile(observations[:, None, ...], tuple(tile_shape))
        observations = observations.view(-1, *observations.shape[2:])
        actions = self._actor(observations)[0]
        if self._use_value_comp_critic:
            values = torch.stack(self._value_comp_critic(observations, actions))
        else:
            values = torch.stack(self._critic(observations, actions))
        values = torch.amin(values, dim=0)
        values = torch.mean(values.view(-1, self._num_actions_value_computation), dim=1)
        return values

    def compute_success_prob(self, observation):
        observation = torch.as_tensor(
            observation, device=self._device, dtype=torch.float32
        ).unsqueeze(0)
        tile_shape = [1] * (len(observation.shape) + 1)
        tile_shape[1] = self._num_actions_value_computation
        observation = torch.tile(observation[:, None, ...], tuple(tile_shape))
        observation = observation.view(-1, *observation.shape[2:])
        actions = self._actor(observation)[0]

        probabilities = torch.stack(self._success_critic(observation, actions))
        probabilities = torch.amin(probabilities, dim=0)
        probabilities = torch.mean(probabilities)
        return probabilities.detach().cpu().item()

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._critic.load_state_dict(checkpoint["critic"])
        self._target_critic.load_state_dict(checkpoint["target_critic"])
        self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self._actor.load_state_dict(checkpoint["actor"])
        self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self._learn_schedule = checkpoint["learn_schedule"]
        self._update_schedule = checkpoint["update_schedule"]
        self._policy_update_schedule = checkpoint["policy_update_schedule"]
        self._replay_buffer.load(os.path.join(dname, "replay"))
