from functools import partial

import gymnasium as gym
import numpy as np
import torch
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import InitializationFn
from hive.replays import BaseReplayBuffer
from hive.utils.loggers import Logger
from hive.utils.schedule import Schedule
from hive.utils.utils import LossFn, OptimizerFn
from replays.circular_replay import CircularReplayBuffer
from dataclasses import asdict
from agents.dqn import DQNAgent
from agents.sac import SACAgent


class GoalConditionedMixin:
    """A mixin implementing augmentations to a base agent to allow for goal conditioning."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        replay_buffer: BaseReplayBuffer = None,
        **kwargs,
    ):
        self._goal_space = observation_space["desired_goal"]
        if replay_buffer is None:
            replay_buffer = CircularReplayBuffer
        replay_buffer = partial(
            replay_buffer,
            extra_storage_types={
                "desired_goal": (self._goal_space.dtype, self._goal_space.shape),
            },
        )
        super().__init__(
            observation_space=observation_space["observation"],
            replay_buffer=replay_buffer,
            **kwargs,
        )

    def create_q_networks(self, *args, **kwargs):
        """Modifies the state size to include the goal size and then calls the
        base class's create_q_networks.

        Args:
            representation_net: A network that outputs the representations that will
                be used to compute Q-values (e.g. everything except the final layer
                of the DQN).
        """
        self._state_size = (
            self._goal_space.shape[0] + self._state_size[0],
            *self._state_size[1:],
        )
        super().create_q_networks(*args, **kwargs)

    def create_networks(self, *args, **kwargs):
        """Modifies the state size to include the goal size and then calls the
        base class's create_q_networks.

        Args:
            representation_net: A network that outputs the representations that will
                be used to compute Q-values (e.g. everything except the final layer
                of the DQN).
        """
        self._state_size = (
            self._goal_space.shape[0] + self._state_size[0],
            *self._state_size[1:],
        )
        super().create_networks(*args, **kwargs)

    def preprocess_update_info(self, update_info):
        """Preprocesses the :obj:`update_info` before it goes into the replay
        buffer. This method is called by the :obj:`update` method. Adds the goal
        information to the update info.

        Args:
            update_info: Contains the information from the current timestep that the
                agent should use to update itself.
        """
        preprocessed_update_info = super().preprocess_update_info(asdict(update_info))
        preprocessed_update_info["desired_goal"] = update_info.observation[
            "desired_goal"
        ]
        preprocessed_update_info["observation"] = update_info.observation["observation"]
        preprocessed_update_info["next_observation"] = update_info.next_observation[
            "observation"
        ]
        return preprocessed_update_info

    def preprocess_update_batch(self, batch):
        """Preprocess the batch sampled from the replay buffer. Concatenates the
        goal information to the state information.

        Args:
            batch: Batch sampled from the replay buffer for the current update.

        Returns:
            (tuple):
                - (tuple) Inputs used to calculate current state values.
                - (tuple) Inputs used to calculate next state values
                - Preprocessed batch.
        """
        _, _, batch = super().preprocess_update_batch(batch)

        current_state_inputs = torch.cat(
            [batch["observation"], batch["desired_goal"]], dim=1
        )
        next_state_inputs = torch.cat(
            [batch["next_observation"], batch["desired_goal"]], dim=1
        )
        return (current_state_inputs,), (next_state_inputs,), batch

    @torch.no_grad()
    def act(self, observation, agent_traj_state=None):
        """Returns the action for the agent. Concatenates the goal information
        to the observation and then calls the base class's act method.

        Args:
            observation: The current observation.
            agent_traj_state: Contains necessary state information for the agent
                to process current trajectory. This should be updated and returned.

        Returns:
            - action
            - agent trajectory state
        """
        concatenation_axis = 0
        if len(observation["observation"].shape) != len(self._observation_space.shape):
            concatenation_axis = 1
        observation = np.concatenate(
            [observation["observation"], observation["desired_goal"]],
            axis=concatenation_axis,
        )

        return super().act(observation, agent_traj_state)

    @staticmethod
    def concat_obs_goal(method):
        """Takes in an observation (np.ndarray) and a goal (np.ndarray) and
        passes the observation-goal pair as the input to the wrapped method."""
        def wrapper(self, observations, goals, *args, **kwargs):
            if len(observations.shape) == len(self._observation_space.shape):
                observations = np.expand_dims(observations, axis=0)
            if len(goals.shape) == len(self._goal_space.shape):
                goals = np.expand_dims(goals, axis=0)
            observations, goals = (
                np.repeat(observations, goals.shape[0], axis=0),
                np.tile(goals, (observations.shape[0], 1, 1, 1))
            )
            states = np.concatenate([observations, goals], axis=1)
            return method(self, states, *args, **kwargs)
        return wrapper

    @concat_obs_goal
    def compute_value(self, states):
        """Takes in an observation (np.ndarray) and a goal (np.ndarray) and
        returns the value of the observation-goal pair."""
        return super().compute_value(states)

    @concat_obs_goal
    def compute_success_prob(self, states):
        """Takes in an observation (np.ndarray) and a goal (np.ndarray) and
        returns the probability that the agent succeeds at the task."""
        return super().compute_success_prob(states)

    @concat_obs_goal
    def compute_uncertainties(self, states):
        """Takes in an observation (np.ndarray) and a goal (np.ndarray) and
        returns the uncertaintainty of the observation-goal pair."""
        return super().compute_uncertainties(states)

    def get_stats(self, observations, goals):
        return super().get_stats(observations, goals)


class GoalConditionedDQNAgent(GoalConditionedMixin, DQNAgent):
    """Goal-conditioned DQN agent.

    This agent is a wrapper around the DQN agent that adds a goal-conditioned
    representation to the agent's Q-network. The goal-conditioned representation
    is a concatenation of the current state and the goal. The goal-conditioned
    representation is then passed to the Q-network.

    Args:
        observation_space (gym.spaces.Dict): Observation space for the agent.
        action_space (gym.spaces.Discrete): Action space for the agent.
        representation_net (FunctionApproximator): A network that outputs the
            representations that will be used to compute Q-values (e.g.
            everything except the final layer of the DQN).
        stack_size: Number of observations stacked to create the state fed to the
            DQN.
        id: Agent identifier.
        optimizer_fn (OptimizerFn): A function that takes in a list of parameters
            to optimize and returns the optimizer. If None, defaults to
            :py:class:`~torch.optim.Adam`.
        loss_fn (LossFn): Loss function used by the agent. If None, defaults to
            :py:class:`~torch.nn.SmoothL1Loss`.
        init_fn (InitializationFn): Initializes the weights of qnet using
            create_init_weights_fn.
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
        update_period_schedule (Schedule): Schedule determining how frequently
            the agent's Q-network is updated.
        target_net_soft_update (bool): Whether the target net parameters are
            replaced by the qnet parameters completely or using a weighted
            average of the target net parameters and the qnet parameters.
        target_net_update_fraction (float): The weight given to the target
            net parameters in a soft update.
        target_net_update_schedule (Schedule): Schedule determining how frequently
            the target net is updated.
        epsilon_schedule (Schedule): Schedule determining the value of epsilon
            through the course of training.
        test_epsilon (float): epsilon (probability of choosing a random action)
            to be used during testing phase.
        min_replay_history (int): How many observations to fill the replay buffer
            with before starting to learn.
        batch_size (int): The size of the batch sampled from the replay buffer
            during learning.
        device: Device on which all computations should be run.
        logger (ScheduledLogger): Logger used to log agent's metrics.
        log_frequency (int): How often to log the agent's metrics.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
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
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            representation_net=representation_net,
            stack_size=stack_size,
            id=id,
            optimizer_fn=optimizer_fn,
            loss_fn=loss_fn,
            init_fn=init_fn,
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
            **kwargs,
        )


class GoalConditionedSACAgent(GoalConditionedMixin, SACAgent):
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
        target_entropy_scale: float = 1.0,
        reward_scale_factor: float = 1.0,
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
        device="cpu",
        id=0,
        forward_demos=None,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            actor_trunk_net=actor_trunk_net,
            critic_trunk_net=critic_trunk_net,
            actor_net=actor_net,
            critic_net=critic_net,
            init_fn=init_fn,
            actor_optimizer_fn=actor_optimizer_fn,
            critic_optimizer_fn=critic_optimizer_fn,
            critic_loss_fn=critic_loss_fn,
            auto_alpha=auto_alpha,
            alpha=alpha,
            alpha_optimizer_fn=alpha_optimizer_fn,
            target_entropy_scale=target_entropy_scale,
            reward_scale_factor=reward_scale_factor,
            n_critics=n_critics,
            stack_size=stack_size,
            replay_buffer=replay_buffer,
            discount_rate=discount_rate,
            n_step=n_step,
            grad_clip=grad_clip,
            reward_clip=reward_clip,
            soft_update_fraction=soft_update_fraction,
            batch_size=batch_size,
            logger=logger,
            log_frequency=log_frequency,
            update_frequency=update_frequency,
            policy_update_frequency=policy_update_frequency,
            min_replay_history=min_replay_history,
            target_net_update_frequency=target_net_update_frequency,
            device=device,
            id=id,
            forward_demos=forward_demos,
            **kwargs,
        )
