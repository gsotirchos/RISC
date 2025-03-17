import logging
from typing import List
from dataclasses import replace

from envs.reset_free_envs import ResetFreeEnv
from hive.agents.agent import Agent
from hive.runners.utils import TransitionInfo
from hive.utils.experiment import Experiment
from hive.utils.loggers import ScheduledLogger
from hive.utils.schedule import PeriodicSchedule
from runners.single_agent_runner import SingleAgentRunner

logging.basicConfig(
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
    level=logging.INFO,
)


class ResetFreeRunner(SingleAgentRunner):
    """Runner for reset free environments.

    Args:
        env_fn (Callable): Function that creates the environment.
        agent (Agent): Agent to be trained.
        loggers (List[ScheduledLogger]): Loggers to be used.
        experiment_manager (Experiment): Experiment manager.
        train_steps (int): Number of training steps.
        test_frequency (int): Frequency of testing.
        test_episodes (int): Number of test episodes.
        stack_size (int): Number of frames to stack.
        test_max_steps (int): Maximum number of steps per test episode.
        train_phase_steps (int): Number of training steps per forward/backward
            trajectory.
        seed (int): Seed for the random number generator.
        eval_every (bool): Whether to evaluate every possible location.
    """

    def __init__(
        self,
        env_fn: ResetFreeEnv,
        agent: Agent,
        loggers: List[ScheduledLogger],
        experiment_manager: Experiment,
        train_steps: int,
        test_frequency: int = -1,
        test_episodes: int = 1,
        stack_size: int = 1,
        test_max_steps: int = 1000000000,
        train_phase_steps: int = 2000,
        seed: int = None,
        eval_every: bool = False,
        **kwargs,
    ):
        self._test_max_steps = test_max_steps
        self._train_phase_schedule = PeriodicSchedule(True, False, train_phase_steps)

        super().__init__(
            env_fn=env_fn,
            agent=agent,
            loggers=loggers,
            experiment_manager=experiment_manager,
            train_steps=train_steps,
            test_frequency=test_frequency,
            test_episodes=test_episodes,
            stack_size=stack_size,
            max_steps_per_episode=train_steps,
            test_max_steps=test_max_steps,
            seed=seed,
            eval_every=eval_every,
        )

    def train_mode(self, training):
        self._max_steps_per_episode = int(1e9) if training else self._test_max_steps
        return super().train_mode(training)

    def run_training(self):
        """Run the training loop. Note, to ensure that the test phase is run during
        the individual runners must call :py:meth:`~Runner.update_step` in their
        :py:meth:`~Runner.run_episode` methods.
        See :py:class:`~hive.runners.single_agent_loop.SingleAgentRunner` and
        :py:class:`~hive.runners.multi_agent_loop.MultiAgentRunner` for examples."""
        self.train_mode(True)
        observation, transition_info, agent_traj_state = self.reset_environment(
            self._train_environment
        )
        while self._train_schedule.get_value():
            # Run training episode
            if not self._training:
                self.train_mode(True)
            (
                episode_metrics,
                observation,
                transition_info,
                agent_traj_state,
            ) = self.run_train_period(
                self._train_environment,
                observation,
                transition_info,
                agent_traj_state
            )
            if self._logger.should_log("train"):
                episode_metrics = episode_metrics.get_flat_dict()
                self._logger.log_metrics(episode_metrics, "train")

            # Save experiment state
            if self._save_experiment:
                self._experiment_manager.save()
                self._save_experiment = False

        # Run a final test episode and save the experiment.
        self.run_testing()
        self._experiment_manager.save()
        self._trigger_sync()

    def run_train_period(
        self, environment, observation, transition_info, agent_traj_state=None
    ):
        """Run a single train period of the environment.

        Args:
            environment (BaseEnv): Environment in which the agent will take a step in.
        """
        episode_metrics = self.create_episode_metrics()

        # Run the loop until the episode ends or times out
        while self._train_phase_schedule.update() and self._train_schedule.get_value():
            (
                terminated,
                truncated,
                observation,
                agent_traj_state,
                _
            ) = self.run_one_step(
                environment,
                observation,
                episode_metrics,
                transition_info,
                agent_traj_state,
            )
            if agent_traj_state.current_direction.startswith("teleport"):
                observation, transition_info, agent_traj_state = self.teleport_to_goal(
                    environment, agent_traj_state
                )
            if terminated or truncated:
                observation, transition_info, agent_traj_state = self.reset_environment(
                    environment
                )
                #self._logger.log_scalar(
                #    "num_interventions",
                #    self._train_environment._env.num_interventions,
                #    "train",
                #)

            self.update_step()

        return episode_metrics, observation, transition_info, agent_traj_state

    def reset_environment(self, environment):
        observation, _ = environment.reset()
        transition_info = TransitionInfo(self._agents, self._stack_size)
        transition_info.start_agent(self._agents[0])
        agent_traj_state = None
        return observation, transition_info, agent_traj_state

    def teleport_to_goal(self, environment, agent_traj_state):
        observation = environment.teleport(agent_traj_state.current_goal)
        transition_info = TransitionInfo(self._agents, self._stack_size)
        agent_traj_state = replace(agent_traj_state, current_goal=None)
        return observation, transition_info, agent_traj_state
