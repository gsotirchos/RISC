import copy
import logging
import os
import time
from dataclasses import asdict
from functools import partial
from typing import List
from dataclasses import replace

import numpy as np
from envs.reset_free_envs import ResetFreeEnv
from envs.types import UpdateInfo
from hive.agents.agent import Agent
from hive.runners import SingleAgentRunner as _SingleAgentRunner
from hive.runners.utils import Metrics, TransitionInfo
from hive.utils.experiment import Experiment
from hive.utils.loggers import ScheduledLogger, NullLogger
from hive.utils.utils import seeder
from wandb_osh.hooks import TriggerWandbSyncHook


class Timer:
    def __init__(self):
        self._start = None
        self._end = None

    def start(self):
        self._start = time.time()

    def stop(self):
        self._end = time.time()

    def get_time(self):
        if self._end is None:
            return time.time() - self._start
        return self._end - self._start

    def __str__(self):
        return f"Time: {self.get_time()}"


class SingleAgentRunner(_SingleAgentRunner):
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
        seed: int = None,
        test_max_steps: int = 1000000000,
        test_random_goals: bool = False,
        eval_every: bool = False,
        max_steps_per_episode: int = 1000000000,
        early_terminal: bool = True,
        send_truncation: bool = False,
    ):
        reset_free_env = env_fn(eval_every=eval_every)
        agent = partial(
            agent,
            success_fn=reset_free_env.success_fn,
            reward_fn=reset_free_env.reward_fn,
            replace_goal_fn=reset_free_env.replace_goal_fn,
            all_states_fn=reset_free_env.all_states_fn,
            vis_fn=reset_free_env.vis_fn,
            get_distance_fn=reset_free_env.get_distance_fn,
            goal_states=reset_free_env.goal_states,
            initial_states=reset_free_env.initial_states,
            forward_demos=reset_free_env.forward_demos,
            backward_demos=reset_free_env.backward_demos,
        )
        self._test_max_steps = test_max_steps
        self._early_terminal = early_terminal
        self._send_truncation = send_truncation
        self._test_random_goals = test_random_goals
        super().__init__(
            reset_free_env.train_env,
            agent,
            loggers,
            experiment_manager,
            train_steps,
            reset_free_env.eval_env,
            test_frequency,
            test_episodes,
            stack_size,
            max_steps_per_episode,
            seed,
        )
        if not isinstance(self._logger, NullLogger):
            self._logger = self._logger._logger_list[0]
        if not isinstance(self._logger, NullLogger):
            self._logger.register_timescale("train", log_timescale=True)
        if self._test_random_goals:
            self._logger.register_timescale("test_random_goals")
        self._eval_every = eval_every
        self._rng = np.random.default_rng(seeder.get_new_seed("runner"))
        self._all_states_fn = reset_free_env.all_states_fn
        self._vis_fn = reset_free_env.vis_fn
        self._train_environment.register_logger(self._logger)
        self._eval_environment.register_logger(self._logger)
        self._train_timer = Timer()
        self._eval_timer = Timer()
        self._overall_timer = Timer()
        self._train_timer.start()
        self._overall_timer.start()
        self._total_steps = 0
        self._phase_steps = 0
        if os.environ.get("WANDB_MODE", "online") == "offline":
            self._trigger_sync = TriggerWandbSyncHook()
        else:
            self._trigger_sync = lambda: None

    def update_step(self):
        """Update steps for various schedules. Run testing if appropriate."""
        if self._training:
            self._train_schedule.update()
            self._logger.update_step("train")
            self._total_steps += 1
            self._phase_steps += 1
            if self._test_schedule.update():
                self._train_timer.stop()
                self._eval_timer.start()
                self.run_testing()
                self._eval_timer.stop()
                metrics = {
                    "current_fps": self._phase_steps / self._train_timer.get_time(),
                    "train_time": self._train_timer.get_time(),
                    "eval_time": self._eval_timer.get_time(),
                    "total_fps": self._total_steps / self._overall_timer.get_time(),
                    "total_time": self._overall_timer.get_time(),
                }
                self._logger.log_metrics(metrics, "progress")
                logging.info(
                    f"{self._total_steps} environment training steps completed\n"
                    + "\n".join([f"{k}: {v}" for k, v in metrics.items()])
                )
                self._phase_steps = 0
                self._train_timer.start()
            self._save_experiment = (
                self._experiment_manager.update_step() or self._save_experiment
            )

    def run_one_step(
        self,
        environment,
        observation,
        episode_metrics,
        transition_info,
        agent_traj_state,
        active=True,
    ):
        """Run one step of the training loop.
        Args:
            observation: Current observation that the agent should create an action
                for.
            episode_metrics (Metrics): Keeps track of metrics for current episode.
        """
        agent = self._agents[0]
        stacked_observation = transition_info.get_stacked_state(agent, observation)
        action, agent_traj_state = agent.act(stacked_observation, agent_traj_state)
        (
            next_observation,
            reward,
            terminated,
            truncated,
            _,
            other_info,
        ) = environment.step(action)
        success = terminated
        if self._training:
            if self._send_truncation:
                terminated = False
                truncated = self._early_terminal and (truncated or success)
            else:
                terminated = terminated and self._early_terminal
                truncated = truncated and self._early_terminal
        # else:
        #     # never send termination/truncation signal in testing
        #     terminated = truncated = False
        update_info = UpdateInfo(
            observation=observation,
            next_observation=next_observation,
            reward=reward,
            action=action,
            terminated=terminated,
            truncated=truncated,
            info=other_info,
        )
        if self._training:
            agent_traj_state = agent.update(copy.deepcopy(update_info), agent_traj_state)

        transition_info.record_info(agent, asdict(update_info))
        episode_metrics[agent.id]["reward"] += update_info.reward * active
        episode_metrics[agent.id]["success"] = float(
            (episode_metrics[agent.id]["success"] + success) > 0
        )
        episode_metrics[agent.id]["episode_length"] += active
        episode_metrics["full_episode_length"] += active


        # handle teleport requests
        if hasattr(agent_traj_state, 'current_direction'):
            if agent_traj_state.current_direction.startswith("teleport"):
                observation, transition_info, agent_traj_state = self.teleport_to_goal(
                    environment, agent_traj_state
                )

        return (
            terminated,
            truncated,
            next_observation,
            agent_traj_state,
            active * ((terminated + truncated) == 0),
        )

    def run_end_step(
        self,
        environment,
        observation,
        episode_metrics,
        transition_info,
        agent_traj_state,
        active,
    ):
        """Run the final step of an episode.
        After an episode ends, set the truncated value to true.
        Args:
            environment (BaseEnv): Environment in which the agent will take a step in.
            observation: Current observation that the agent should create an action
                for.
            episode_metrics (Metrics): Keeps track of metrics for current
                episode.
            transition_info (TransitionInfo): Used to keep track of the most
                recent transition for the agent.
            agent_traj_state: Trajectory state object that will be passed to the
                agent when act and update are called. The agent returns a new
                trajectory state object to replace the state passed in.
        """
        agent = self._agents[0]
        stacked_observation = transition_info.get_stacked_state(agent, observation)
        action, agent_traj_state = agent.act(stacked_observation, agent_traj_state)
        next_observation, reward, terminated, _, _, other_info = environment.step(
            action
        )
        success = terminated
        if self._training and self._send_truncation:
            truncated = True
            terminated = False
        else:
            truncated = np.logical_not(terminated)

        update_info = UpdateInfo(
            observation=observation,
            next_observation=next_observation,
            reward=reward,
            action=action,
            terminated=terminated,
            truncated=truncated,
            info=other_info,
        )
        if self._training:
            agent_traj_state = agent.update(copy.deepcopy(update_info), agent_traj_state)

        transition_info.record_info(agent, asdict(update_info))
        episode_metrics[agent.id]["reward"] += update_info.reward * active
        episode_metrics[agent.id]["episode_length"] += active
        episode_metrics["full_episode_length"] += active
        episode_metrics[agent.id]["success"] = float(
            (episode_metrics[agent.id]["success"] + success) > 0
        )
        return terminated, truncated, next_observation, agent_traj_state

    def create_episode_metrics(self):
        """Create the metrics used during the loop."""
        return Metrics(
            self._agents,
            [("reward", 0), ("episode_length", 0), ("success", 0.0)],
            [("full_episode_length", 0)],
        )

    def run_training(self):
        """Run the training loop. Note, to ensure that the test phase is run during
        the individual runners must call :py:meth:`~Runner.update_step` in their
        :py:meth:`~Runner.run_episode` methods.
        See :py:class:`~hive.runners.single_agent_loop.SingleAgentRunner` and
        :py:class:`~hive.runners.multi_agent_loop.MultiAgentRunner` for examples."""
        self.train_mode(True)
        while self._train_schedule.get_value():
            #print("New training episode")
            # Run training episode
            if not self._training:
                self.train_mode(True)
            episode_metrics = self.run_episode(self._train_environment)
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

    def run_testing(self):
        """Run a testing phase."""
        if self._eval_environment is None:
            return
        self.train_mode(False)
        self.test_and_log(self._eval_environment, prefix="test")
        if self._test_random_goals:
            self.test_and_log(self._eval_environment, random_goal=True, prefix="test_random_goals")
        self._eval_environment.close()
        self._run_testing = False
        self._trigger_sync()
        self.train_mode(True)

    def test_and_log(self, environment, random_goal=False, prefix="test"):
        aggregated_episode_metrics = self.create_episode_metrics().get_flat_dict()
        for _ in range(self._test_episodes):
            if random_goal:
                environment.randomize_goal()
            episode_metrics = self.run_episode(environment, self._test_max_steps)
            for metric, value in episode_metrics.get_flat_dict().items():
                aggregated_episode_metrics[metric] += value / self._test_episodes
        self._logger.update_step(prefix)
        if self._eval_every:
            rewards = aggregated_episode_metrics[f"{self._agents[0]._id}_reward"]
            episode_lengths = aggregated_episode_metrics["full_episode_length"]
            initial_states = environment._initial_states

            def get_log_fn(metric_name, prefix):
                def log_fn(x):
                    self._logger.log_scalar(metric_name, x, prefix)
                return log_fn

            reward_plot, _ = self._vis_fn(
                initial_states,
                get_log_fn("reward_plot", prefix),
                totals=rewards,
                max_count=1,
                fmt=".1f",
            )
            episode_length_plot, _ = self._vis_fn(
                initial_states,
                get_log_fn("episode_length_plot", prefix),
                totals=episode_lengths,
                max_count=self._max_steps_per_episode,
            )
            eval_every_metrics = {
                "reward_plot": reward_plot,
                "episode_length_plot": episode_length_plot,
            }
            _, x, y = np.nonzero(np.array(initial_states)[:, 0])
            locations = np.column_stack((x, y))
            for idx, location in enumerate(locations):
                eval_every_metrics[f"reward/{location}"] = rewards[idx]
                eval_every_metrics[f"length/{location}"] = episode_lengths[idx]
            for metric, value in aggregated_episode_metrics.items():
                aggregated_episode_metrics[metric] = np.mean(value)
            self._logger.log_metrics(eval_every_metrics, "eval_every")
        self._logger.log_metrics(aggregated_episode_metrics, prefix)
        if random_goal:
            environment.reset_goal()

    def run_episode(self, environment, max_steps=None):
        """Run a single episode of the environment.

        Args:
            environment (BaseEnv): Environment in which the agent will take a step in.
        """
        if max_steps is None:
            max_steps = self._max_steps_per_episode
        episode_metrics = self.create_episode_metrics()
        terminated, truncated = False, False
        observation, transition_info, agent_traj_state = self.reset_environment(environment)
        steps = 0
        active = True
        # Run the loop until the episode ends or times out
        while (
            not (terminated or truncated)
            and steps < max_steps - 1
            and (not self._training or self._train_schedule.get_value())
        ):
            (
                terminated,
                truncated,
                observation,
                agent_traj_state,
                active,
            ) = self.run_one_step(
                environment,
                observation,
                episode_metrics,
                transition_info,
                agent_traj_state,
                active,
            )
            steps += 1
            self.update_step()
            if self._eval_every and not self._training:
                terminated, truncated = np.all(terminated), np.all(truncated)
        if not (terminated or truncated):
            self.run_end_step(
                environment,
                observation,
                episode_metrics,
                transition_info,
                agent_traj_state,
                active,
            )
            self.update_step()

        return episode_metrics

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
