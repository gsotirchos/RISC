import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import numpy as np
from envs.types import UpdateInfo
from hive.agents import Agent
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import PeriodicSchedule
from hive.utils.utils import create_folder
from replays.circular_replay import CircularReplayBuffer

from agents.goal_generators import GoalGenerator, GoalSwitcher
from collections import deque
import pickle


@dataclass(frozen=True)
class GCAgentState:
    subagent_traj_state: Any = None
    current_goal: Any = None
    forward: bool = True
    phase_steps: int = 0
    phase_return: int = 0
    goal_switching_state: Any = None
    forward_success: bool = False
    reset_success: bool = False
    forward_goal_idx: int = -1
    reset_goal_idx: int = -1


class GCResetFree(Agent):
    """Goal Conditioned Reset Free Agent."""

    def __init__(
        self,
        observation_space,
        action_space,
        base_agent: Agent,
        goal_generator: GoalGenerator,
        goal_switcher: GoalSwitcher,
        success_fn,
        reward_fn,
        all_states_fn,
        vis_fn,
        get_distance_fn,
        goal_states,
        initial_states,
        forward_demos,
        backward_demos,
        replace_goal_fn,
        logger: Logger,
        replay_buffer: CircularReplayBuffer,
        distance_type: str = "l2_cluster",
        phase_step_limit: int = 300,
        id=0,
        log_frequency=5,
        use_termination_signal=True,
        local_visitation_vis_frequency=2000,
        separate_agents=False,
        device="cpu",
        oracle=False,
        log_success=False,
        never_truncate=False,
        use_demo=True,
        **kwargs,
    ):
        """
        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
            base_agent: The agent that will be used to interact with the environment.
            goal_generator: The goal generator that will be used to generate new goals.
            goal_switcher: The goal switcher that will be used to determine when to
                switch goals.
            success_fn: The function that will be used to determine if the agent has
                succeeded.
            reward_fn: The function that will be used to determine the reward.
            all_states_fn: The function that will be used to get all states that the
                goal generator can use.
            vis_fn: The function that will be used to visualize the states.
            get_distance_fn: The function that will be used to get the distance between
                states.
            goal_states: The goal states.
            initial_states: The initial states.
            forward_demos: The forward demonstrations.
            backward_demos: The backward demonstrations.
            replace_goal_fn: The function that will be used to replace the goal.
            logger: The logger that will be used to log the metrics.
            replay_buffer: The replay buffer that will be used to store the transitions.
            distance_type: The type of distance function that will be used.
            phase_step_limit: The maximum number of steps the agent will take before
                switching directions.
            id: The id of the agent.
            log_frequency: The frequency of the logging.
            use_termination_signal: Whether to actually end the trajectory on a
                successful transition or to keep going until the phase_step_limit.
            local_visitation_vis_frequency: The frequency of the local visitation
                visualization.
            separate_agents: Whether to use separate agents for each direction.
            device: The device that will be used.
            oracle: Whether to use a backward oracle agent (only for 4rooms environment).
            log_success: Whether to log the success.
            never_truncate: Whether to truncate trajectories on phase_step_limit.
            use_demo: Whether to use the demo.
        """
        self._separate_agents = separate_agents
        distance_fn = get_distance_fn(distance_type=distance_type)
        super().__init__(observation_space, action_space, id)
        candidates = []
        if separate_agents:
            self._forward_agent = base_agent(
                observation_space=observation_space,
                action_space=action_space,
                id="forward_agent",
                logger=logger,
                replay_buffer=replay_buffer,
            )
            self._backward_agent = base_agent(
                observation_space=observation_space,
                action_space=action_space,
                id="backward_agent",
                logger=logger,
                replay_buffer=replay_buffer,
                oracle=oracle,
            )
        else:
            if forward_demos is not None:
                candidates.append(forward_demos["observation"])
            if backward_demos is not None:
                candidates.append(backward_demos["observation"])

            if len(candidates) > 0:
                candidates = np.concatenate(candidates, axis=0)
            self._forward_agent = base_agent(
                observation_space=observation_space,
                action_space=action_space,
                id="base_agent",
                logger=logger,
                replay_buffer=replay_buffer,
            )
            self._backward_agent = self._forward_agent
        self._goal_generator = goal_generator(
            agent=self._forward_agent,
            backward_agent=self._backward_agent,
            logger=logger,
            forward_demos=forward_demos,
            backward_demos=backward_demos,
            initial_states=initial_states,
            goal_states=goal_states,
            distance_fn=distance_fn,
            all_states_fn=all_states_fn,
            replace_goal_fn=replace_goal_fn,
            candidates=np.copy(candidates),
            device=self._forward_agent._device,
        )
        self._goal_switcher = goal_switcher(
            forward_agent=self._forward_agent,
            backward_agent=self._backward_agent,
            logger=logger,
            forward_demos=forward_demos,
            backward_demos=backward_demos,
            initial_states=initial_states,
            goal_states=goal_states,
            distance_fn=distance_fn,
            all_states_fn=all_states_fn,
            replace_goal_fn=replace_goal_fn,
            candidates=np.copy(candidates),
            device=self._forward_agent._device,
            log_success=log_success,
        )
        self._success_fn = success_fn
        self._reward_fn = reward_fn
        self._all_states_fn = all_states_fn
        self._vis_fn = vis_fn
        if self._vis_fn is not None:
            self._forward_local = deque(maxlen=local_visitation_vis_frequency)
            self._backward_local = deque(maxlen=local_visitation_vis_frequency)
            self._forward_global = 0
            self._backward_global = 0
            self._local_visitation_vis_frequency = local_visitation_vis_frequency
            self._vis_schedule = PeriodicSchedule(
                False, True, local_visitation_vis_frequency
            )
        self._distance_fn = get_distance_fn(distance_type="l2_cluster")
        self._goal_states = goal_states
        self._initial_states = initial_states
        self._forward_demos = forward_demos
        self._backward_demos = backward_demos
        self._replace_goal_fn = replace_goal_fn
        if use_demo and forward_demos is not None:
            self._forward_agent.load_demos(forward_demos)
        if use_demo and backward_demos is not None:
            self._backward_agent.load_demos(backward_demos)
        self._logger = logger
        if self._logger is None:
            self._logger = NullLogger([])
        self._logger.register_timescale(
            self._id, PeriodicSchedule(False, True, log_frequency)
        )
        self._timescale = self.id
        self._phase_step_limit = phase_step_limit
        self._use_termination_signal = use_termination_signal
        self._log_success = log_success
        self._never_truncate = never_truncate
        if log_success:
            self._success_table = []
        else:
            self._success_table = None

    def act(self, observation, agent_traj_state=None):
        if not self._training:
            return self._forward_agent.act(observation, agent_traj_state)
        if agent_traj_state is None:
            agent_traj_state = GCAgentState()
        if agent_traj_state.current_goal is None:
            agent_traj_state = self.get_new_goal(observation, agent_traj_state)

        observation = self._replace_goal_fn(observation, agent_traj_state.current_goal)
        agent = (
            self._forward_agent if agent_traj_state.forward else self._backward_agent
        )
        action, subagent_traj_state = agent.act(
            observation, agent_traj_state.subagent_traj_state
        )
        return action, replace(
            agent_traj_state,
            subagent_traj_state=subagent_traj_state,
            phase_steps=agent_traj_state.phase_steps + 1,
        )

    def update(self, update_info: UpdateInfo, agent_traj_state):
        if not self._training:
            return agent_traj_state
        if self._vis_fn is not None:
            if agent_traj_state.forward:
                self._forward_local.append(update_info.next_observation["observation"])
            else:
                self._backward_local.append(update_info.next_observation["observation"])
            if self._vis_schedule.update():
                self._log_visualizations()

        terminated, truncated, success = self.should_switch(
            update_info, agent_traj_state
        )
        forward_success = agent_traj_state.forward_success
        reset_success = agent_traj_state.reset_success
        if agent_traj_state.forward:
            forward_success = forward_success or success
        else:
            reset_success = reset_success or success
        update_info = replace(
            update_info,
            observation=self._replace_goal_fn(
                update_info.observation, agent_traj_state.current_goal
            ),
            next_observation=self._replace_goal_fn(
                update_info.next_observation, agent_traj_state.current_goal
            ),
            reward=self._reward_fn(
                update_info.next_observation,
                agent_traj_state.current_goal,
            ),
            terminated=success == 1,
            truncated=truncated,
        )
        agent = (
            self._forward_agent if agent_traj_state.forward else self._backward_agent
        )
        subagent_traj_state = agent.update(
            update_info, agent_traj_state.subagent_traj_state
        )
        agent_traj_state = replace(
            agent_traj_state,
            phase_return=agent_traj_state.phase_return + update_info.reward,
            subagent_traj_state=subagent_traj_state,
            forward_success=forward_success,
            reset_success=reset_success,
        )
        if terminated or truncated:
            if self._logger.update_step(self._timescale):
                self._logger.log_metrics(
                    {
                        "return": agent_traj_state.phase_return,
                        "steps": agent_traj_state.phase_steps,
                        "distance": self._distance_fn(
                            update_info.next_observation["observation"]
                        ),
                    },
                    "forward" if agent_traj_state.forward else "reset",
                )

            agent_traj_state = GCAgentState(
                forward=not agent_traj_state.forward,
                forward_success=agent_traj_state.forward_success,
                forward_goal_idx=agent_traj_state.forward_goal_idx,
                reset_success=agent_traj_state.reset_success,
                reset_goal_idx=agent_traj_state.reset_goal_idx,
            )
        return agent_traj_state

    def _log_visualizations(self):
        metrics = {}
        if len(self._forward_local) > 0:
            forward_local_image, forward_counts = self._vis_fn(
                self._forward_local, None, name="forward/local_visitation"
            )
            self._forward_global += forward_counts
            forward_global_image, _ = self._vis_fn(
                self._forward_global,
                None,
                already_counts=True,
                name="forward/global_visitation",
            )
            metrics["forward_local_visitation"] = forward_local_image
            metrics["forward_global_visitation"] = forward_global_image
        if len(self._backward_local) > 0:
            backward_local_image, backward_counts = self._vis_fn(
                self._backward_local, None, name="reset/local_visitation"
            )
            self._backward_global += backward_counts
            backward_global_image, _ = self._vis_fn(
                self._backward_global,
                None,
                already_counts=True,
                name="reset/global_visitation",
            )
            metrics["backward_local_visitation"] = backward_local_image
            metrics["backward_global_visitation"] = backward_global_image
        if self._all_states_fn is not None:
            all_states = self._all_states_fn()["observation"]
            forward_agent_vis = self._forward_agent.get_stats(
                all_states, self._goal_states[0]
            )
            reset_agent_vis = self._backward_agent.get_stats(
                all_states, self._initial_states[0]
            )

            for key, value in forward_agent_vis.items():
                image, _ = self._vis_fn(
                    all_states, None, totals=value, max_count=1, name=f"forward/{key}"
                )
                metrics[f"forward/{key}"] = image

            for key, value in reset_agent_vis.items():
                image, _ = self._vis_fn(
                    all_states, None, totals=value, max_count=1, name=f"reset/{key}"
                )
                metrics[f"reset/{key}"] = image

        self._logger.log_metrics(metrics, self._timescale)

    def should_switch(self, update_info, agent_traj_state):
        success = self._success_fn(
            update_info.next_observation, agent_traj_state.current_goal
        )
        if self._use_termination_signal:
            terminated = success
        else:
            terminated = False
        should_switch, success_prob = self._goal_switcher.should_switch(
            update_info.observation, agent_traj_state
        )
        failure = agent_traj_state.phase_steps >= self._phase_step_limit

        truncated = not terminated and (failure or should_switch)
        truncated = truncated or update_info.truncated
        if self._never_truncate:
            truncated = not self._use_termination_signal and success
        if self._log_success:
            success = int(success)
            self._success_table.append(
                [
                    success_prob,
                    success,
                    terminated or truncated,
                    agent_traj_state.forward,
                ]
            )

        return terminated, truncated, success

    def get_new_goal(self, observation, agent_traj_state):
        goal, agent_traj_state = self._goal_generator.generate_goal(
            observation, agent_traj_state
        )
        return replace(agent_traj_state, current_goal=goal)

    def train(self):
        super().train()
        self._forward_agent.train()
        self._backward_agent.train()

    def eval(self):
        super().eval()
        self._forward_agent.eval()
        self._backward_agent.eval()

    def save(self, dname):
        agent_path = os.path.join(dname, "gc_agent")
        goal_generator_path = os.path.join(dname, "goal_generator")
        create_folder(agent_path)
        create_folder(goal_generator_path)
        self._forward_agent.save(agent_path)
        if self._separate_agents:
            backward_agent_path = os.path.join(dname, "gc_agent_backward")
            self._backward_agent.save(backward_agent_path)
        self._goal_generator.save(goal_generator_path)
        if self._log_success:
            with open(os.path.join(dname, "success_table.pkl"), "wb") as f:
                pickle.dump(self._success_table, f)

    def load(self, dname):
        agent_path = os.path.join(dname, "gc_agent")
        goal_generator_path = os.path.join(dname, "goal_generator")
        self._forward_agent.load(agent_path)
        if self._separate_agents:
            backward_agent_path = os.path.join(dname, "gc_agent_backward")
            self._backward_agent.load(backward_agent_path)
        self._goal_generator.load(goal_generator_path)
