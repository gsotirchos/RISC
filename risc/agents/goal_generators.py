from dataclasses import replace

import numpy as np
from hive import registry
from hive.utils.loggers import Logger, NullLogger
from hive.utils.registry import Registrable
from hive.utils.schedule import PeriodicSchedule
from hive.utils.utils import seeder
from scipy.special import expit as sigmoid
from scipy.special import softmax
from replays.counts_replay import HashStorage

epsilon = np.finfo(float).eps


def softmin(x):
    return softmax(-x)


def standardize(x):
    return (x - np.mean(x)) / (np.std(x) + epsilon)


class GoalGenerator(Registrable):
    """Base class for Goal generators. Each subclass should implement
    generate_goaln which takes in arbitrary arguments and returns a goal.

    Args:
        goal_candidates_fn (Callable): A function that returns a list of goal
        candidates.
    """

    @classmethod
    def type_name(cls):
        return "goal_generator"

    def generate_goal(self, observation, agent_traj_state):
        raise NotImplementedError

    def __init__(self, logger=None, **kwargs):
        self._logger = NullLogger() if logger is None else logger

    def save(self, dname):
        pass

    def load(self, dname):
        pass


class FBGoalGenerator(GoalGenerator):
    """Generates goals from a fixed set of initial and goal states."""

    def __init__(
        self,
        logger,
        initial_states,
        goal_states,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        self._initial_states = initial_states
        self._goal_states = goal_states
        self._rng = np.random.default_rng(seeder.get_new_seed("goal_generator"))

    def generate_goal(self, observation, agent_traj_state):
        if agent_traj_state.forward:
            goal = self._goal_states[self._rng.integers(len(self._goal_states))]
        else:
            goal = self._initial_states[self._rng.integers(len(self._initial_states))]
        return goal


class OmniGoalGenerator(GoalGenerator):
    """Generates goals from the set of expored states (frontier) or start/goal states."""

    def __init__(
        self,
        forward_agent,
        backward_agent,
        logger,
        initial_states,
        goal_states,
        weights,
        log_frequency: int = 1,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        self._forward_agent = forward_agent
        self._backward_agent = backward_agent
        self._lateral_agent = None
        self._logger = logger
        self._rng = np.random.default_rng(seeder.get_new_seed("goal_switcher"))
        self._log_schedule = PeriodicSchedule(False, True, log_frequency)
        self._initial_states = initial_states
        self._goal_states = goal_states
        self._weights = weights

    def _get_knn_distances(self, counts, distances, k, max_visitations):
        all_visited_states = np.array([state for state in counts])
        max_visitations = max_visitations or 0
        newly_visited_states = np.array(
            [state for state, count in counts.items() if count <= max_visitations]
        )
        if newly_visited_states.size == 0:
            newly_visited_states = all_visited_states
        knn_distances = HashStorage()
        for state in newly_visited_states:
            neighbors_dists = [
                distances[state, other_state] for other_state in all_visited_states
                if not np.array_equal(other_state, state)
            ]
            # print(f"neighbors_dists[{self._debug_fmt(state[0])}]: {sorted(neighbors_dists)}")
            kk = min(len(neighbors_dists), k)
            knn_distances[state] = np.mean(np.partition(neighbors_dists, kk-1)[:kk])
        return knn_distances

    def _get_proportion(self, dictionary, proportion):
        sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        num_keys = np.ceil(proportion * len(sorted_items)).astype(int)
        selected_keys = np.array([key for key, value in sorted_items[:num_keys]])
        return selected_keys

    def _frontier_states(self, proportion=0.5, k=4, max_visitations=None):
        counts = self._lateral_agent._replay_buffer.counts
        if len(counts) == 0:
            return None
        distances = self._lateral_agent._replay_buffer.distances
        knn_distances = self._get_knn_distances(counts, distances, k, max_visitations)
        frontier_states = self._get_proportion(knn_distances, proportion)
        # print(f"    visited_states: {self._debug_fmt(visited_states[:, 0])}")
        # print("    knn_distances[newly_visited_states]")
        # _ = [print(f"        {self._debug_fmt(state[0])}: {dist}") for state, dist in knn_distances.items()]
        return frontier_states

    def _novelty(self, states):
        if states.ndim == len(self._lateral_agent._observation_space.shape):
            states = np.expand_dims(states, axis=0)
        counts = self._lateral_agent._replay_buffer.counts
        return np.array([1 / counts[state] for state in states])

    def _confidence(self, observations, goals, agent=None):
        if agent is None:
            agent = self._lateral_agent
        if observations.ndim == len(agent._observation_space.shape):
            observations = np.expand_dims(observations, axis=0)
        if goals.ndim == len(agent._goal_space.shape):
            goals = np.expand_dims(goals, axis=1)
        observations, goals = (
            np.repeat(observations, goals.shape[0], axis=0),
            np.tile(goals, (observations.shape[0], 1, 1, 1))
        )
        return np.array([agent.compute_success_prob(observation, goal)
                         for observation, goal in zip(observations, goals, strict=True)])

    def _cost(self, *args):
        return 1 / (self._confidence(*args) + epsilon)

    def _debug_fmt(self, states: np.ndarray, value: int=255):
        return np.flip(np.argwhere(np.array(states) == value)[..., -2:].squeeze()).tolist()

    def generate_goal(self, observation, agent_traj_state):
        # print("=== Generating goal")
        observation = observation["observation"]
        initial_state = self._initial_states[self._rng.integers(len(self._initial_states))]
        goal_state = self._goal_states[self._rng.integers(len(self._goal_states))]
        match agent_traj_state.current_direction.removeprefix("teleport_"):
            case "forward":
                return goal_state
            case "backward":
                return initial_state
            case "lateral":
                if agent_traj_state.forward:
                    self._lateral_agent = self._forward_agent
                    lateral_initial_state = initial_state
                    lateral_goal_state = goal_state
                else:
                    self._lateral_agent = self._backward_agent
                    lateral_initial_state = goal_state
                    lateral_goal_state = initial_state
                frontier_states = self._frontier_states(proportion=0.5)
                if frontier_states is None:
                    return goal_state if agent_traj_state.forward else initial_state
                if len(frontier_states) == 1:
                    return frontier_states[:, 0]
                novelty_cost = np.zeros(len(frontier_states))
                cost_to_reach = np.zeros(len(frontier_states))
                cost_to_come = np.zeros(len(frontier_states))
                cost_to_go = np.zeros(len(frontier_states))
                if self._weights[0] != 0:
                    novelty_cost = sigmoid(standardize(1 / self._novelty(frontier_states)))
                if self._weights[1] != 0:
                    cost_to_reach = self._cost(
                        observation,
                        frontier_states[:, 0]
                    )
                if self._weights[2] != 0:
                    cost_to_come = self._cost(
                        np.concatenate([lateral_initial_state, observation[1][None, ...]]),
                        frontier_states[:, 0]
                    )
                if self._weights[3] != 0:
                    cost_to_go = self._cost(
                        frontier_states,
                        lateral_goal_state
                    )
                priority = softmin(
                    novelty_cost ** self._weights[0] * (
                        + cost_to_reach * self._weights[1]
                        + cost_to_come * self._weights[2]
                        + cost_to_go * self._weights[3]
                    )
                )
                goal_idx = np.random.choice(len(priority), p=priority)  # np.argmin(priority)
                goal = frontier_states[goal_idx, 0][None, ...]
                if self._log_schedule.update():
                    self._logger.log_metrics(
                        {
                            f"{agent_traj_state.current_direction}/novelty_cost":
                                novelty_cost[goal_idx],
                            f"{agent_traj_state.current_direction}/cost_to_reach":
                                cost_to_reach[goal_idx],
                            f"{agent_traj_state.current_direction}/cost_to_come":
                                cost_to_come[goal_idx],
                            f"{agent_traj_state.current_direction}/cost_to_go":
                                cost_to_go[goal_idx],
                        },
                        "goal_generator",
                    )
                # print(f"    frontier_states: {self._debug_fmt(frontier_states[:, 0])} (unordered)")
                # print(f"    visitations: {1 / self._novelty(frontier_states)}")
                # print(f"    standardized vis.: {standardize(1 / self._novelty(frontier_states))}")
                # print(f"    novelty_cost: {np.round(novelty_cost, decimals=3)}")
                # print(f"    priority: {priority}")
                # print(f"    lateral goal: {self._debug_fmt(goal)}")
                return goal


registry.register_all(
    GoalGenerator,
    {
        "FBGoalGenerator": FBGoalGenerator,
        "OmniGoalGenerator": OmniGoalGenerator,
    }
)


class GoalSwitcher(Registrable):
    def __init__(self, **kwargs): ...

    @classmethod
    def type_name(cls):
        return "goal_switcher"

    def should_switch(self, observation, agent_traj_state):
        raise NotImplementedError


class BasicGoalSwitcher(GoalSwitcher):
    """Goal switcher that never switches goals."""

    def __init__(self, forward_agent, backward_agent, log_success, **kwargs):
        self._forward_agent = forward_agent
        self._backward_agent = backward_agent
        self.log_success = log_success

    def should_switch(self, observation, agent_traj_state):
        if self.log_success:
            agent = (
                self._forward_agent
                if agent_traj_state.forward
                else self._backward_agent
            )
            success_prob = agent.compute_success_prob(
                observation["observation"], agent_traj_state.current_goal
            )
        else:
            success_prob = 0.0
        return False, success_prob


class SuccessProbabilityGoalSwitcher(GoalSwitcher):
    """Goal switcher that switches goals based on the success probability of the
    current goal. The probability of switching is proportional to the success
    probability and increases with the current trajectory length."""

    def __init__(
        self,
        forward_agent,
        backward_agent,
        logger: Logger,
        conservative_factor: float = 0.95,
        log_frequency: int = 25,
        switch_on_backward: bool = True,
        switch_on_forward: bool = True,
        minimum_steps: int = 0,
        trajectory_proportion: float = 1.0,
        **kwargs,
    ):
        self._forward_agent = forward_agent
        self._backward_agent = backward_agent
        self._conservative_factor = conservative_factor
        self._logger = logger
        self._rng = np.random.default_rng(seeder.get_new_seed("goal_switcher"))
        self._log_schedule = PeriodicSchedule(False, True, log_frequency)
        self._switch_on_backward = switch_on_backward
        self._switch_on_forward = switch_on_forward
        self._minimum_steps = minimum_steps
        self._trajectory_proportion = trajectory_proportion

    def should_switch(self, observation, agent_traj_state):
        agent = (
            self._forward_agent if agent_traj_state.forward else self._backward_agent
        )
        success_prob = agent.compute_success_prob(
            observation["observation"], agent_traj_state.current_goal
        )
        if not agent_traj_state.forward and not self._switch_on_backward:
            return False, success_prob
        if agent_traj_state.forward and not self._switch_on_forward:
            return False, success_prob
        if agent_traj_state.phase_steps < self._minimum_steps:
            return False, success_prob
        if agent_traj_state.goal_switching_state is None:
            apply_goal_switch = self._rng.random() < self._trajectory_proportion
            agent_traj_state = replace(
                agent_traj_state, goal_switching_state=apply_goal_switch
            )
        if not agent_traj_state.goal_switching_state:
            return False, success_prob
        switching_prob = success_prob * (
            1 - self._conservative_factor**agent_traj_state.phase_steps
        )
        should_switch = self._rng.random() < switching_prob
        prefix = "forward" if agent_traj_state.forward else "backward"
        if self._log_schedule.update():
            self._logger.log_metrics(
                {
                    f"{prefix}/success_probability": success_prob,
                    f"{prefix}/switching_probability": switching_prob,
                },
                "goal_switcher",
            )
        return should_switch, success_prob


class ReverseCurriculumGoalSwitcher(GoalSwitcher):
    """Goal switcher that switches goals based on the success probability of the
    current goal. The agent switches when the success probability is below a
    threshold."""

    def __init__(
        self,
        forward_agent,
        logger: Logger,
        goal_states,
        log_frequency: int = 20,
        success_threshold: float = 0.2,
        **kwargs,
    ):
        self._forward_agent = forward_agent
        self._goal_states = goal_states
        self._logger = logger
        self._rng = np.random.default_rng(seeder.get_new_seed("goal_switcher"))
        self._log_schedule = PeriodicSchedule(False, True, log_frequency)
        self._success_threshold = success_threshold

    def should_switch(self, observation, agent_traj_state):
        if agent_traj_state.forward:
            return False
        success_prob = self._forward_agent.compute_success_prob(
            observation["observation"],
            self._goal_states[self._rng.integers(len(self._goal_states))],
        )
        should_switch = success_prob < self._success_threshold
        if should_switch:
            self._logger.log_metrics(
                {
                    "backward/traj_length": agent_traj_state.phase_steps,
                },
                "goal_switcher",
            )
        if self._log_schedule.update():
            self._logger.log_metrics(
                {"forward/success_probability": success_prob}, "goal_switcher"
            )
        return should_switch


registry.register_all(
    GoalSwitcher,
    {
        "BasicGoalSwitcher": BasicGoalSwitcher,
        "SuccessProbabilityGoalSwitcher": SuccessProbabilityGoalSwitcher,
        "ReverseCurriculumGoalSwitcher": ReverseCurriculumGoalSwitcher,
    },
)
