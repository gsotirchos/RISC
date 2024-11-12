from dataclasses import replace

import numpy as np
from hive import registry
from hive.utils.loggers import Logger, NullLogger
from hive.utils.registry import Registrable
from hive.utils.schedule import PeriodicSchedule
from hive.utils.utils import seeder
from scipy.special import softmax


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
        return goal, agent_traj_state


class FLGoalGenerator(GoalGenerator):
    """Generates goals from a fixed set of expored (frontier) and goal states."""

    def __init__(
        self,
        agent,
        backward_agent,
        logger,
        goal_states,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        self._forward_agent = agent
        self._lateral_agent = backward_agent
        self._goal_states = goal_states
        self._rng = np.random.default_rng(seeder.get_new_seed("goal_generator"))
        self._visitation_counts = {}

    def totuple(self, arr):
        if isinstance(arr, np.ndarray):
            return tuple(self.totuple(sub_arr) for sub_arr in arr)
        else:
            return arr

    def update_novelty(self, observation):
        state = self.totuple(observation["observation"])
        self._visitation_counts[state] = self._visitation_counts.get(state, 0) + 1
        #print(f'=== Visiting: {self._debug_fmt_states(observation["observation"][0])}'
        #      + f" now visited {self._visitation_counts[state]} times")

    def _novelty(self, states):
        if len(states.shape) == len(self._forward_agent._observation_space.shape):
            states = np.expand_dims(states, axis=0)
        return np.array([1 / (1 + self._visitation_counts.get(self.totuple(state), 0))
                         for state in states])

    def _confidence(self, observations, goals, agent):
        if len(observations.shape) == len(agent._observation_space.shape):
            observations = np.expand_dims(observations, axis=0)
        if len(goals.shape) == len(agent._goal_space.shape):
            goals = np.expand_dims(goals, axis=1)
        observations, goals = \
            np.repeat(observations, goals.shape[0], axis=0), \
            np.tile(goals, (observations.shape[0], 1, 1, 1))
        return np.array([agent.compute_success_prob(observation, goal)
                         for observation, goal in zip(observations, goals, strict=True)])

    def _visited_states(self):
        return np.array([key for key in self._visitation_counts.keys()])

    def _debug_fmt_states(self, states: np.ndarray, value: int=255):
        return np.argwhere(states == value)[..., -2:].squeeze().tolist()

    def generate_goal(self, observation, agent_traj_state):
        #print("=== Generating goal")
        main_goal = self._goal_states[self._rng.integers(len(self._goal_states))]
        if agent_traj_state.forward:
            goal = main_goal
            #print("    main goal")
        else:
            frontier_states = self._visited_states()
            #print("    observation:")
            #print(f'       {self._debug_fmt_states(observation["observation"][0])}')
            #print("    frontier states:")
            #for state in frontier_states:
            #    print(f"       {self._debug_fmt_states(state[0])}: "
            #          + f"{self._visitation_counts.get(self.totuple(state))}")
            promisingness = softmax(
                self._novelty(frontier_states)
                * self._confidence(observation["observation"],
                                   frontier_states[:, 0],
                                   self._lateral_agent)
                * self._confidence(frontier_states,
                                   main_goal,
                                   self._forward_agent)
            )
            goal_idx = np.random.choice(len(promisingness), p=promisingness)
            goal = frontier_states[goal_idx, 0]
            goal = np.expand_dims(goal, axis=0)
            #print("    frontier goal:")
            #print(f"       {self._debug_fmt_states(goal)}")
        return goal, agent_traj_state


registry.register_all(
    GoalGenerator,
    {
        "FBGoalGenerator": FBGoalGenerator,
        "FLGoalGenerator": FLGoalGenerator,
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
        prefix = "forward" if agent_traj_state.forward else "reset"
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
                    "reset/traj_length": agent_traj_state.phase_steps,
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
