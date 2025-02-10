from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import wandb
from envs.utils import heatmap as _heatmap
from hive import registry
from hive.utils.loggers import Logger, NullLogger
from hive.utils.registry import Registrable
from hive.utils.schedule import PeriodicSchedule
from hive.utils.utils import seeder
from replays.counts_replay import HashableKeyDict
from scipy.special import expit as sigmoid
from scipy.special import softmax

np.set_printoptions(precision=3)

epsilon = np.finfo(float).eps


def heatmap(states, metric, **kwargs):
    if np.isnan(metric).any():
        print(f"Warning: NaN values in {metric}")
        metric[np.isnan(metric)] = 0
    if np.isinf(metric).any():
        print(f"Warning: Inf values in {metric}")
        metric[np.isinf(metric)] = 0
    height, width = states.shape[-2:]
    _, y, x = np.nonzero(states[:, 0])
    counts = np.zeros((height, width))
    np.add.at(counts, (np.array(y), np.array(x)), metric)
    try:
        fig, ax = plt.subplots()
        _heatmap(
            data=counts,
            row_labels=np.arange(height),
            col_labels=np.arange(width),
            ax=ax,
            mask=(counts <= 0),
            **kwargs
        )
    except Exception as e:
        print(f"Warning: Failed to generate heatmap\n{e}")
        fig = plt.figure()
    fig.tight_layout()
    image = wandb.Image(fig)
    plt.close(fig)
    return image


def softmin(x):
    return softmax(-x)


def zscore(x):
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
        frontier_proportion: float = 0.5,
        max_visitations: int = 0,
        k: int = 4,
        oracle: bool = False,
        log_frequency: int = 10,
        vis_frequency: int = 100,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        self._forward_agent = forward_agent
        self._backward_agent = backward_agent
        self._lateral_agent = None
        self._oracle = oracle
        self._logger = logger
        self._rng = np.random.default_rng(seeder.get_new_seed("goal_switcher"))
        self._log_schedule = PeriodicSchedule(False, True, log_frequency)
        self._vis_schedule = PeriodicSchedule(False, True, vis_frequency)
        self._initial_states = initial_states
        self._goal_states = goal_states
        self._weights = weights
        self._proportion = frontier_proportion
        self._max_visitations = max_visitations
        self._k = k
        self._debug = debug

    def _get_knn_dist_dict(self, visitations, distances):
        all_visited_states = np.array(list(visitations.keys()))
        self._debug_print(f"all_visited_states: {self._debug_fmt(all_visited_states[:, 0])}")
        if self._max_visitations == 0 or self._max_visitations >= max(visitations.values()):
            newly_visited_states = all_visited_states
        else:
            newly_visited_states = np.array(
                [state for state, count in visitations.items() if count <= self._max_visitations]
            )
            if newly_visited_states.size == 0:
                newly_visited_states = all_visited_states
        self._debug_print(f"newly_visited_states: {self._debug_fmt(newly_visited_states[:, 0])}")
        knn_dist_dict = HashableKeyDict()
        self._debug_print("neighbors_dists:")
        for state in newly_visited_states:
            neighbors_dists = np.array([dist for dist in distances[state].values() if dist != 0])
            self._debug_print(neighbors_dists, "    ")
            k = min(len(neighbors_dists), self._k)
            knn_dist_dict[state] = np.mean(np.partition(neighbors_dists, k-1)[:k])
        self._debug_print("knn_dist_dict[newly_visited_states]:")
        for state, dist in knn_dist_dict.items():
            self._debug_print(f"{self._debug_fmt(state[0])}: {dist}", "     ")
        if self._vis_schedule.update() and not isinstance(self._logger, NullLogger):
            print("visualizing knn")
            self._logger.log_metrics(
                {f"{self._k}-nn_mean_distance": heatmap(
                    newly_visited_states,
                    np.array(list(knn_dist_dict.values())),
                    logscale=True
                )},
                f"{self._lateral_agent._id.removesuffix('_agent')}_goal_generator",
            )
        return knn_dist_dict

    def _get_num_untried_actions(self, counts):
        num_possible_actions = self._lateral_agent._action_space.n
        return {state: num_possible_actions - len(actions) for state, actions in counts.items()}

    def _get_proportion(self, dictionary, proportion):
        num_keys = int(np.ceil(proportion * len(dictionary)))
        if num_keys >= len(dictionary):
            return np.array(list(dictionary.keys()))
        keys, values = np.array(list(dictionary.keys())), np.array(list(dictionary.values()))
        sorted_indices = np.argsort(values)[::-1]
        sorted_keys, sorted_values = keys[sorted_indices], values[sorted_indices]
        # get valid values with random tie breaking
        cutoff_value = sorted_values[num_keys - 1]
        greater_mask = sorted_values > cutoff_value
        m = np.sum(greater_mask)
        if m >= num_keys:
            return sorted_keys[greater_mask]
        threshold_keys = sorted_keys[sorted_values == cutoff_value]
        random_indices = np.random.choice(len(threshold_keys), size=num_keys - m, replace=False)
        return np.concatenate([sorted_keys[greater_mask], threshold_keys[random_indices]])

    def _get_frontier_states(self):
        counts = self._lateral_agent._replay_buffer.counts
        if len(counts) == 0:
            return None
        #return np.array(list(counts.keys()))
        #visitations = self._lateral_agent._replay_buffer.visitations
        #distances = self._lateral_agent._replay_buffer.distances
        #knn_dist_dict = self._get_knn_dist_dict(visitations, distances)
        untried_actions_counts = self._get_num_untried_actions(counts)
        frontier_states = self._get_proportion(untried_actions_counts, self._proportion)
        return frontier_states

    def _get_untried_actions(self, observation, agent):
        all_actions = set(range(agent._action_space.n))
        tried_actions = agent._replay_buffer.counts[observation].keys()
        untried_actions = list(all_actions.difference(tried_actions))
        return untried_actions or list(all_actions)

    def _novelty(self, states):
        if states.ndim == len(self._lateral_agent._observation_space.shape):
            states = np.expand_dims(states, axis=0)
        visitations = self._lateral_agent._replay_buffer.visitations
        return np.array([1 / visitations[state] for state in states])

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
        if self._oracle:
            agent = agent._oracle
        return np.array([agent.compute_success_prob(observation, goal)
                         for observation, goal in zip(observations, goals)])

    def _cost(self, *args, **kwargs):
        return 1 / (self._confidence(*args, **kwargs) + epsilon)
        # return self._confidence(*args, **kwargs)  # alt. cost

    def _debug_fmt(self, states: np.ndarray, value: int = 255):
        return np.flip(np.argwhere(np.array(states) == value)[..., -2:].squeeze(), axis=-1).tolist()

    def _debug_print(self, text, prefix="ℹ️ ", color_code="90m"):
        if not self._debug:
            return
        print(prefix + f"\033[{color_code}{text}\033[0m")

    def generate_goal(self, observation, agent_traj_state):
        observation = observation["observation"]
        initial_state = self._initial_states[self._rng.integers(len(self._initial_states))]
        goal_state = self._goal_states[self._rng.integers(len(self._goal_states))]
        self._debug_print(f"observation: {self._debug_fmt(observation[0])}")
        self._debug_print(f"curent direction: {agent_traj_state.current_direction}")
        match agent_traj_state.current_direction.removeprefix("teleport_"):
            case "forward":
                if self._debug:
                    untried_actions = self._get_untried_actions(observation, self._forward_agent)
                    action = np.min(untried_actions)  # np.random.choice(untried_actions)
                    goal = self._forward_agent._oracle.next_state(observation, action)[None, 0]
                else:
                    goal = goal_state
            case "backward":
                goal = initial_state
            case "lateral":
                if agent_traj_state.forward:
                    self._lateral_agent = self._forward_agent
                    lateral_initial_state = initial_state
                    lateral_goal_state = goal_state
                else:
                    self._lateral_agent = self._backward_agent
                    lateral_initial_state = goal_state
                    lateral_goal_state = initial_state
                frontier_states = self._get_frontier_states()
                if frontier_states is None:
                    self._debug_print("no frontier states yet", "   ")
                    return goal_state if agent_traj_state.forward else initial_state
                if len(frontier_states) == 1:
                    self._debug_print(f"goal: {self._debug_fmt(frontier_states[:, 0])}", "   ")
                    return frontier_states[:, 0]
                novelty_cost = np.zeros(len(frontier_states))
                cost_to_reach = np.zeros(len(frontier_states))
                cost_to_come = np.zeros(len(frontier_states))
                cost_to_go = np.zeros(len(frontier_states))
                if self._weights[0] != 0:
                    novelty_cost = sigmoid(zscore(1 / (self._novelty(frontier_states) + epsilon)))
                    # novelty_cost = self._novelty(frontier_states)  # alt. cost
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
                # priority = softmax(  # alt. cost
                #     novelty_cost ** self._weights[0]
                #     * cost_to_reach ** self._weights[1]
                #     * cost_to_come ** self._weights[2]
                #     * cost_to_go ** self._weights[3]
                # )
                goal_idx = np.random.choice(len(priority), p=priority)  # np.argmin(priority)
                goal = frontier_states[goal_idx, 0][None, ...]
                self._debug_print(f"frontier states: {self._debug_fmt(frontier_states[:, 0])}", "   ")
                self._debug_print(f"visitations: {1 / self._novelty(frontier_states)}", "   ")
                self._debug_print(f"stdzd vis: {zscore(1 / self._novelty(frontier_states))}", "   ")
                self._debug_print(f"novelty costs: {novelty_cost}", "   ")
                self._debug_print(f"priority: {priority}", "   ")
                if self._log_schedule.update() and not isinstance(self._logger, NullLogger):
                    self._logger.log_metrics(
                        {
                            "goal/novelty_cost": novelty_cost[goal_idx],
                            "goal/cost_to_reach": cost_to_reach[goal_idx],
                            "goal/cost_to_come": cost_to_come[goal_idx],
                            "goal/cost_to_go": cost_to_go[goal_idx],
                        },
                        f"{self._lateral_agent._id.removesuffix('_agent')}_goal_generator",
                    )
                if self._vis_schedule.update() and not isinstance(self._logger, NullLogger):
                    self._logger.log_metrics(
                        {
                            "novelty_cost": heatmap(frontier_states, novelty_cost, logscale=True),
                            "cost_to_reach": heatmap(frontier_states, cost_to_reach, vmin=0),
                            "cost_to_come": heatmap(frontier_states, cost_to_come, vmin=0),
                            "cost_to_go": heatmap(frontier_states, cost_to_go, vmin=0),
                            "priority": heatmap(frontier_states, priority, logscale=True),
                        },
                        f"{self._lateral_agent._id.removesuffix('_agent')}_goal_generator",
                    )
        self._debug_print(f"goal: {self._debug_fmt(goal)}", "   ")
        if self._debug and isinstance(self._logger, NullLogger):
            breakpoint()
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
        switch_on_lateral: bool = True,
        minimum_steps: int = 0,
        trajectory_proportion: float = 1.0,
        oracle: bool = False,
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
        self._switch_on_lateral = switch_on_lateral
        self._minimum_steps = minimum_steps
        self._trajectory_proportion = trajectory_proportion
        self._oracle = oracle

    def should_switch(self, observation, agent_traj_state):
        agent = (
            self._forward_agent if agent_traj_state.forward else self._backward_agent
        )
        if self._oracle:
            agent = agent._oracle
        success_prob = agent.compute_success_prob(
            observation["observation"], agent_traj_state.current_goal
        )
        if (agent_traj_state.current_direction == "backward" and not self._switch_on_backward
              or agent_traj_state.current_direction == "forward" and not self._switch_on_forward
              or agent_traj_state.current_direction == "lateral" and not self._switch_on_lateral):
            return False, success_prob
        if agent_traj_state.phase_steps < self._minimum_steps:
            return False, success_prob
        if agent_traj_state.goal_switching_state is None:
            apply_goal_switch = self._rng.random() < self._trajectory_proportion
            agent_traj_state = replace(agent_traj_state, goal_switching_state=apply_goal_switch)
        if not agent_traj_state.goal_switching_state:
            return False, success_prob
        switching_prob = success_prob*(1 - self._conservative_factor**agent_traj_state.phase_steps)
        should_switch =  self._rng.random() < switching_prob
        prefix = agent_traj_state.current_direction
        if self._log_schedule.update() and not isinstance(self._logger, NullLogger):
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
        if self._log_schedule.update() and not isinstance(self._logger, NullLogger):
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
