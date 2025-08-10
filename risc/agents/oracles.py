import numpy as np
from collections import defaultdict, deque
from enum import IntEnum
from hive import registry
from hive.utils.utils import seeder
from hive.utils.loggers import Logger, NullLogger
from hive.utils.registry import Registrable


class Oracle(Registrable):
    """Base class for Oracles. Each subclass should implement
    generate_goaln which takes in arbitrary arguments and returns a goal.
    """

    @classmethod
    def type_name(cls):
        return "oracle"

    def __init__(self, logger: Logger = None, **kwargs):
        self._logger = NullLogger() if logger is None else logger
        self._rng = np.random.default_rng(seed=seeder.get_new_seed("oracle"))

    def act(self, observation, **kwargs):
        raise NotImplementedError

    def value(self, observation, step_reward=None, goal_reward=None, **kwargs):
        raise NotImplementedError

    def success_prob(self, observation, goal):
        raise NotImplementedError

    def compute_value(self, observations, goals, **kwargs):
        raise NotImplementedError

    def compute_success_prob(self, observations, goals):
        raise NotImplementedError


class MiniGridOracle(Oracle):
    def __init__(
        self,
        observation_shape,
        discount_rate=1,
        step_reward=0,
        goal_reward=1,
        observation=None,
        logger=None,
        **kwargs
    ):
        super().__init__(logger, **kwargs)
        self._observation_shape = observation_shape
        self._discount_rate = discount_rate
        self._step_reward = step_reward
        self._goal_reward = goal_reward
        self._walls = None
        self._process_observation(observation)

    class _Actions(IntEnum):
        right = 0
        down = 1
        left = 2
        up = 3

    _movements = {
        _Actions.right: (1, 0),
        _Actions.down: (0, 1),
        _Actions.left: (-1, 0),
        _Actions.up: (0, -1),
    }

    def _move(self, action, cell=(0, 0)):
        return cell[0] + self._movements[action][0], cell[1] + self._movements[action][1]

    def _process_observation(self, *args):
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

    #@staticmethod
    def processobservation(method):
        def wrapper(self, *args, **kwargs):
            self._process_observation(*args)
            return method(self, *args, **kwargs)
        return wrapper

    def _bfs(self, goal):
        dist = defaultdict(lambda: -1)  # -1 = unreachable
        dist[goal] = 0
        queue = deque([goal])
        while queue:
            cell = queue.popleft()
            for action in self._Actions:
                next_cell = self._move(action, cell)
                if next_cell in self._valid_cells and dist[next_cell] == -1:
                    dist[next_cell] = dist[cell] + 1
                    queue.append(next_cell)
        return dist

    def _compute_distances(self):
        print("Computing oracle's paths' distances... ", end='', flush=True)
        goal_dist = dict()
        for goal_cell in self._valid_cells:
            goal_dist[goal_cell] = self._bfs(goal_cell)
        distances = defaultdict(lambda: dict())
        for cell in self._valid_cells:
            for goal_cell in self._valid_cells:
                current_dist = goal_dist[goal_cell][cell]
                for action in self._Actions:
                    next_cell = self._move(action, cell)
                    if next_cell in self._valid_cells:
                        next_dist = goal_dist[goal_cell][next_cell]
                    else:
                        next_dist = current_dist

                    distances[cell, goal_cell][action] = next_dist
        print("done.")
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
        return np.argmax(self._rng.random(x.shape) * (x == x.max()), **kwargs)

    @processobservation
    def act(self, observation, **kwargs):
        return self._randargmax(self._return(self._agent2goal_distances, **kwargs))

    @processobservation
    def next_state(self, observation, action):
        next_cell = self._move(action, self._agent_cell)
        if next_cell not in self._valid_cells:
            return observation
        next_observation = np.copy(observation)
        next_observation[(0, *np.flip(next_cell))] = \
            next_observation[(0, *np.flip(self._agent_cell))]
        next_observation[(0, *np.flip(self._agent_cell))] = 0
        return next_observation

    @processobservation
    def value(self, observation, step_reward=None, goal_reward=None, **kwargs):
        return np.max(self._return(1 + self._agent2goal_distances, **kwargs), keepdims=True)

    @processobservation
    def success_prob(self, observation, goal):
        return 1 / np.min(1 + self._agent2goal_distances)

    #@staticmethod
    def broadcast_obs_goals(method):
        def wrapper(self, observations, goals, *args, **kwargs):
            if len(observations.shape) == len(self._observation_shape):
                observations = np.expand_dims(observations, axis=0)
            if len(goals.shape) == len(self._observation_shape):
                goals = np.expand_dims(goals, axis=0)
            observations, goals = (
                np.repeat(observations, goals.shape[0], axis=0),
                np.tile(goals, (observations.shape[0], 1, 1, 1))
            )
            return method(self, observations, goals, *args, **kwargs)
        return wrapper

    @broadcast_obs_goals
    def compute_value(self, observations, goals, **kwargs):
        return np.array([self.value(obs, goal, **kwargs) for obs, goal in zip(observations, goals)])

    @broadcast_obs_goals
    def compute_success_prob(self, observations, goals):
        return np.array([self.success_prob(obs, goal) for obs, goal in zip(observations, goals)])



registry.register_all(
    Oracle,
    {
        "MiniGridOracle": MiniGridOracle,
    }
)
