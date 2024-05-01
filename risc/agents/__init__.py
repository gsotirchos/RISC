import agents.goal_generators
import agents.networks

from agents.gc_agents import GoalConditionedDQNAgent, GoalConditionedSACAgent
from agents.gc_rf_agent import GCResetFree
from hive.agents.agent import Agent
from hive.utils.registry import registry

registry.register_all(
    Agent,
    {
        "GCResetFree": GCResetFree,
        "GoalConditionedDQNAgent": GoalConditionedDQNAgent,
        "GoalConditionedSACAgent": GoalConditionedSACAgent,
    },
)
