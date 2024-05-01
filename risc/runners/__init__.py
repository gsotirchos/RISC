from hive.utils.registry import registry
from runners.single_agent_runner import SingleAgentRunner
from runners.reset_free_runner import ResetFreeRunner

registry.register("ResetFreeRunner", ResetFreeRunner, ResetFreeRunner)
registry.register("SingleAgentRunner", SingleAgentRunner, SingleAgentRunner)
