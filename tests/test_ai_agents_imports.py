import importlib
import pytest


AI_AGENT_MODULES = [
    "jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.autonomous_orchestrator",
    "jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.exploitation_agent",
    "jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.external_power_tools",
    "jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.master_coordinator_agent",
    "jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.reconnaissance_agent",
    "jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.session_logger",
    "jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.tool_synthesizer",
]


@pytest.mark.parametrize("mod_name", AI_AGENT_MODULES)
def test_ai_agent_modules_import(mod_name):
    """
    Smoke test: each ai_agents module should import without raising.
    This validates that required dependencies are present and the module-level
    side effects (if any) don't crash at import time.
    """
    module = importlib.import_module(mod_name)
    assert module is not None
