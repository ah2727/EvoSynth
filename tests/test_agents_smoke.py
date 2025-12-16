import types
import pytest
from pathlib import Path

from jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.autonomous_orchestrator import AutonomousOrchestrator
from jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.reconnaissance_agent import ReconnaissanceAgent
from jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.tool_synthesizer import ToolCreationAgent
from jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.exploitation_agent import ExploitationAgent
from jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents.master_coordinator_agent import MasterCoordinatorAgent


class DummyModel:
    def __init__(self, name="dummy"):
        self.model_name = name


def _base_config(tmp_path: Path):
    dummy_openai_model = "dummy-openai"
    return {
        "logs_dir": str(tmp_path / "logs"),
        "disable_print_redirection": True,
        "model_objects": {
            "openai_model": dummy_openai_model,
            "attack_model_base": "dummy-attack",
            "judge_model_base": "dummy-judge",
            "target_model_name": "dummy-target",
            "judge_model_name": "dummy-judge",
        },
        "target_model": DummyModel("target"),
        "judge_model": DummyModel("judge"),
    }


def test_agents_construct(tmp_path):
    cfg = _base_config(tmp_path)

    # Individual agents should construct without side effects/network calls
    ReconnaissanceAgent(cfg)
    ToolCreationAgent(cfg)
    ExploitationAgent(cfg)
    MasterCoordinatorAgent(cfg)


def test_autonomous_orchestrator_construct(tmp_path):
    cfg = _base_config(tmp_path)
    ao = AutonomousOrchestrator(cfg)
    assert ao.logs_dir
    assert ao.agents
