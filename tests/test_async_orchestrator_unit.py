import asyncio
import os
import pytest
from typing import List, Dict, Any

from jailbreak_toolbox.models.base_model import BaseModel
from jailbreak_toolbox.attacks.base_attack import BaseAttack, AttackResult
from jailbreak_toolbox.evaluators.base_evaluator import BaseEvaluator, EvaluationMetrics
from jailbreak_toolbox.datasets.implementations.static_dataset import StaticDataset
from jailbreak_toolbox.core.async_orchestrator import AsyncOrchestrator
from jailbreak_toolbox.attacks.blackbox.implementations.evosynth.evosynth_attack import EvosynthAttack


class FakeModel(BaseModel):
    def __init__(self, name="fake-model"):
        super().__init__()
        self.model_name = name

    def query(self, text_input="", image_input=None, maintain_history=False, **kwargs):
        return "ok"


class FakeAttack(BaseAttack):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    async def attack_async(self, query: str, **kwargs) -> AttackResult:
        return AttackResult(target=query, success=True, output_text="ok")

    def attack(self, query: str, **kwargs) -> AttackResult:
        # Fallback sync path delegates to async implementation
        return asyncio.get_event_loop().run_until_complete(self.attack_async(query, **kwargs))


class FakeEvaluator(BaseEvaluator):
    def evaluate(self, results: List[AttackResult]) -> EvaluationMetrics:
        if not results:
            return EvaluationMetrics(attack_success_rate=0.0)
        success_count = sum(1 for r in results if r.success)
        return EvaluationMetrics(attack_success_rate=success_count / len(results))


@pytest.mark.asyncio
async def test_async_orchestrator_runs_with_fake_attack(monkeypatch, tmp_path):
    # Avoid API key requirements by faking a local base_url
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

    model = FakeModel()
    dataset = StaticDataset(["hello world"])
    evaluator = FakeEvaluator()

    orchestrator = AsyncOrchestrator(
        model=model,
        dataset=dataset,
        attack_class=FakeAttack,
        evaluator=evaluator,
        max_concurrent_queries=1,
        base_logs_dir=str(tmp_path / "logs"),
        enable_progress_bars=False,
        model_name="fake-model",
        attack_name="fake-attack",
        attack_kwargs={},
        orchestrator_config={"base_logs_dir": str(tmp_path / "logs")},
        judge_model=model,
    )

    metrics, results = await orchestrator.run()

    assert len(results) == 1
    assert results[0].success is True
    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.attack_success_rate == 1.0


def test_extract_content_handles_completions_shape():
    data = {
        "choices": [
            {
                "message": {"content": "hello"},
                "delta": None,
            }
        ]
    }
    assert EvosynthAttack._extract_content(data) == "hello"

    data_delta = {
        "choices": [
            {
                "delta": {"content": "delta"},
            }
        ]
    }
    assert EvosynthAttack._extract_content(data_delta) == "delta"

    data_msg = {"message": {"content": "hi"}}
    assert EvosynthAttack._extract_content(data_msg) == "hi"
