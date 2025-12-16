import types
import asyncio
import pytest

from jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents import external_power_tools as ept


class TupleModel:
    def query(self, text_input="", tools=None, **kwargs):
        return ("tuple content", [{"id": "call1", "type": "function", "function": {"name": "ping", "arguments": {"a": 1}}}])


def _make_ctx(target_model):
    ctx = types.SimpleNamespace()
    ctx.context = types.SimpleNamespace()
    ctx.context.target_model = target_model
    ctx.context.judge_model = None
    return ctx


@pytest.mark.asyncio
async def test_judge_response_unwraps_tuple():
    ctx = _make_ctx(TupleModel())
    res = await ept.judge_response.on_invoke_tool(ctx, '{"attack_prompt": "ap", "original_query": "oq"}')
    assert res["judgment_completed"] is True
    assert res["judge_type"] == "fallback_keyword"
    assert res["target_response"] == "tuple content"
