import types
import asyncio

from jailbreak_toolbox.attacks.blackbox.implementations.evosynth.ai_agents import external_power_tools as ept


class DummyModel:
    def __init__(self, reply="ok"):
        self.reply = reply

    def query(self, text_input="", *args, **kwargs):
        return self.reply


def _make_ctx(target_model, judge_model=None):
    ctx = types.SimpleNamespace()
    ctx.context = types.SimpleNamespace()
    ctx.context.target_model = target_model
    ctx.context.judge_model = judge_model
    return ctx


def test_judge_response_fallback_keyword():
    ctx = _make_ctx(DummyModel("safe response"), judge_model=None)
    # invoke through FunctionTool to mirror runtime behavior
    res = asyncio.run(
        ept.judge_response.on_invoke_tool(ctx, '{"attack_prompt": "test", "original_query": "orig"}')
    )
    assert res["judgment_completed"] is True
    assert res["judge_type"] == "fallback_keyword"
    assert "score" in res


def test_judge_response_target_error():
    class BoomModel:
        def query(self, *_, **__):
            raise RuntimeError("boom")

    ctx = _make_ctx(BoomModel(), judge_model=None)
    res = asyncio.run(
        ept.judge_response.on_invoke_tool(ctx, '{"attack_prompt": "test", "original_query": "orig"}')
    )
    assert res["judgment_completed"] is False
    assert res["judge_type"] == "target_model_error"
    assert "boom" in res["error"]
