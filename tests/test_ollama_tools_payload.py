from jailbreak_toolbox.models.implementations.ollama_model import OllamaModel


def test_ollama_model_includes_tools_and_choice():
    model = OllamaModel(model_name="dummy", host="http://localhost:11434")
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}]
    # We won't actually send; just build the payload via query mocking by intercepting requests
    # Instead, verify the payload fields through the prepared request body
    # by monkeypatching requests.post to capture json
    captured = {}

    import requests
    real_post = requests.post

    def fake_post(url, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        class _R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"message": {"content": "ok", "tool_calls": []}}
        return _R()

    requests.post = fake_post
    try:
        model.query(text_input="hi", tools=tools, tool_choice="auto")
    finally:
        requests.post = real_post

    payload = captured["json"]
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"
