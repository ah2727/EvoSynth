from __future__ import annotations

import os
import requests
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse, urlunparse


def _normalize_ollama_host(raw_host: Optional[str]) -> str:
    """
    Ensure host has a scheme and strip any trailing Ollama endpoint paths
    (e.g., '/api/chat' or '/api/generate') so we can append '/api/chat' safely.
    """
    if not raw_host:
        raw_host = "http://localhost:11434"
    if not raw_host.startswith(("http://", "https://")):
        raw_host = f"http://{raw_host}"

    parsed = urlparse(raw_host)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/chat"):
        path = path[: -len("/api/chat")]
    elif path.endswith("/api/generate"):
        path = path[: -len("/api/generate")]

    parsed = parsed._replace(path=path, params="", query="", fragment="")
    return urlunparse(parsed).rstrip("/")

from ..base_model import BaseModel
from ...core.registry import model_registry
from ...utils.llm_logger import log_messages


@model_registry.register("ollama")
class OllamaModel(BaseModel):
    """
    Minimal Ollama chat wrapper using the local Ollama HTTP API.

    Defaults:
      host: http://192.168.100.37:10101  (override with OLLAMA_HOST)
      model_name: llama3
    """

    def __init__(
        self,
        model_name: str = "llama3",
        host: Optional[str] = None,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        timeout: float = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name

        # Normalize host so `requests` always sees a valid URL (scheme required) and no trailing /api/chat
        raw_host = host or os.getenv("OLLAMA_HOST") or "http://192.168.100.37:10101"
        self.host = _normalize_ollama_host(raw_host)
        self.system_message = system_message
        self.temperature = temperature
        self.timeout = timeout

        # Simple conversation history
        self.conversation_history: List[Dict[str, str]] = [{"role": "system", "content": self.system_message}]

    def _build_messages(
        self,
        text_input: Union[str, List[Dict[str, Any]]],
        maintain_history: bool,
    ) -> List[Dict[str, str]]:
        if isinstance(text_input, list):
            return text_input
        if maintain_history:
            self.conversation_history.append({"role": "user", "content": str(text_input)})
            return self.conversation_history
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": str(text_input)},
        ]

    def query(
        self,
        text_input: Union[str, List[Dict[str, Any]]] = "",
        image_input: Any = None,
        maintain_history: bool = False,
        **kwargs: Any,
    ) -> str:
        # Ollama chat endpoint
        url = f"{self.host}/api/chat"
        messages = self._build_messages(text_input, maintain_history)
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
            },
            # Ollama tool-calling: pass through tools and optional tool_choice
            "tools": kwargs.get("tools"),
            "tool_choice": kwargs.get("tool_choice"),
        }

        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code == 404:
            raise RuntimeError(f"Ollama model '{self.model_name}' not found at {self.host}.")
        resp.raise_for_status()
        data = resp.json()

        # Extract text and tool calls from Ollama response
        msg = data.get("message") or {}
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

        if maintain_history:
            self.conversation_history.append({"role": "assistant", "content": content})

        # Log prompt/response for observability
        try:
            log_messages(
                log_dir=os.getenv("OPENAI_LOG_PATH") or "./logs",
                model_name=self.model_name,
                messages=messages,
                response={"content": content, "tool_calls": tool_calls},
            )
        except Exception:
            pass

        # If tool calls exist, return them alongside content (as tuple)
        if tool_calls:
            return content, tool_calls
        return content
