from __future__ import annotations

import os
import requests
from typing import Any, Dict, List, Optional, Union

from ..base_model import BaseModel
from ...core.registry import model_registry


@model_registry.register("ollama")
class OllamaModel(BaseModel):
    """
    Minimal Ollama chat wrapper using the local Ollama HTTP API.

    Defaults:
      host: http://localhost:11434  (override with OLLAMA_HOST)
      model_name: llama3
    """

    def __init__(
        self,
        model_name: str = "llama3",
        host: Optional[str] = None,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name

        # Normalize host so `requests` always sees a valid URL (scheme required)
        raw_host = host or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        if not raw_host.startswith(("http://", "https://")):
            raw_host = f"http://{raw_host}"
        self.host = raw_host.rstrip("/")
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
            "tools": kwargs.get("tools"),  # support tool-calling when provided
        }

        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code == 404:
            raise RuntimeError(f"Ollama model '{self.model_name}' not found at {self.host}.")
        resp.raise_for_status()
        data = resp.json()

        # Extract text from Ollama response
        content = ""
        msg = data.get("message") or {}
        content = msg.get("content", "")

        if maintain_history:
            self.conversation_history.append({"role": "assistant", "content": content})

        return content
