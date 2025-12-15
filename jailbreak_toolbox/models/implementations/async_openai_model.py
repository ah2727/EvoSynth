from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Union

from openai import AsyncOpenAI

from jailbreak_toolbox.models.base_model import BaseModel
import os
from datetime import datetime


def _extract_text_from_responses(resp: Any) -> str:
    """
    Extract text content from the Responses API shape with graceful fallbacks.
    """
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    chunks: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if isinstance(t, str):
                chunks.append(t)
    return "\n".join(chunks).strip()


class AsyncOpenAIModel(BaseModel):
    """
    Concrete BaseModel powered by AsyncOpenAI.

    - query_async(): preferred (safe under asyncio)
    - query(): sync wrapper ONLY when no event loop is running
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        # Provide chat shortcut expected by some utilities
        self.chat = self.client.chat
        self._log_path = os.getenv("OPENAI_LOG_PATH")

        # Optional: store extra defaults for later calls
        self._defaults = dict(kwargs)

    def _log_session(self, role: str, content: Any):
        """Best-effort append-only session logging."""
        if not self._log_path:
            return
        try:
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                ts = datetime.utcnow().isoformat()
                f.write(f"{ts} [{role}] {content}\n")
        except Exception:
            pass

    async def query_async(
        self,
        text_input: Union[str, List[Dict]] = "",
        image_input: Any = None,
        maintain_history: bool = False,
        **kwargs: Any,
    ) -> str:
        # Merge defaults with call overrides
        call_kwargs = {**self._defaults, **kwargs}
        temperature = call_kwargs.pop("temperature", self.temperature)

        # Chat-style messages use chat.completions
        if isinstance(text_input, list):
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=text_input,
                temperature=temperature,
                **call_kwargs,
            )
            msg = resp.choices[0].message
            response_text = (msg.content or "").strip()
            self._log_session("user", text_input)
            self._log_session("assistant", response_text)
            return response_text

        # Plain text path uses Responses API
        resp = await self.client.responses.create(
            model=self.model_name,
            input=text_input or "",
            temperature=temperature,
            **call_kwargs,
        )
        response_text = _extract_text_from_responses(resp)
        self._log_session("user", text_input)
        self._log_session("assistant", response_text)
        return response_text

    def query(
        self,
        text_input: Union[str, List[Dict]] = "",
        image_input: Any = None,
        maintain_history: bool = False,
        **kwargs: Any,
    ) -> str:
        # Safe sync wrapper ONLY if we're not already in an event loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.query_async(text_input, image_input, maintain_history, **kwargs))

        raise RuntimeError(
            "AsyncOpenAIModel.query() called inside a running event loop. "
            "Use: await model.query_async(...)"
        )
