"""
Lightweight LLM message logger.

Appends JSONL entries to ``llm_messages.log`` so runs can be inspected later.
Each line looks like:
{"ts": "...", "model": "...", "messages": [...], "response": "..."}
"""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Iterable, Optional
import json
import threading

_log_lock = threading.Lock()


def log_messages(
    log_dir: str,
    model_name: str,
    messages: Iterable[Any],
    response: Optional[Any] = None,
) -> None:
    """
    Append a single JSONL record for an LLM call.

    Args:
        log_dir: Directory to place llm_messages.log (will be created if missing)
        model_name: Name of the model used
        messages: Prompt messages sent to the model
        response: Optional response payload / text
    """
    try:
        path = Path(log_dir or "./logs") / "llm_messages.log"
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "model": model_name,
            "messages": list(messages),
            "response": response,
        }

        line = json.dumps(entry, ensure_ascii=False)
        with _log_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Logging should never break the caller
        return
