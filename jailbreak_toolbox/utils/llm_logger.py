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
import os

_log_lock = threading.Lock()


def _write_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as f:
        f.write(line + "\n")
        f.flush()


def log_messages(
    log_dir: str,
    model_name: str,
    messages: Iterable[Any],
    response: Optional[Any] = None,
    tool_calls: Optional[Any] = None,
) -> None:
    """
    Append a single JSONL record for an LLM call.

    Writes to the provided log_dir/llm_messages.log and also mirrors to ./logs/llm_messages.log
    so tailing is predictable even if callers pass custom paths.
    """
    try:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "model": model_name,
            "messages": list(messages),
            "response": response,
            "tool_calls": tool_calls,
        }
        line = json.dumps(entry, ensure_ascii=False)

        primary = Path(log_dir or "./logs") / "llm_messages.log"
        mirror = Path("./logs/llm_messages.log")
        session_path_env = os.getenv("EVOSYNTH_SESSION_LOG")
        session_path = Path(session_path_env) if session_path_env else None

        with _log_lock:
            _write_line(primary, line)
            if mirror.resolve() != primary.resolve():
                _write_line(mirror, line)
            if session_path and session_path.resolve() not in {primary.resolve(), mirror.resolve()}:
                _write_line(session_path, line)
    except Exception:
        # Logging should never break the caller
        return
