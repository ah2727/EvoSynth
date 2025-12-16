import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_eval_async_dry_run(tmp_path):
    """
    Smoke test: the eval_async CLI should parse args and exit cleanly in dry-run mode.
    This avoids hitting real models or network but exercises end-to-end CLI wiring.
    """
    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"

    cmd = [
        sys.executable,
        "eval_async.py",
        "--dry-run",
        "--attacker-model",
        "ollama/test",
        "--judge-model",
        "ollama/test",
        "--target-models",
        "ollama/test",
        "--base-url",
        "http://localhost:11434",
        "--results-dir",
        str(results_dir),
        "--logs-dir",
        str(logs_dir),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, f"stderr: {completed.stderr}"
    assert "Dry run: configuration resolved" in completed.stdout
    # Ensure results dir was created by the CLI setup
    assert results_dir.exists()
