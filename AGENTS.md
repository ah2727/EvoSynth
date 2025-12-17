# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `jailbreak_toolbox/` with EvoSynth logic under `jailbreak_toolbox/attacks/blackbox/implementations/evosynth/` (agents, config, data_structures, utils).
- Entry points: `eval_async.py` for CLI evaluations and `scripts/run_session.py` for orchestrator samples; helper diagnostics in `scripts/diagnose_orchestrator_import.py` and `scripts/openai_connectivity_probe.py`.
- Datasets and artifacts: `data/harmbench.csv` for prompts; generated logs in `logs/` or `async_logs/`; static assets under `asset/`.
- Tests live in `tests/` and mirror agent modules (async orchestrator, tool payloads, import health).

## Build, Test, and Development Commands
- Set up environment (Python 3.10+): `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
- Quick smoke run (requires API keys): `PYTHONPATH=. OPENAI_API_KEY=... OPENAI_MODEL=... python eval_async.py --attacker-model $OPENAI_MODEL --judge-model gpt-4o-mini --target-models gpt-4o --dataset harmbench`.
- Orchestrator sample: `PYTHONPATH=. OPENAI_API_KEY=... python scripts/run_session.py` after wiring `target_model`/`judge_model`.
- Tests (CI uses this): `PYTHONPATH=. pytest -q`; target a module with `pytest tests/test_async_orchestrator_unit.py -q`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; keep imports standard-library first, then third-party, then local.
- Prefer type hints and concise docstrings; preserve async-friendly patterns (`asyncio` + `pytest.mark.asyncio`).
- Modules and files use `snake_case.py`; classes use `CamelCase`; constants `UPPER_SNAKE`; configuration dataclasses live near their usage (see `EvosynthConfig`).
- Logging: reuse `logs_dir` plumbing and `utils/llm_logger.log_messages` where applicable instead of ad-hoc prints.

## Testing Guidelines
- Use `pytest` with `pytest-asyncio` for coroutine tests; mark async tests with `@pytest.mark.asyncio`.
- Mirror new agent behaviors with focused unit tests in `tests/` and keep fixtures lightweight (avoid live API calls; stub clients or pass fakes).
- CI runs `pytest -q` on Python 3.11 across Linux/macOS/Windows—keep tests deterministic and OS-agnostic.

## Commit & Pull Request Guidelines
- Commit messages have been short and imperative (e.g., “patch for ollama”, emoji prefixes optional); keep subjects under ~72 chars and group related changes.
- PRs should include: what changed and why, setup/env vars used (`OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`), reproduction steps or commands, and relevant logs/screenshots for failures.
- Document any new configuration knobs in `README.md` or inline docstrings; avoid committing secrets or `.env` files.

## Security & Configuration Tips
- Provide API credentials via environment variables or a local `.env` (see README) and exclude them from commits.
- When adding external calls, honor existing timeout/retry patterns (`AsyncOpenAI(...timeout=..., max_retries=...)`) and surface errors via orchestrator logging to keep attacks reproducible and auditable.
