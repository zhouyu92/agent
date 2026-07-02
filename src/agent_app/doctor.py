from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import AgentConfig
from .memory import MemoryStore


class ProbeClient(Protocol):
    def ping(self) -> str:
        ...


@dataclass(frozen=True)
class DoctorResult:
    ok: bool
    checks: list[str]
    errors: list[str]


def run_doctor(config: AgentConfig, client: ProbeClient | None = None, online: bool = False) -> DoctorResult:
    checks: list[str] = []
    errors: list[str] = []

    if config.api_key.strip() and config.base_url.startswith("https://") and config.model.strip():
        checks.append("config")
    else:
        errors.append("Invalid model configuration.")

    try:
        MemoryStore(config.memory_db_path)
        checks.append("memory")
    except OSError as exc:
        errors.append(f"Memory database unavailable: {exc}")

    if online:
        if client is None:
            from .llm import QwenClient

            client = QwenClient(config)
        try:
            client.ping()
            checks.append("model")
        except Exception as exc:  # pragma: no cover - exercised in real diagnostics.
            errors.append(f"Model ping failed: {exc}")

    return DoctorResult(ok=not errors, checks=checks, errors=errors)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check agent configuration and optional model connectivity.")
    parser.add_argument("--online", action="store_true", help="Also send a small request to the configured model.")
    args = parser.parse_args()

    try:
        config = AgentConfig.from_env()
    except ValueError as exc:
        print(f"Doctor failed: {exc}")
        raise SystemExit(1) from exc

    result = run_doctor(config, online=args.online)
    for check in result.checks:
        print(f"ok: {check}")
    for error in result.errors:
        print(f"error: {error}")
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
