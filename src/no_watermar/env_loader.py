from __future__ import annotations

import os
from pathlib import Path


def load_local_env(env_path: Path) -> None:
    for key, value in read_local_env_file(env_path).items():
        if key in os.environ:
            continue
        os.environ[key] = value


def read_local_env_file(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _resolve_env_value(key, value.strip().strip('"').strip("'"), env_path)
        if not key:
            continue
        values[key] = value
    return values


def _resolve_env_value(key: str, value: str, env_path: Path) -> str:
    if key.endswith("_PYTHON") or key == "NO_WATERMAR_CONFIG":
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                return str((env_path.parent / candidate).resolve())
    return value
