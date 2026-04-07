from __future__ import annotations

from pathlib import Path

from ..env_loader import load_local_env
from .app import main


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    load_local_env(project_root / ".env")
    raise SystemExit(main())
