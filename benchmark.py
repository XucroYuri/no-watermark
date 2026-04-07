from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from no_watermar.env_loader import load_local_env
    from no_watermar.cli import main as cli_main

    load_local_env(project_root / ".env")
    args = sys.argv[1:]
    translated_args = args if args[:1] == ["benchmark"] else ["benchmark", *args]
    return cli_main(translated_args)


if __name__ == "__main__":
    raise SystemExit(main())
