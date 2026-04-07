from __future__ import annotations

import sys
from typing import Sequence

from .cli import main as root_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    translated_args = args if args[:1] == ["benchmark"] else ["benchmark", *args]
    return root_main(translated_args)
