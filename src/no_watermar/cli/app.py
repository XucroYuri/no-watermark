from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .commands.batch import configure_batch_group
from .commands.benchmark import configure_benchmark_group
from .commands.config import configure_config_group
from .commands.providers import configure_providers_group
from .commands.scan import configure_scan_group


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Root CLI for local-first watermark removal and benchmarking.")
    subparsers = parser.add_subparsers(dest="group", required=True)

    configure_scan_group(subparsers)
    configure_batch_group(subparsers)
    configure_benchmark_group(subparsers)
    configure_config_group(subparsers)
    configure_providers_group(subparsers)

    parsed = parser.parse_args(args)
    handler = getattr(parsed, "handler", None)
    if handler is None:
        parser.error("No command handler configured.")
    return int(handler(parsed))
