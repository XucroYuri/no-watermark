from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.review_bundle import build_review_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local side-by-side review bundle from benchmark reports."
    )
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Benchmark report JSON path. Repeat to include multiple providers.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional display label for the matching --report entry. Repeat in the same order as --report.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Optional benchmark compare JSON path to copy into the review bundle.",
    )
    parser.add_argument(
        "--trend",
        action="append",
        default=[],
        help="Optional benchmark trend artifact path to copy into the review bundle.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Review bundle output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_bundle = build_review_bundle(
        report_paths=[Path(path) for path in args.report],
        provider_labels=args.label or None,
        compare_paths=[Path(path) for path in args.compare],
        trend_paths=[Path(path) for path in args.trend],
        output_dir=Path(args.output),
    )
    print(f"Built review bundle: {Path(args.output).resolve()}")
    print(f"Providers: {', '.join(review_bundle['provider_labels'])}")
    print(f"Items: {review_bundle['item_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
