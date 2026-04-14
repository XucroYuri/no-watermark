from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.benchmark_dataset import DATASET_REGULAR, prepare_benchmark_dataset
from no_watermar.benchmark_evidence import build_stable_baseline_evidence
from no_watermar.benchmark_runner import run_benchmark
from no_watermar.disposable_benchmark_fixture import create_disposable_benchmark_fixture
from no_watermar.io_utils import print_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build disposable repeated stable evidence from repo-native synthetic inputs.")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=REPO_ROOT / "runtime" / "disposable-evidence",
        help="Workspace root for disposable inputs and benchmark outputs.",
    )
    parser.add_argument("--repetitions", type=int, default=3, help="Repeated benchmark runs per provider pair.")
    parser.add_argument(
        "--minimum-run-count",
        type=int,
        default=None,
        help="Minimum repeated runs required for evidence readiness. Defaults to --repetitions.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove the existing workspace before generating new evidence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    minimum_run_count = args.minimum_run_count or args.repetitions

    if args.clean and workspace_root.exists():
        shutil.rmtree(workspace_root)

    input_root = workspace_root / "inputs"
    benchmark_root = workspace_root / "benchmarks"

    fixture_summary = create_disposable_benchmark_fixture(input_root)
    dataset_summary = prepare_benchmark_dataset(input_root, benchmark_root, recursive=True)

    for _ in range(args.repetitions):
        run_benchmark(
            benchmark_root,
            dataset_id=DATASET_REGULAR,
            mask_provider_name="seed_manifest",
            restore_provider_name="telea",
        )
        run_benchmark(
            benchmark_root,
            dataset_id=DATASET_REGULAR,
            mask_provider_name="seed_manifest",
            restore_provider_name="noop",
        )
        run_benchmark(
            benchmark_root,
            dataset_id=DATASET_REGULAR,
            mask_provider_name="seed_manifest",
            restore_provider_name="corner_crop",
        )

    evidence_summary = build_stable_baseline_evidence(
        benchmark_root=benchmark_root,
        dataset_id=DATASET_REGULAR,
        baseline_mask_provider="seed_manifest",
        baseline_restore_provider="telea",
        candidate_mask_provider="seed_manifest",
        candidate_restore_provider="noop",
        optional_mask_provider="seed_manifest",
        optional_restore_provider="corner_crop",
        minimum_run_count=minimum_run_count,
        output_dir=benchmark_root / "evidence",
    )

    print_json(
        {
            "workspace_root": str(workspace_root),
            "fixture": fixture_summary,
            "dataset": dataset_summary,
            "evidence": evidence_summary,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
