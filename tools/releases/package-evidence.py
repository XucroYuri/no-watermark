from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a benchmark evidence directory into a release-friendly zip bundle.")
    parser.add_argument("--evidence-root", type=Path, required=True, help="Directory containing latest.json and latest.md.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where the zip bundle should be written.")
    parser.add_argument("--bundle-name", default="benchmark-evidence", help="Prefix for the generated archive file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_root = args.evidence_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_json = evidence_root / "latest.json"
    latest_markdown = evidence_root / "latest.md"
    if not latest_json.exists():
        raise FileNotFoundError(f"Evidence summary not found: {latest_json}")
    if not latest_markdown.exists():
        raise FileNotFoundError(f"Evidence markdown not found: {latest_markdown}")

    summary = json.loads(latest_json.read_text(encoding="utf-8"))
    benchmark_root = Path(str(summary["benchmark_root"])).resolve()
    evidence_id = str(summary["evidence_id"])
    archive_path = output_dir / f"{args.bundle_name}-{evidence_id}.zip"

    referenced_paths = _collect_referenced_paths(summary)
    included_names: list[str] = []

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(latest_json, arcname="latest.json")
        included_names.append("latest.json")
        bundle.write(latest_markdown, arcname="latest.md")
        included_names.append("latest.md")

        for path in referenced_paths:
            if path == latest_json or path == latest_markdown:
                continue
            arcname = _build_archive_name(path, benchmark_root=benchmark_root, evidence_root=evidence_root)
            bundle.write(path, arcname=arcname)
            included_names.append(arcname)

        manifest = {
            "bundle_name": args.bundle_name,
            "archive_path": str(archive_path),
            "latest_evidence_id": evidence_id,
            "evidence_status": summary.get("status"),
            "benchmark_root": str(benchmark_root),
            "included_files": included_names,
        }
        bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        included_names.append("manifest.json")

    print(
        json.dumps(
            {
                "archive_path": str(archive_path),
                "bundle_name": args.bundle_name,
                "latest_evidence_id": evidence_id,
                "evidence_status": summary.get("status"),
                "included_file_count": len(included_names),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _collect_referenced_paths(summary: object) -> list[Path]:
    references: set[Path] = set()

    def walk(value: object, *, key: str | None = None) -> None:
        if isinstance(value, dict):
            for current_key, current_value in value.items():
                walk(current_value, key=current_key)
            return
        if isinstance(value, list):
            for item in value:
                walk(item, key=key)
            return
        if isinstance(value, str) and key is not None:
            if not value.strip():
                return
            if key.endswith(("_json", "_csv", "_markdown", "_report")):
                path = Path(value)
                if path.exists():
                    references.add(path.resolve())

    walk(summary)
    return sorted(references)


def _build_archive_name(path: Path, *, benchmark_root: Path, evidence_root: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(evidence_root):
        return f"artifacts/evidence/{resolved.relative_to(evidence_root).as_posix()}"
    if resolved.is_relative_to(benchmark_root):
        return f"artifacts/{resolved.relative_to(benchmark_root).as_posix()}"
    return f"artifacts/external/{resolved.name}"


if __name__ == "__main__":
    raise SystemExit(main())
