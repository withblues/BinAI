#!/usr/bin/env python3
"""Stage a paper artifact from explicit manifests without copying Trainer state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


EXCLUDED_NAMES = {
    ".cache",
    "__pycache__",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "wandb",
}
EXCLUDED_PREFIXES = (
    "cache-",  # Hugging Face Dataset map/filter cache shards.
    "checkpoint-",  # Hugging Face Trainer intermediate checkpoints.
)


@dataclass(frozen=True)
class ManifestEntry:
    kind: str
    group: str
    required: bool
    source: Path
    destination: Path
    paper_use: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Repository/data root containing outputs and outputs_recalculate.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="New, empty staging directory. Required unless --check-only is used.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help="TSV manifest to include. Defaults to artifact/manifest.tsv.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help="Only validate or stage these manifest groups.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate sources and print the expected size without copying.",
    )
    parser.add_argument(
        "--print-files",
        action="store_true",
        help="Print every source-to-destination mapping for manual review.",
    )
    parser.add_argument(
        "--allow-missing-optional",
        action="store_true",
        help="Do not fail when optional entries are absent.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"kind", "group", "required", "source", "destination", "paper_use"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            if row["kind"] not in {"file", "directory"}:
                raise ValueError(f"Unsupported kind {row['kind']!r} in {path}")
            entries.append(
                ManifestEntry(
                    kind=row["kind"],
                    group=row["group"],
                    required=row["required"].lower() == "yes",
                    source=Path(row["source"]),
                    destination=Path(row["destination"]),
                    paper_use=row["paper_use"],
                )
            )
    return entries


def is_excluded(path: Path) -> bool:
    return path.name in EXCLUDED_NAMES or path.name.startswith(EXCLUDED_PREFIXES)


def included_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and not is_excluded(candidate)
        and not any(is_excluded(parent) for parent in candidate.parents)
    )


def copy_directory(source: Path, destination: Path) -> None:
    for source_file in included_files(source):
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)


def copy_metric_report(source: Path, destination: Path) -> None:
    """Copy a metric report after removing host-specific provenance fields."""
    with source.open(encoding="utf-8") as handle:
        report = json.load(handle)

    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"Metric report has no protocol object: {source}")

    ground_truth_key = protocol.get("ground_truth_key")
    supported_keys = {
        ("binary_name", "function_name"),
        ("project", "binary_name", "function_name"),
    }
    if tuple(ground_truth_key or ()) not in supported_keys:
        raise ValueError(
            f"Unexpected ground-truth key in {source}: {ground_truth_key!r}"
        )
    protocol["ground_truth_key"] = ["binary_name", "function_name"]
    protocol["ground_truth_key_audit"] = (
        "No (binary_name, function_name) key spans multiple projects in the "
        "project test split; including project therefore produces identical "
        "relevance labels and metrics."
    )

    if protocol.pop("test_dataset_path", None) is not None:
        protocol["test_dataset_source"] = "project test split"
    eligible_ids_path = protocol.pop("eligible_ids_path", None)
    if eligible_ids_path:
        protocol["eligible_ids_source"] = (
            "eligible-query population from the CLAP zero-shot evaluation"
        )

    serialized = json.dumps(report, indent=2) + "\n"
    if "/home/" in serialized or "/work/" in serialized:
        raise ValueError(f"Absolute host path remains in metric report: {source}")
    destination.write_text(serialized, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    script_root = Path(__file__).resolve().parents[1]
    manifests = args.manifest or [script_root / "artifact" / "manifest.tsv"]

    entries: list[ManifestEntry] = []
    for manifest in manifests:
        entries.extend(load_manifest(manifest.resolve()))
    if args.groups:
        selected_groups = set(args.groups)
        entries = [entry for entry in entries if entry.group in selected_groups]

    missing_required: list[Path] = []
    missing_optional: list[Path] = []
    total_bytes = 0
    total_files = 0
    resolved: list[tuple[ManifestEntry, Path, list[Path]]] = []
    destinations: set[Path] = set()

    for entry in entries:
        source = source_root / entry.source
        if not source.exists():
            (missing_required if entry.required else missing_optional).append(entry.source)
            continue
        if entry.kind == "file" and not source.is_file():
            raise ValueError(f"Expected file: {source}")
        if entry.kind == "directory" and not source.is_dir():
            raise ValueError(f"Expected directory: {source}")
        if entry.destination in destinations:
            raise ValueError(f"Duplicate destination: {entry.destination}")
        destinations.add(entry.destination)
        files = included_files(source)
        total_files += len(files)
        total_bytes += sum(path.stat().st_size for path in files)
        resolved.append((entry, source, files))

    print(f"Selected manifest entries: {len(entries)}")
    print(f"Resolved files: {total_files}")
    print(f"Expected size: {total_bytes / (1024 ** 3):.2f} GiB")
    if args.print_files:
        print("Resolved copy operations:")
        for entry, source, files in resolved:
            if entry.kind == "file":
                print(f"  {source} -> {entry.destination}")
            else:
                for source_file in files:
                    relative = source_file.relative_to(source)
                    print(f"  {source_file} -> {entry.destination / relative}")
    if missing_optional:
        print("Missing optional entries:")
        for path in missing_optional:
            print(f"  {path}")
    if missing_required:
        print("Missing required entries:", file=sys.stderr)
        for path in missing_required:
            print(f"  {path}", file=sys.stderr)
        return 2
    if missing_optional and not args.allow_missing_optional:
        return 2
    if args.check_only:
        return 0
    if args.destination is None:
        raise ValueError("--destination is required unless --check-only is used")

    destination_root = args.destination.resolve()
    if destination_root.exists() and any(destination_root.iterdir()):
        raise ValueError(f"Destination must be new or empty: {destination_root}")
    destination_root.mkdir(parents=True, exist_ok=True)

    for entry, source, _ in resolved:
        destination = destination_root / entry.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry.kind == "file":
            if entry.group == "metric" and source.suffix == ".json":
                copy_metric_report(source, destination)
            else:
                shutil.copy2(source, destination)
        else:
            copy_directory(source, destination)

    copied_files = sorted(path for path in destination_root.rglob("*") if path.is_file())
    sums_path = destination_root / "SHA256SUMS"
    with sums_path.open("w", encoding="utf-8") as handle:
        for path in copied_files:
            handle.write(f"{sha256(path)}  {path.relative_to(destination_root)}\n")

    print(f"Staged {len(copied_files)} files in {destination_root}")
    print(f"Checksums written to {sums_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
