#!/usr/bin/env python3
"""Summarize MemArena eval-matrix progress from manifests and logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS_OUTPUT = REPO_ROOT / "eval_progress_trial_paths.json"
DEFAULT_MODELS = ["0_6b", "llama3b", "7b", "8b", "32b"]
DEFAULT_BACKENDS = ["vanilla", "inmem", "oracle", "memobase", "memos"]
DEFAULT_TRIALS = ["s2", "s3", "s4"]
MODEL_ALIASES = {
    "3b": "llama3b",
    "llama": "llama3b",
    "llama3": "llama3b",
    "llama3b": "llama3b",
}
LOG_HEADER_RE = re.compile(
    r"^\[run-eval-matrix\] \[\d+/\d+\] "
    r"backend=(?P<backend>\S+) model=(?P<model>\S+) "
    r"trial=(?P<trial>\S+) seed=(?P<seed>\S+) namespace=(?P<namespace>\S+)"
)
LOG_COMPLETE_MARKERS = (
    "[run-accuracy] completed",
    "[accuracy] overall=",
)


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def _normalise_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _iter_manifests(paths: Iterable[Path]) -> list[Path]:
    manifests: set[Path] = set()
    for raw in paths:
        path = raw.expanduser()
        if path.is_file():
            manifests.add(path.resolve())
            continue
        direct = path / "run_eval_matrix_manifest.tsv"
        if direct.exists():
            manifests.add(direct.resolve())
            continue
        for manifest in path.glob("accuracy_*/run_eval_matrix_manifest.tsv"):
            manifests.add(manifest.resolve())
    return sorted(manifests)


def _iter_matrix_logs(paths: Iterable[Path]) -> list[Path]:
    logs: set[Path] = set()
    for raw in paths:
        path = raw.expanduser()
        if path.is_file():
            if path.suffix == ".log":
                logs.add(path.resolve())
            continue
        if (path / "matrix_logs").exists():
            for log in path.glob("matrix_logs/*/*.log"):
                logs.add(log.resolve())
            continue
        for log in path.glob("accuracy_*/matrix_logs/*/*.log"):
            logs.add(log.resolve())
    return sorted(logs)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except OSError:
        return []


def _parse_matrix_log(path: Path) -> dict[str, str] | None:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None

    header = None
    for line in text.splitlines()[:20]:
        match = LOG_HEADER_RE.match(line)
        if match:
            header = match.groupdict()
            break
    if header is None:
        return None

    header["model"] = _normalise_model(header["model"])
    header["source"] = str(path)
    header["complete"] = "1" if any(marker in text for marker in LOG_COMPLETE_MARKERS) else "0"
    return header


def summarize(
    *,
    roots: list[Path],
    models: list[str],
    backends: list[str],
    trials: list[str],
) -> dict[str, object]:
    models = [_normalise_model(m) for m in models]
    expected = [(m, b, t) for m in models for b in backends for t in trials]
    expected_set = set(expected)

    manifests = _iter_manifests(roots)
    matrix_logs = _iter_matrix_logs(roots)
    finished: dict[tuple[str, str, str], str] = {}
    seen_failures: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    trial_paths: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_manifest_rows = 0
    seen_log_rows = 0

    for manifest in manifests:
        for row in _read_manifest(manifest):
            model = _normalise_model(row.get("model_tag", "").strip())
            backend = row.get("backend", "").strip()
            trial = row.get("trial", "").strip()
            key = (model, backend, trial)
            if key not in expected_set:
                continue
            seen_manifest_rows += 1
            source = str(manifest.parent)
            key_name = "|".join(key)
            returncode = row.get("returncode", "").strip()
            trial_paths[key_name].append({
                "source_type": "manifest",
                "status": "complete" if returncode == "0" else "failed",
                "path": str(manifest),
                "run_dir": source,
                "model": model,
                "backend": backend,
                "trial": trial,
                "namespace": row.get("namespace", "").strip(),
                "returncode": returncode,
            })
            if returncode == "0":
                finished[key] = source
            else:
                failure_row = dict(row)
                failure_row["source"] = source
                seen_failures[key].append(failure_row)

    for log in matrix_logs:
        row = _parse_matrix_log(log)
        if row is None:
            continue
        key = (row["model"], row["backend"], row["trial"])
        if key not in expected_set:
            continue
        seen_log_rows += 1
        key_name = "|".join(key)
        trial_paths[key_name].append({
            "source_type": "matrix_log",
            "status": "complete" if row["complete"] == "1" else "incomplete",
            "path": row["source"],
            "seed": row["seed"],
            "namespace": row["namespace"],
        })
        if row["complete"] == "1":
            finished[key] = row["source"]
        elif key not in finished:
            seen_failures[key].append({
                "model_tag": row["model"],
                "backend": row["backend"],
                "trial": row["trial"],
                "seed": row["seed"],
                "namespace": row["namespace"],
                "returncode": "incomplete_log",
                "source": row["source"],
            })

    missing = [key for key in expected if key not in finished]
    failed_without_success = {
        "|".join(key): rows[-1]
        for key, rows in sorted(seen_failures.items())
        if key not in finished and rows
    }

    by_model = {m: sum(1 for key in finished if key[0] == m) for m in models}
    by_backend = {b: sum(1 for key in finished if key[1] == b) for b in backends}
    by_trial = {t: sum(1 for key in finished if key[2] == t) for t in trials}

    return {
        "manifest_count": len(manifests),
        "log_count": len(matrix_logs),
        "seen_manifest_rows": seen_manifest_rows,
        "seen_log_rows": seen_log_rows,
        "seen_rows": seen_manifest_rows,
        "expected_total": len(expected),
        "finished_total": len(finished),
        "finished_percent": (len(finished) / len(expected) * 100.0) if expected else 0.0,
        "by_model": by_model,
        "by_backend": by_backend,
        "by_trial": by_trial,
        "missing": ["|".join(key) for key in missing],
        "failed_without_success": failed_without_success,
        "trial_paths": dict(sorted(trial_paths.items())),
        "manifests": [str(p) for p in manifests],
        "logs": [str(p) for p in matrix_logs],
    }


def _print_table(title: str, rows: dict[str, int]) -> None:
    print()
    print(f"{title}:")
    width = max([len(k) for k in rows] + [1])
    for key, value in rows.items():
        print(f"  {key:<{width}} {value:>3}")


def _write_trial_paths(path: Path, report: dict[str, object]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "finished_total": report["finished_total"],
            "expected_total": report["expected_total"],
            "finished_percent": report["finished_percent"],
            "manifest_count": report["manifest_count"],
            "log_count": report["log_count"],
            "seen_manifest_rows": report["seen_manifest_rows"],
            "seen_log_rows": report["seen_log_rows"],
        },
        "trial_paths": report["trial_paths"],
    }
    path = path.expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("out")],
        help="Output root(s), run directory/directories, or manifest TSV file(s). Default: out",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model tags. Alias: 3b maps to llama3b.",
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help="Comma-separated backend names.",
    )
    parser.add_argument(
        "--trials",
        default=",".join(DEFAULT_TRIALS),
        help="Comma-separated trial names.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--show-manifests", action="store_true", help="Print manifest paths used.")
    parser.add_argument(
        "--paths-output",
        type=Path,
        default=DEFAULT_PATHS_OUTPUT,
        help=f"Write scanned trial source paths to this JSON file. Default: {DEFAULT_PATHS_OUTPUT}",
    )
    parser.add_argument("--no-write-paths", action="store_true", help="Do not write the trial-path JSON file.")
    args = parser.parse_args()

    report = summarize(
        roots=args.paths,
        models=_split_csv(args.models, DEFAULT_MODELS),
        backends=_split_csv(args.backends, DEFAULT_BACKENDS),
        trials=_split_csv(args.trials, DEFAULT_TRIALS),
    )

    if not args.no_write_paths:
        _write_trial_paths(args.paths_output, report)
        if not args.json:
            print(f"[eval-progress] wrote trial paths: {args.paths_output}")

    if args.json:
        if not args.no_write_paths:
            print(f"[eval-progress] wrote trial paths: {args.paths_output}", file=sys.stderr)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    finished = int(report["finished_total"])
    expected = int(report["expected_total"])
    pct = float(report["finished_percent"])
    print(
        f"[eval-progress] manifests: {report['manifest_count']}  "
        f"manifest_rows: {report['seen_manifest_rows']}  "
        f"logs: {report['log_count']}  parsed_logs: {report['seen_log_rows']}"
    )
    print(f"[eval-progress] finished unique cells: {finished}/{expected} ({pct:.1f}%)")

    _print_table("finished by model", report["by_model"])  # type: ignore[arg-type]
    _print_table("finished by backend", report["by_backend"])  # type: ignore[arg-type]
    _print_table("finished by trial", report["by_trial"])  # type: ignore[arg-type]

    missing = report["missing"]
    print()
    print(f"missing ({len(missing)}):")
    for item in missing:
        print(f"  {item}")

    failed = report["failed_without_success"]
    if failed:
        print()
        print("seen failed/incomplete rows without a successful duplicate:")
        for key, row in failed.items():  # type: ignore[union-attr]
            print(f"  {key} rc={row.get('returncode')} source={row.get('source')}")

    if args.show_manifests:
        print()
        print("manifests:")
        for path in report["manifests"]:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
