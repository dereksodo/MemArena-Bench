#!/usr/bin/env python3
"""Stage completed eval-matrix cells into the paper-figure input layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRESS_JSON = REPO_ROOT / "eval_progress_trial_paths.json"
DEFAULT_OUT_DIR = REPO_ROOT / "out" / "paper_accuracy_available_input"
DEFAULT_MASIM_RUN = REPO_ROOT / "MASim" / "runs" / "5a10d_real"
DEFAULT_MODELS = ["0_6b", "llama3b", "7b", "8b", "32b"]
DEFAULT_BACKENDS = ["vanilla", "inmem", "oracle", "memobase", "memos"]
DEFAULT_TRIALS = ["s2", "s3", "s4"]
MODEL_ALIASES = {
    "3b": "llama3b",
    "llama": "llama3b",
    "llama3": "llama3b",
    "llama3b": "llama3b",
}
MEMCACHE_MODEL_RAW = {
    "0_6b": "qwen3_0_6b",
    "llama3b": "llama3_2_3b",
    "7b": "mistral_7b",
    "8b": "qwen3_8b",
    "32b": "qwen3_32b_awq",
}
SOURCE_SYSTEM_DIR = {
    "vanilla": "vanilla",
    "inmem": "inmem",
    "oracle": "oracle",
    "memobase": "memory_cache",
    "memos": "memory_cache",
}
BASELINE_PAPER_BACKEND = {
    "vanilla": "vanilla",
    "inmem": "rag",
    "oracle": "oracle",
}
JUDGE_REASON_PREFIXES = ("evidence_judge:", "llm_judge:")
FALLBACK_REASONS = {"token_f1", "token_f1_fallback"}


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def _normalise_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _source_root(entry: dict[str, str]) -> Path | None:
    if entry.get("source_type") == "manifest":
        run_dir = entry.get("run_dir")
        return Path(run_dir).expanduser() if run_dir else None
    path = entry.get("path")
    if not path:
        return None
    log_path = Path(path).expanduser()
    try:
        # <run>/matrix_logs/<model>/<backend>_<trial>.log
        return log_path.parents[2]
    except IndexError:
        return None


def _candidate_model_tags(model: str) -> list[str]:
    tags = [model]
    if model == "llama3b":
        tags.append("3b")
    return tags


def _source_candidates(root: Path, backend: str, model: str, trial: str) -> list[Path]:
    system_dir = SOURCE_SYSTEM_DIR[backend]
    base = root / f"eval_results_{trial}" / system_dir
    candidates: list[Path] = []
    for tag in _candidate_model_tags(model):
        candidates.append(base / f"evaluation_results_{backend}_{tag}_{trial}_judge_remote.json")
        candidates.append(base / f"evaluation_results_{backend}_{tag}_{trial}.json")
        candidates.append(base / f"evaluation_results_{backend}_{tag}.json")
    return candidates


def _source_backend_prefixes(backend: str) -> list[str]:
    if backend == "inmem":
        return ["inmem", "rag"]
    return [backend]


def _fallback_candidates(root: Path, backend: str, model: str, trial: str) -> list[Path]:
    system_dir = SOURCE_SYSTEM_DIR[backend]
    base = root / f"eval_results_{trial}" / system_dir
    if not base.exists():
        return []
    hits: list[Path] = []
    for prefix in _source_backend_prefixes(backend):
        for tag in _candidate_model_tags(model):
            hits.extend(base.glob(f"evaluation_results_{prefix}_{tag}_{trial}*.json"))
            hits.extend(base.glob(f"evaluation_results_{prefix}_{tag}.json"))
            hits.extend(base.glob(f"evaluation_results_{prefix}_*{tag}*{trial}*.json"))
            hits.extend(base.glob(f"evaluation_results_{prefix}_*{tag}.json"))
    ignored_prefixes = (
        "add_report_",
        "answer_results_",
        "answer_results_light_",
        "input_output_",
        "masim_qa_",
        "run_meta_",
        "search_results_",
    )
    return sorted(path for path in hits if not path.name.startswith(ignored_prefixes))


def _has_llm_judge(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    details = data.get("details") if isinstance(data, dict) else None
    if not isinstance(details, list):
        return False
    judged = 0
    fallback = 0
    for row in details:
        if not isinstance(row, dict) or row.get("answer_scored") is False:
            continue
        reason = str(row.get("reason") or "")
        if reason.startswith(JUDGE_REASON_PREFIXES):
            judged += 1
        if reason in FALLBACK_REASONS or reason.startswith("token_f1_fallback"):
            fallback += 1
    return judged > 0 and fallback == 0


def _find_eval_result(
    entries: list[dict[str, str]],
    backend: str,
    model: str,
    trial: str,
    *,
    require_llm_judge: bool,
) -> Path | None:
    complete_entries = [entry for entry in entries if entry.get("status") == "complete"]
    for entry in reversed(complete_entries):
        root = _source_root(entry)
        if root is None:
            continue
        primary = [path for path in _source_candidates(root, backend, model, trial) if path.exists()]
        fallback = _fallback_candidates(root, backend, model, trial)
        seen: set[Path] = set()
        primary = [path for path in primary if not (path in seen or seen.add(path))]
        fallback = [path for path in fallback if not (path in seen or seen.add(path))]

        for path in primary:
            if _has_llm_judge(path):
                return path.resolve()
        for path in fallback:
            if _has_llm_judge(path):
                return path.resolve()
        if not require_llm_judge:
            if primary:
                return primary[0].resolve()
            if fallback:
                return fallback[0].resolve()
    return None


def _paper_destination(out_dir: Path, backend: str, model: str, trial: str) -> Path:
    if backend in BASELINE_PAPER_BACKEND:
        paper_backend = BASELINE_PAPER_BACKEND[backend]
        on_disk_backend = "inmem" if backend == "inmem" else backend
        return (
            out_dir
            / f"eval_results_{trial}"
            / on_disk_backend
            / f"evaluation_results_{paper_backend}_{model}.json"
        )
    model_raw = MEMCACHE_MODEL_RAW[model]
    return (
        out_dir
        / f"eval_results_{trial}"
        / "memory_cache"
        / "memory_cache"
        / f"evaluation_results_memcache_{backend}_A_paired_{model_raw}.json"
    )


def _place_file(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def _place_dir(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copytree(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=True)


def _link_masim_artifacts(out_dir: Path, masim_run: Path, roots: list[Path], *, copy: bool) -> list[str]:
    linked: list[str] = []
    eval_instances = masim_run / "eval_instances"
    if eval_instances.exists():
        _place_dir(eval_instances.resolve(), out_dir / "eval_instances", copy=copy)
        linked.append(str(out_dir / "eval_instances"))
    corpus_sessions = masim_run / "corpus_sessions.jsonl"
    if corpus_sessions.exists():
        _place_file(corpus_sessions.resolve(), out_dir / "corpus_sessions.jsonl", copy=copy)
        linked.append(str(out_dir / "corpus_sessions.jsonl"))

    # Older gen_pvalues versions expect one canonical oracle QA file for
    # qid -> agent metadata. Keep writing it when any eval run produced a
    # masim_qa file, while current gen_pvalues can also read the seed-scoped
    # files directly.
    qa_dst = out_dir / "eval_results" / "oracle" / "masim_qa_oracle_8b.json"
    for root in roots:
        hits = sorted(root.glob("eval_results_s*/oracle/masim_qa_oracle_*.json"))
        if not hits:
            hits = sorted(root.glob("eval_results_s*/*/masim_qa_*.json"))
        if hits:
            _place_file(hits[0].resolve(), qa_dst, copy=copy)
            linked.append(str(qa_dst))
            break
    return linked


def stage_inputs(
    *,
    progress_json: Path,
    out_dir: Path,
    masim_run: Path,
    models: list[str],
    backends: list[str],
    trials: list[str],
    force: bool,
    copy: bool,
    require_llm_judge: bool,
) -> dict[str, object]:
    progress_json = _resolve_repo_path(progress_json)
    out_dir = _resolve_repo_path(out_dir)
    masim_run = _resolve_repo_path(masim_run)
    models = [_normalise_model(model) for model in models]

    if not progress_json.exists():
        raise FileNotFoundError(
            f"missing progress JSON: {progress_json}; run scripts/summarize_eval_progress.py first"
        )
    if out_dir.exists() and force:
        if out_dir.is_symlink() or out_dir.is_file():
            out_dir.unlink()
        else:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(progress_json.read_text())
    trial_paths = data.get("trial_paths") or {}
    staged: list[dict[str, str]] = []
    missing: list[str] = []
    roots: set[Path] = set()

    for model in models:
        if model not in MEMCACHE_MODEL_RAW:
            raise ValueError(f"unsupported model tag for paper staging: {model}")
        for backend in backends:
            if backend not in SOURCE_SYSTEM_DIR:
                raise ValueError(f"unsupported backend for paper staging: {backend}")
            for trial in trials:
                key = f"{model}|{backend}|{trial}"
                entries = list(trial_paths.get(key) or [])
                for entry in entries:
                    root = _source_root(entry)
                    if root is not None:
                        roots.add(root)
                src = _find_eval_result(
                    entries,
                    backend,
                    model,
                    trial,
                    require_llm_judge=require_llm_judge,
                )
                if src is None:
                    missing.append(key)
                    continue
                dst = _paper_destination(out_dir, backend, model, trial)
                _place_file(src, dst, copy=copy)
                staged.append({
                    "cell": key,
                    "source": str(src),
                    "destination": str(dst),
                })

    linked_artifacts = _link_masim_artifacts(out_dir, masim_run, sorted(roots), copy=copy)
    manifest = {
        "progress_json": str(progress_json),
        "out_dir": str(out_dir),
        "masim_run": str(masim_run),
        "mode": "copy" if copy else "symlink",
        "require_llm_judge": require_llm_judge,
        "staged_count": len(staged),
        "missing_count": len(missing),
        "staged": staged,
        "missing": missing,
        "linked_artifacts": linked_artifacts,
    }
    (out_dir / "staged_accuracy_inputs_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-json", type=Path, default=DEFAULT_PROGRESS_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--masim-run", type=Path, default=DEFAULT_MASIM_RUN)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--trials", default=",".join(DEFAULT_TRIALS))
    parser.add_argument("--force", action="store_true", help="Replace the staging directory first.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    parser.add_argument(
        "--allow-token-f1-fallback",
        action="store_true",
        help="Allow staging evaluation files that were not scored by an LLM judge.",
    )
    args = parser.parse_args()

    manifest = stage_inputs(
        progress_json=args.progress_json,
        out_dir=args.out_dir,
        masim_run=args.masim_run,
        models=_split_csv(args.models, DEFAULT_MODELS),
        backends=_split_csv(args.backends, DEFAULT_BACKENDS),
        trials=_split_csv(args.trials, DEFAULT_TRIALS),
        force=args.force,
        copy=args.copy,
        require_llm_judge=not args.allow_token_f1_fallback,
    )
    print(f"[stage-paper-inputs] staged {manifest['staged_count']} eval result file(s)")
    print(f"[stage-paper-inputs] missing {manifest['missing_count']} cell(s)")
    if manifest["missing"]:
        for cell in manifest["missing"]:
            print(f"  missing {cell}")
    print(f"[stage-paper-inputs] out-dir: {manifest['out_dir']}")
    print(f"[stage-paper-inputs] manifest: {manifest['out_dir']}/staged_accuracy_inputs_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
