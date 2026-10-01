#!/usr/bin/env python3
"""Run LLM-as-a-judge over existing MemArena answer artifacts.

This script intentionally does not run reader inference. It reads
``answer_results_*.json`` files produced by ``scripts/run_accuracy.py`` and
writes ``evaluation_results_*.json`` files next to them. This keeps expensive
answering runs reusable when a remote judge fails, rate-limits, or needs to be
changed from one provider/model to another.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.src.masim_loader import load_corpus_sessions_dict, load_masim_qa
from eval.src.scoring import evaluate_answers, evaluate_answers_hybrid
from eval.src.types import AnswerRecord, QAItem


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"
QWEN235B_MODEL = "qwen/qwen3-235b-a22b"
DEFAULT_MODELS = ["0_6b", "llama3b", "7b", "8b", "32b"]
DEFAULT_BACKENDS = ["vanilla", "inmem", "oracle", "memobase", "memos"]
DEFAULT_TRIALS = ["s2", "s3", "s4"]
MODEL_ALIASES = {
    "3b": "llama3b",
    "llama": "llama3b",
    "llama3": "llama3b",
    "llama3b": "llama3b",
}
SYSTEM_DIR = {
    "vanilla": "vanilla",
    "inmem": "inmem",
    "oracle": "oracle",
    "memobase": "memory_cache",
    "memos": "memory_cache",
}
JUDGE_REASON_PREFIXES = ("evidence_judge:", "llm_judge:")
FALLBACK_REASONS = {"token_f1_fallback", "token_f1"}


def _load_env_file(path: Path = REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        existing = os.environ.get(key)
        if existing is None or existing.strip().upper() in {"", "EMPTY", "NONE", "NULL"}:
            os.environ[key] = value


def _normalise_openai_base(url: str) -> str:
    value = (url or "").strip().rstrip("/")
    if not value:
        return value
    return value if value.endswith("/v1") else f"{value}/v1"


def _is_missing_api_key(value: str | None) -> bool:
    return not value or value.strip().upper() in {"EMPTY", "NONE", "NULL"}


def _endpoint_requires_auth(endpoint: str) -> bool:
    lowered = str(endpoint or "").lower()
    return "openrouter.ai" in lowered or "api.openai.com" in lowered


def _first_api_key(candidates: list[str | None], *, allow_empty: bool) -> str:
    for candidate in candidates:
        if allow_empty:
            if candidate is not None and str(candidate).strip():
                return str(candidate)
        elif not _is_missing_api_key(candidate):
            return str(candidate)
    return "EMPTY" if allow_empty else ""


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower() or "judge"


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
    raw_path = entry.get("path")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    try:
        # <run>/matrix_logs/<model>/<backend>_<trial>.log
        return path.parents[2]
    except IndexError:
        return None


def _answer_namespace_candidates(namespace: str, judge_tag: str) -> list[str]:
    values = [namespace]
    if not re.search(r"_judge_[A-Za-z0-9_]+$", namespace):
        values.append(f"{namespace}_judge_{judge_tag}")
        values.append(f"{namespace}_judge_remote")
        values.append(f"{namespace}_judge_local")
    return list(dict.fromkeys(values))


def _find_answer_path(root: Path, backend: str, trial: str, namespace: str, judge_tag: str) -> Path | None:
    system_dir = SYSTEM_DIR.get(backend)
    if not system_dir:
        return None
    base = root / f"eval_results_{trial}" / system_dir
    if not base.exists():
        return None
    for ns in _answer_namespace_candidates(namespace, judge_tag):
        path = base / f"answer_results_{ns}.json"
        if path.exists():
            return path.resolve()
    hits = sorted(path for path in base.glob(f"answer_results_{namespace}*.json")
                  if "answer_results_light_" not in path.name)
    return hits[-1].resolve() if hits else None


def _infer_answer_namespace(answer_path: Path) -> str:
    stem = answer_path.stem
    prefix = "answer_results_"
    if not stem.startswith(prefix):
        raise ValueError(f"cannot infer namespace from answer path: {answer_path}")
    return stem[len(prefix):]


def _output_namespace(answer_namespace: str, judge_tag: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    match = re.search(r"^(?P<base>.+)_judge_(?P<tag>[A-Za-z0-9_]+)$", answer_namespace)
    if match:
        if match.group("tag") == judge_tag:
            return answer_namespace
        return f"{match.group('base')}_judge_{judge_tag}"
    return f"{answer_namespace}_judge_{judge_tag}"


def _eval_path_for(answer_path: Path, output_namespace: str) -> Path:
    return answer_path.parent / f"evaluation_results_{output_namespace}.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_llm_judge(path: Path, min_judged_rows: int = 1) -> bool:
    if not path.exists():
        return False
    try:
        data = _load_json(path)
    except Exception:
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
    return judged >= min_judged_rows and fallback == 0


def _reason_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        data = _load_json(path)
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for row in data.get("details", []):
        reason = str(row.get("reason") or "")
        key = reason.split(":", 1)[0] if reason else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _resolve_judge(args: argparse.Namespace) -> tuple[str, str, str, str]:
    preset = str(args.judge_preset or "openrouter").lower()
    if preset in {"openrouter", "remote", "4omini", "gpt4omini", "gpt-4o-mini"}:
        endpoint = _normalise_openai_base(args.judge_endpoint or OPENROUTER_ENDPOINT)
        model = args.judge_model or "openai/gpt-4o-mini-2024-07-18"
        api_key = _first_api_key(
            [args.judge_api_key, os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENAI_API_KEY")],
            allow_empty=False,
        )
        # --judge-tag names the output file: ``remote`` (default) keeps the
        # historical gpt-4o-mini name; scripts/judge_cell.sh passes the file tag
        # of memarena.judges (``deepseek`` for the DeepSeek judge).
        return args.judge_tag or "remote", model, endpoint, api_key
    if preset in {"qwen235b", "qwen3-235b", "qwen235"}:
        endpoint = _normalise_openai_base(args.judge_endpoint or os.getenv("QWEN235B_BASE_URL") or OPENROUTER_ENDPOINT)
        model = args.judge_model or os.getenv("QWEN235B_MODEL") or QWEN235B_MODEL
        api_key = _first_api_key(
            [
                args.judge_api_key,
                os.getenv("QWEN235B_API_KEY"),
                os.getenv("OPENROUTER_API_KEY"),
                os.getenv("OPENAI_API_KEY"),
            ],
            allow_empty=not _endpoint_requires_auth(endpoint),
        )
        return "qwen235b", model, endpoint, api_key
    if preset in {"custom", "local"}:
        if not args.judge_model or not args.judge_endpoint:
            raise SystemExit("--judge-preset custom/local requires --judge-model and --judge-endpoint")
        endpoint = _normalise_openai_base(args.judge_endpoint)
        api_key = _first_api_key(
            [args.judge_api_key, os.getenv("OPENAI_API_KEY")],
            allow_empty=(preset == "local" or not _endpoint_requires_auth(endpoint)),
        )
        return args.judge_tag or _slug(args.judge_model), args.judge_model, endpoint, api_key
    raise SystemExit(f"unsupported --judge-preset: {args.judge_preset}")


def _make_client(openai_cls: Any, api_key: str, endpoint: str, concurrency: int) -> Any:
    """OpenAI-compatible client whose connection pool fits ``concurrency``.

    The SDK's default httpx pool keeps only 100 connections alive; with 512
    requests in flight the rest would reconnect (TLS handshake) on every call.
    """
    kwargs: dict[str, Any] = {"api_key": api_key or "EMPTY", "base_url": endpoint,
                              # a stuck reply is dropped after a minute and resent by the judge
                              "timeout": 60, "max_retries": 1}
    try:
        import httpx
        from openai import DefaultHttpxClient

        n = max(1, int(concurrency))
        kwargs["http_client"] = DefaultHttpxClient(
            limits=httpx.Limits(max_connections=max(1000, n), max_keepalive_connections=n),
            timeout=60,
        )
    except ImportError:  # very old SDK: keep its defaults
        pass
    return openai_cls(**kwargs)


def _load_answers(path: Path) -> list[AnswerRecord]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"answer file must be a JSON list: {path}")
    return [AnswerRecord(**row) for row in raw]


def _load_qas_for_answers(run_dir: Path, answers: list[AnswerRecord], dimensions: list[str] | None) -> tuple[list[QAItem], dict[str, dict]]:
    corpus = load_corpus_sessions_dict(run_dir)
    qas = load_masim_qa(run_dir, dimensions=dimensions, corpus_sessions=corpus)
    wanted = {a.question_id for a in answers}
    filtered = [qa for qa in qas if qa.question_id in wanted]
    missing = wanted - {qa.question_id for qa in filtered}
    if missing:
        raise ValueError(f"{len(missing)} answer question_id(s) are missing from MASim QA, e.g. {sorted(missing)[:5]}")
    return filtered, corpus


async def _judge_one(
    *,
    answer_path: Path,
    run_dir: Path,
    judge_tag: str,
    judge_model: str,
    judge_endpoint: str,
    judge_api_key: str,
    output_namespace: str | None,
    dimensions: list[str] | None,
    concurrency: int,
    force: bool,
    hybrid_judge: bool,
    judge_runs: int,
    only_marked: bool = False,
) -> dict[str, Any]:
    answer_namespace = _infer_answer_namespace(answer_path)
    out_namespace = _output_namespace(answer_namespace, judge_tag, output_namespace)
    eval_path = _eval_path_for(answer_path, out_namespace)
    if only_marked:
        return await _judge_marked(
            answer_path=answer_path, eval_path=eval_path, run_dir=run_dir, judge_tag=judge_tag,
            judge_model=judge_model, judge_endpoint=judge_endpoint, judge_api_key=judge_api_key,
            dimensions=dimensions, concurrency=concurrency, namespace=out_namespace,
        )
    if eval_path.exists() and not force and _has_llm_judge(eval_path):
        return {
            "status": "skipped_existing_judged",
            "answer_path": str(answer_path),
            "eval_path": str(eval_path),
            "namespace": out_namespace,
        }

    from openai import OpenAI

    answers = _load_answers(answer_path)
    qas, corpus = _load_qas_for_answers(run_dir, answers, dimensions)
    client = _make_client(OpenAI, judge_api_key, judge_endpoint, concurrency)
    if hybrid_judge:
        payload = await evaluate_answers_hybrid(
            qas=qas,
            answers=answers,
            judge_enabled=True,
            judge_runs=max(1, int(judge_runs)),
            judge_client=client,
            judge_model=judge_model,
            corpus_sessions=corpus,
            concurrency=max(1, int(concurrency)),
        )
    else:
        payload = await evaluate_answers(
            qas=qas,
            answers=answers,
            judge_client=client,
            judge_model=judge_model,
            corpus_sessions=corpus,
            concurrency=max(1, int(concurrency)),
        )
    payload.setdefault("summary", {})
    payload["summary"].update({
        "judge_model": judge_model,
        "judge_endpoint": judge_endpoint,
        "judge_tag": judge_tag,
        "answer_path": str(answer_path),
        "generated_by": "scripts/llmjudge.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    })
    eval_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "judged",
        "answer_path": str(answer_path),
        "eval_path": str(eval_path),
        "namespace": out_namespace,
        "summary": payload.get("summary", {}),
        "reason_counts": _reason_counts(eval_path),
    }


async def _judge_marked(
    *,
    answer_path: Path,
    eval_path: Path,
    run_dir: Path,
    judge_tag: str,
    judge_model: str,
    judge_endpoint: str,
    judge_api_key: str,
    dimensions: list[str] | None,
    concurrency: int,
    namespace: str,
) -> dict[str, Any]:
    """Judge only the records of ``eval_path`` marked ``needs_judge`` and merge them back.

    scripts/grid_helper.py judge-mark sets the mark per paper dimension; every
    other record keeps its verdict. The summary is recomputed from all records.
    """
    from openai import OpenAI
    from scripts.rerun_d6_judge import _refresh_summary

    data = _load_json(eval_path)
    details = data.get("details") or []
    marked = {d["question_id"] for d in details if d.get("needs_judge")}
    if not marked:
        return {"status": "nothing_marked", "answer_path": str(answer_path),
                "eval_path": str(eval_path), "namespace": namespace}
    answers = [a for a in _load_answers(answer_path) if a.question_id in marked]
    qas, corpus = _load_qas_for_answers(run_dir, answers, dimensions)
    client = _make_client(OpenAI, judge_api_key, judge_endpoint, concurrency)
    from eval.src.scoring import JUDGE_FAILED_PREFIX

    payload = await evaluate_answers(
        qas=qas, answers=answers, judge_client=client, judge_model=judge_model,
        corpus_sessions=corpus, concurrency=max(1, int(concurrency)), judge_errors="keep_going",
    )
    fresh = {row["question_id"]: row for row in payload.get("details") or []}
    # a record the judge could not score stays marked: the next run re-judges just it
    for row in fresh.values():
        if str(row.get("reason") or "").startswith(JUDGE_FAILED_PREFIX):
            row["needs_judge"] = True
    missing = marked - set(fresh)
    if missing:
        raise ValueError(f"{len(missing)} marked records were not judged, e.g. {sorted(missing)[:5]}")
    data["details"] = [fresh.get(d["question_id"], d) for d in details]
    _refresh_summary(data)
    data["summary"].update({
        "judge_model": judge_model,
        "judge_tag": judge_tag,
        "summary_refreshed_by": "scripts/llmjudge.py --only-marked",
        "rejudged_marked": len(marked),
        "still_marked": sum(1 for r in fresh.values() if r.get("needs_judge")),
        "rejudged_at": datetime.now().isoformat(timespec="seconds"),
    })
    eval_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "judged_marked", "answer_path": str(answer_path), "eval_path": str(eval_path),
            "namespace": namespace, "rejudged": len(marked), "summary": data["summary"],
            "reason_counts": _reason_counts(eval_path)}


def _jobs_from_progress(
    *,
    progress_json: Path,
    judge_tag: str,
    models: list[str],
    backends: list[str],
    trials: list[str],
    only_missing_judge: bool,
) -> list[Path]:
    if not progress_json.exists():
        raise FileNotFoundError(
            f"missing progress JSON: {progress_json}; run scripts/summarize_eval_progress.py first"
        )
    data = _load_json(progress_json)
    trial_paths = data.get("trial_paths") or {}
    jobs: list[Path] = []
    seen: set[Path] = set()
    for model in [_normalise_model(m) for m in models]:
        for backend in backends:
            for trial in trials:
                key = f"{model}|{backend}|{trial}"
                entries = [e for e in trial_paths.get(key, []) if e.get("status") == "complete"]
                answer_path = None
                for entry in reversed(entries):
                    root = _source_root(entry)
                    ns = entry.get("namespace") or f"{backend}_{model}_{trial}"
                    if root is None:
                        continue
                    answer_path = _find_answer_path(root, backend, trial, ns, judge_tag)
                    if answer_path:
                        break
                if answer_path is None:
                    continue
                out_ns = _output_namespace(_infer_answer_namespace(answer_path), judge_tag, None)
                eval_path = _eval_path_for(answer_path, out_ns)
                if only_missing_judge and _has_llm_judge(eval_path):
                    continue
                if answer_path not in seen:
                    seen.add(answer_path)
                    jobs.append(answer_path)
    return jobs


async def _run_jobs(args: argparse.Namespace) -> int:
    _load_env_file()
    judge_tag, judge_model, judge_endpoint, judge_api_key = _resolve_judge(args)
    if _endpoint_requires_auth(judge_endpoint) and _is_missing_api_key(judge_api_key) and not args.dry_run:
        raise SystemExit(
            "judge API key missing for authenticated judge endpoint; "
            "set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --judge-api-key"
        )
    run_dir = _resolve_repo_path(args.run_dir)
    dimensions = _split_csv(args.dimensions, []) or None

    answer_paths: list[Path] = []
    if args.answer_path:
        answer_paths = [_resolve_repo_path(p) for p in args.answer_path]
    elif args.from_progress_json:
        answer_paths = _jobs_from_progress(
            progress_json=_resolve_repo_path(args.from_progress_json),
            judge_tag=judge_tag,
            models=_split_csv(args.models, DEFAULT_MODELS),
            backends=_split_csv(args.backends, DEFAULT_BACKENDS),
            trials=_split_csv(args.trials, DEFAULT_TRIALS),
            only_missing_judge=bool(args.only_missing_judge),
        )
    elif args.out_dir and args.system and args.trial_name and args.namespace:
        root = _resolve_repo_path(args.out_dir)
        path = _find_answer_path(root, args.system, args.trial_name, args.namespace, judge_tag)
        if path:
            answer_paths = [path]
    else:
        raise SystemExit("pass --answer-path, --from-progress-json, or --out-dir/--system/--trial-name/--namespace")

    print(f"[llmjudge] judge: {judge_model} @ {judge_endpoint} tag={judge_tag} concurrency={args.concurrency}")
    print(f"[llmjudge] jobs: {len(answer_paths)}")
    if args.dry_run:
        for path in answer_paths:
            out_ns = _output_namespace(_infer_answer_namespace(path), judge_tag, args.output_namespace)
            print(f"  {path} -> {_eval_path_for(path, out_ns)}")
        return 0

    results: list[dict[str, Any]] = []
    failures = 0
    for i, answer_path in enumerate(answer_paths, 1):
        print(f"[llmjudge] [{i}/{len(answer_paths)}] {answer_path}")
        try:
            result = await _judge_one(
                answer_path=answer_path,
                run_dir=run_dir,
                judge_tag=judge_tag,
                judge_model=judge_model,
                judge_endpoint=judge_endpoint,
                judge_api_key=judge_api_key,
                output_namespace=args.output_namespace,
                dimensions=dimensions,
                concurrency=max(1, int(args.concurrency)),
                force=bool(args.force),
                hybrid_judge=bool(args.hybrid_judge),
                judge_runs=max(1, int(args.judge_runs)),
                only_marked=bool(args.only_marked),
            )
            print(f"[llmjudge] {result['status']}: {result['eval_path']}")
            results.append(result)
        except Exception as exc:
            failures += 1
            print(f"[llmjudge] ERROR: {answer_path}: {exc}", file=sys.stderr)
            results.append({"status": "failed", "answer_path": str(answer_path), "error": str(exc)})
            if not args.keep_going:
                break

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "judge_model": judge_model,
        "judge_endpoint": judge_endpoint,
        "judge_tag": judge_tag,
        "jobs": len(answer_paths),
        "failures": failures,
        "results": results,
    }
    manifest_path = _resolve_repo_path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[llmjudge] manifest: {manifest_path}")
    return 1 if failures else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "MASim" / "runs" / "5a10d_real")
    parser.add_argument("--answer-path", type=Path, action="append", help="Specific answer_results_*.json file. Can be repeated.")
    parser.add_argument("--from-progress-json", type=Path, help="Use eval_progress_trial_paths.json to find completed answer cells.")
    parser.add_argument("--out-dir", type=Path, help="Accuracy output root for one-cell mode.")
    parser.add_argument("--system", choices=sorted(SYSTEM_DIR), help="Backend for one-cell mode.")
    parser.add_argument("--trial-name", help="Trial for one-cell mode, e.g. s2.")
    parser.add_argument("--namespace", help="Answer namespace for one-cell mode.")
    parser.add_argument("--output-namespace", help="Override output evaluation namespace.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--trials", default=",".join(DEFAULT_TRIALS))
    parser.add_argument("--dimensions", default=None)
    parser.add_argument("--only-missing-judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-preset", default="openrouter", help="openrouter, qwen235b, custom, or local.")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-endpoint", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-tag", default=None,
                        help="Output file tag: evaluation_results_<ns>_judge_<tag>.json. Default: remote "
                             "for the remote/openrouter preset (the gpt-4o-mini files).")
    parser.add_argument("--concurrency", type=int, default=512, help="Judge request concurrency. Default 512 (OpenRouter gpt-4o-mini handles high parallelism well).")
    parser.add_argument("--hybrid-judge", action="store_true")
    parser.add_argument("--judge-runs", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Overwrite existing evaluation_results even if already judged.")
    parser.add_argument("--only-marked", action="store_true",
                        help="Judge only the records of the existing evaluation_results marked needs_judge "
                             "(scripts/grid_helper.py judge-mark) and merge them back; the rest are kept.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifest-out", type=Path, default=REPO_ROOT / "out" / "llmjudge_manifest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return asyncio.run(_run_jobs(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[llmjudge] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
