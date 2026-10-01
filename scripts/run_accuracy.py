#!/usr/bin/env python3
"""Run MemArena accuracy evaluation after MASim generation.

This script has two modes:

1. Production mode (`--run-dir ...`):
   wraps `python -m eval.cli` so a completed MASim run can be evaluated
   without hand-written commands. It answers with a local/OpenAI-compatible
   reader endpoint, then scores with a remote judge, a local judge, or both.

2. Fixture mode (no `--run-dir`):
   keeps the original tiny JSONL dry-run harness used by CI tests. It has no
   network dependency and writes `accuracy_results.json` plus
   `accuracy_summary.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from memarena.runtime import configure_live_output, flush_standard_streams

DRY_TAG = "[dry-run]"
DEFAULT_SGLANG_URL = "http://localhost:16000"
DEFAULT_READER_MODEL = "qwen3"
OPENAI_ENDPOINT = "https://api.openai.com/v1"
SECRET_FLAGS = {"--api-key", "--judge-api-key", "--secondary-judge-api-key"}


@dataclass(frozen=True)
class JudgePass:
    tag: str
    model: str
    endpoint: str
    api_key_arg: str | None = None
    env: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------

def _normalise_openai_base(url: str) -> str:
    url = str(url or "").strip().rstrip("/")
    if not url:
        raise ValueError("endpoint URL cannot be empty")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def _slug(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("/", "_").replace(":", "_")
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.-")
    return text or "run"


def _redact_cmd(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for part in cmd:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        redacted.append(part)
        if part in SECRET_FLAGS:
            hide_next = True
    return redacted


def _run_and_tee(
    cmd: list[str],
    log_path: Path,
    cwd: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run-accuracy] $ {' '.join(_redact_cmd(cmd))}", flush=True)
    with log_path.open("wb") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            try:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            except AttributeError:
                print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
            log_fh.write(chunk)
            log_fh.flush()
        rc = proc.wait()
    flush_standard_streams()
    return rc


def _add_if_not_none(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _resolve_reader_endpoint(args: argparse.Namespace) -> str:
    raw = args.sglang_url or os.getenv("MEMARENA_SGLANG_URL") or DEFAULT_SGLANG_URL
    return _normalise_openai_base(raw)


def _resolve_reader_model(args: argparse.Namespace) -> str:
    return args.model_name or os.getenv("MEMARENA_MODEL_NAME") or DEFAULT_READER_MODEL


def _openai_key_for_child(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    explicit_judge_key = args.judge_api_key if args.judge_api_key not in {None, "", "EMPTY"} else ""
    key = args.openai_api_key or explicit_judge_key or os.getenv("OPENAI_API_KEY", "")
    env: dict[str, str] = {}
    if args.openai_api_key or explicit_judge_key:
        env["OPENAI_API_KEY"] = key
    return key, env


def _local_judge(args: argparse.Namespace) -> JudgePass:
    endpoint = _normalise_openai_base(
        args.local_judge_url
        or args.judge_sglang_url
        or args.sglang_url
        or DEFAULT_SGLANG_URL
    )
    return JudgePass(
        tag="local",
        model=args.local_judge_model or args.qwen_judge_model or args.model_name or DEFAULT_READER_MODEL,
        endpoint=endpoint,
        api_key_arg=args.local_judge_api_key or "EMPTY",
    )


def _remote_judge(args: argparse.Namespace) -> JudgePass:
    key, env = _openai_key_for_child(args)
    if not key and not args.dry_run and not getattr(args, "test", False):
        raise SystemExit(
            "Remote judge requires OPENAI_API_KEY, --openai-api-key, "
            "or --judge-api-key. For local judging use --judge-preset local."
        )
    return JudgePass(
        tag="remote",
        model=args.openai_judge_model or args.remote_judge_model,
        endpoint=_normalise_openai_base(args.remote_judge_endpoint or OPENAI_ENDPOINT),
        api_key_arg=None,
        env=env,
    )


def _custom_judge(args: argparse.Namespace) -> JudgePass:
    if not args.judge_model or not args.judge_endpoint:
        raise SystemExit("--judge-preset custom requires --judge-model and --judge-endpoint")
    endpoint = _normalise_openai_base(args.judge_endpoint)
    return JudgePass(
        tag=args.judge_tag or _slug(args.judge_model),
        model=args.judge_model,
        endpoint=endpoint,
        api_key_arg=args.judge_api_key,
    )


def _judge_passes(args: argparse.Namespace) -> list[JudgePass]:
    preset = str(args.judge_preset or "remote").lower()
    if preset == "none":
        # D4 (permission-aware access) is now scored by the Arm-B 3-way LLM
        # judge inside `eval/src/scoring.py`. Running the evaluate stage
        # without a judge will raise per-record JudgeScoringError on every
        # D4 question, so fail loud here when the user opted out and the
        # pipeline is going to run evaluate anyway.
        wants_evaluate = (
            args.stages and "evaluate" in args.stages
        ) or not args.stages  # default chain includes evaluate
        if wants_evaluate and not getattr(args, "test", False):
            raise SystemExit(
                "--judge-preset none cannot run the evaluate stage: D4 "
                "permission scoring requires the Arm-B LLM judge. Pass "
                "--judge-preset remote / local / custom, or skip evaluate "
                "with --stages answer."
            )
        return []
    if preset in {"remote", "gpt4omini", "gpt-4o-mini", "4omini"}:
        return [_remote_judge(args)]
    if preset in {"local", "qwen3_8b", "qwen3-8b", "qwen"}:
        return [_local_judge(args)]
    if preset == "both":
        return [_remote_judge(args), _local_judge(args)]
    if preset == "custom":
        return [_custom_judge(args)]
    raise SystemExit(f"unsupported --judge-preset: {args.judge_preset}")


def _secondary_judge(args: argparse.Namespace, primary: JudgePass) -> JudgePass | None:
    preset = str(args.secondary_judge_preset or "none").lower()
    if preset == "none":
        return None
    shadow = argparse.Namespace(**vars(args))
    shadow.judge_preset = preset
    passes = _judge_passes(shadow)
    if not passes:
        return None
    secondary = passes[0]
    if secondary.tag == primary.tag:
        return None
    return secondary


def _namespace_for(args: argparse.Namespace, judge: JudgePass | None) -> str:
    if args.namespace:
        base = _slug(args.namespace)
    else:
        model_tag = args.model_tag or _slug(_resolve_reader_model(args))
        parts = [_slug(args.system), model_tag]
        if args.trial_name:
            parts.append(_slug(str(args.trial_name)))
        base = "_".join(parts)
    if judge is not None:
        base = f"{base}_judge_{judge.tag}"
    return base


def _production_stages(args: argparse.Namespace, judge: JudgePass | None) -> list[str]:
    if args.stages:
        return list(args.stages)
    if args.dry_run:
        return ["answer"]
    system = str(args.system or "").strip().lower()
    if system in {"vanilla", "oracle", "oracle_retrieval", "omniscient"}:
        stages = ["answer"]
    elif system in {"baseline_session"}:
        stages = ["add", "answer"]
    elif system in {"memory_cache"}:
        stages = ["search", "answer"]
    else:
        stages = ["add", "search", "answer"]
    if judge is not None:
        stages.append("evaluate")
    return stages


def _build_eval_cli_cmd(
    args: argparse.Namespace,
    judge: JudgePass | None,
) -> tuple[list[str], dict[str, str], str]:
    if not args.run_dir:
        raise ValueError("--run-dir is required in production mode")

    stages = _production_stages(args, judge)
    namespace = _namespace_for(args, judge)
    cmd = [
        sys.executable,
        "-m",
        "eval.cli",
        "--run-dir",
        str(Path(args.run_dir).expanduser()),
        "--system",
        str(args.system),
        "--output-dir",
        str(args.out_dir),
        "--namespace",
        namespace,
        "--config",
        str(args.eval_config),
        "--stages",
        *stages,
        "--model",
        _resolve_reader_model(args),
        "--endpoint",
        _resolve_reader_endpoint(args),
        "--api-key",
        args.api_key or "EMPTY",
    ]

    if judge is not None:
        cmd.extend(["--judge-model", judge.model, "--judge-endpoint", judge.endpoint])
        if judge.api_key_arg is not None:
            cmd.extend(["--judge-api-key", judge.api_key_arg])

    secondary = _secondary_judge(args, judge) if judge is not None else None
    extra_env: dict[str, str] = {}
    if judge is not None:
        extra_env.update(judge.env)
    if secondary is not None:
        cmd.extend(["--secondary-judge-model", secondary.model, "--secondary-judge-endpoint", secondary.endpoint])
        if secondary.api_key_arg is not None:
            cmd.extend(["--secondary-judge-api-key", secondary.api_key_arg])
        extra_env.update(secondary.env)

    if args.hybrid_judge and "evaluate" in stages:
        cmd.append("--judge-enabled")
        cmd.extend(["--judge-runs", str(max(1, int(args.judge_runs)))])

    if args.trial_name:
        cmd.extend(["--trial-name", str(args.trial_name)])
    if args.seed is not None:
        cmd.extend(["--trial-seed", str(args.seed)])
    if args.dimensions:
        cmd.extend(["--dimensions", *[str(d) for d in args.dimensions]])
    if args.cache_path:
        cmd.extend(["--cache-path", str(args.cache_path)])
    if args.expected_extractor:
        cmd.extend(["--expected-extractor", str(args.expected_extractor)])
    if args.expected_memory_system:
        cmd.extend(["--expected-memory-system", str(args.expected_memory_system)])
    if args.expected_config:
        cmd.extend(["--expected-config", str(args.expected_config)])
    if not args.cache_strict:
        cmd.append("--no-cache-strict")

    _add_if_not_none(cmd, "--qa-limit", args.qa_limit)
    _add_if_not_none(cmd, "--message-limit", args.message_limit)
    _add_if_not_none(cmd, "--user-limit", args.user_limit)
    _add_if_not_none(cmd, "--top-k", args.top_k)
    _add_if_not_none(cmd, "--answer-concurrency", args.answer_concurrency)
    _add_if_not_none(cmd, "--eval-concurrency", args.eval_concurrency)
    _add_if_not_none(cmd, "--context-length", args.context_length)
    _add_if_not_none(cmd, "--max-tokens", args.max_tokens)
    _add_if_not_none(cmd, "--temperature", args.temperature)
    _add_if_not_none(cmd, "--max-previous-context", args.max_previous_context)
    _add_if_not_none(cmd, "--d6-arm", args.d6_arm)

    if args.force:
        cmd.append("--force")
    if not args.resume:
        cmd.append("--no-resume")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.test:
        cmd.append("--test")
    if args.verbose:
        cmd.append("--verbose")
    if args.log_every is not None:
        cmd.extend(["--log-every", str(args.log_every)])
    if not args.progress:
        cmd.append("--no-progress")

    return cmd, extra_env, namespace


def _run_production(args: argparse.Namespace) -> int:
    out_root = Path(args.out_dir).resolve()
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    explicit_evaluate = bool(args.stages and "evaluate" in args.stages)
    inline_judge = bool(args.judge_inline or explicit_evaluate)
    passes = _judge_passes(args) if inline_judge else []
    if not passes:
        passes = [None]  # type: ignore[list-item]

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(Path(args.run_dir).expanduser().resolve()),
        "system": args.system,
        "reader_model": _resolve_reader_model(args),
        "reader_endpoint": _resolve_reader_endpoint(args),
        "judge_preset": args.judge_preset,
        "judge_inline": inline_judge,
        "test": bool(getattr(args, "test", False)),
        "passes": [],
    }

    for judge in passes:
        cmd, env, namespace = _build_eval_cli_cmd(args, judge)
        tag = judge.tag if judge is not None else "answer_only"
        log_path = out_root / "logs" / f"run_accuracy_{namespace}.log"
        rc = _run_and_tee(cmd, log_path, REPO_ROOT, extra_env=env)
        manifest["passes"].append(
            {
                "tag": tag,
                "namespace": namespace,
                "returncode": rc,
                "log_path": str(log_path),
                "command": _redact_cmd(cmd),
            }
        )
        (out_root / "run_accuracy_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if rc != 0:
            print(f"[run-accuracy] pass {tag} failed with exit code {rc}; see {log_path}", flush=True)
            return rc

    print(f"[run-accuracy] completed {len(passes)} pass(es); manifest={out_root / 'run_accuracy_manifest.json'}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Legacy fixture mode — kept for CI and tiny local smoke tests.
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def resolve(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    cur = cfg
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def detect_device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def reader_inference(prompt: str, *, backend_cfg: dict, dry_run: bool) -> str:
    """Return the model's answer to `prompt`."""
    if dry_run:
        return f"{DRY_TAG} answer :: " + prompt[:80].replace("\n", " ") + "..."
    url = backend_cfg.get("url") or DEFAULT_SGLANG_URL
    model = backend_cfg.get("model_name") or "Qwen/Qwen3-0.6B"
    timeout = backend_cfg.get("timeout_seconds", 120)
    import requests
    resp = requests.post(
        f"{_normalise_openai_base(url)}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def judge_score(question: str, gold: str, model_answer: str, *,
                judge_cfg: dict, dry_run: bool, rng: random.Random) -> int:
    """Return 1 if `model_answer` is judged correct against `gold`, else 0."""
    del rng
    if dry_run:
        return 1 if (len(question) + len(model_answer)) % 2 == 0 else 0
    provider = judge_cfg.get("provider", "openai")
    if provider == "openai":
        from openai import OpenAI  # type: ignore
        client = OpenAI()
    else:
        raise SystemExit(f"unsupported judge provider: {provider}")
    prompt = (
        "You are grading a model's answer. Respond with ONLY '1' if the answer "
        "is substantively correct given the gold reference, else '0'.\n\n"
        f"Question: {question}\nGold: {gold}\nAnswer: {model_answer}\n"
    )
    r = client.chat.completions.create(
        model=judge_cfg.get("model_name", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4,
    )
    text = (r.choices[0].message.content or "").strip()
    return 1 if text.startswith("1") else 0


def load_instances(path: Path, n: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if n is not None and len(rows) >= n:
                break
    return rows


def build_prompt_vanilla(instance: dict) -> str:
    return (
        "Answer the following question based on general knowledge.\n"
        f"Question: {instance['question']}\nAnswer:"
    )


def build_prompt_oracle(instance: dict) -> str:
    evidence = instance.get("evidence_text") or instance.get("evidence_session_ids") or []
    ev = evidence if isinstance(evidence, str) else "\n".join(map(str, evidence))
    return (
        f"Context:\n{ev}\n\n"
        f"Question: {instance['question']}\nAnswer:"
    )


BUILDERS = {
    "vanilla": build_prompt_vanilla,
    "oracle": build_prompt_oracle,
}


def _run_fixture(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    sglang_cfg = dict(resolve(cfg, "sglang", {}))
    if os.environ.get("MEMARENA_SGLANG_URL"):
        sglang_cfg["url"] = os.environ["MEMARENA_SGLANG_URL"]
    if args.sglang_url:
        sglang_cfg["url"] = args.sglang_url
    if args.model_name:
        sglang_cfg["model_name"] = args.model_name
    judge_cfg = dict(resolve(cfg, "judge", {}))

    device = detect_device() if not args.dry_run else "skipped(dry-run)"
    rng = random.Random(args.seed)

    n = None if args.full else args.n
    inst = load_instances(args.instances, n)
    if not inst:
        raise SystemExit(f"no instances loaded from {args.instances}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.trial_name:
        trial = str(args.trial_name).strip()
        if not trial:
            raise SystemExit("--trial-name must be non-empty (e.g. s1, s2, s3)")
        out = args.out_dir / f"eval_results_{trial}" / args.backend
    else:
        out = args.out_dir / ts
    out.mkdir(parents=True, exist_ok=True)

    print(f"[accuracy] fixture_mode=1 dry_run={args.dry_run} backend={args.backend} "
          f"n={len(inst)} seed={args.seed} device={device}")
    print(f"[accuracy] sglang={sglang_cfg.get('url', '<dry>')} "
          f"model={sglang_cfg.get('model_name', '<dry>')} out={out}")

    t0 = time.time()
    rows: list[dict] = []
    build = BUILDERS[args.backend]
    for i, item in enumerate(inst, 1):
        prompt = build(item)
        ans = reader_inference(prompt, backend_cfg=sglang_cfg, dry_run=args.dry_run)
        score = judge_score(
            item["question"], item.get("gold_answer", ""),
            ans, judge_cfg=judge_cfg, dry_run=args.dry_run, rng=rng,
        )
        rows.append({
            "instance_id": item.get("instance_id", f"i{i}"),
            "dimension": item.get("dimension", "unknown"),
            "question": item["question"],
            "gold": item.get("gold_answer", ""),
            "model_answer": ans,
            "score": score,
        })
        if i % max(1, len(inst) // 10) == 0:
            print(f"  [{i:>4d}/{len(inst)}] score_running={sum(r['score'] for r in rows)/i:.3f}")

    from collections import defaultdict
    by_dim = defaultdict(list)
    for r in rows:
        by_dim[r["dimension"]].append(r["score"])
    dim_acc = {d: sum(xs) / len(xs) for d, xs in by_dim.items() if xs}
    overall = sum(r["score"] for r in rows) / len(rows)

    (out / "accuracy_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "accuracy_summary.json").write_text(json.dumps({
        "generated_at": ts,
        "dry_run": args.dry_run,
        "backend": args.backend,
        "trial_name": args.trial_name,
        "n": len(rows),
        "seed": args.seed,
        "device": device,
        "overall_accuracy": overall,
        "per_dimension_accuracy": dim_acc,
        "elapsed_seconds": round(time.time() - t0, 2),
    }, indent=2), encoding="utf-8")
    print(f"[accuracy] overall={overall:.3f} per_dim={dim_acc} "
          f"· elapsed={time.time() - t0:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)

    # Production pipeline.
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Completed MASim run directory. Enables production eval.cli mode.")
    ap.add_argument("--system", default="oracle",
                    help="eval.cli system/backend, e.g. oracle, vanilla, inmem, baseline_simplerag")
    ap.add_argument("--stages", nargs="+", choices=["add", "search", "answer", "evaluate"], default=None,
                    help=(
                        "eval.cli stages. Default: answer-only. Pass evaluate explicitly, "
                        "or use --judge-inline, to run LLM judging in this command."
                    ))
    ap.add_argument("--namespace", default=None,
                    help="Stable eval namespace. Default includes system, reader model, and judge tag.")
    ap.add_argument("--model-tag", default=None,
                    help="Short label used in the auto namespace for the reader model.")
    ap.add_argument("--cache-path", default=None,
                    help="Path to memory-cache JSONL when --system memory_cache.")
    ap.add_argument("--expected-extractor", default=None,
                    help="Assert memory-cache extractor_model.")
    ap.add_argument("--expected-memory-system", default=None,
                    help="Assert memory-cache source system, e.g. memobase or memos.")
    ap.add_argument("--expected-config", default=None,
                    help="Assert memory-cache config, e.g. A_paired.")
    ap.add_argument("--cache-strict", action=argparse.BooleanOptionalAction, default=True,
                    help="Raise on memory-cache misses. Default: true.")
    ap.add_argument("--eval-config", type=Path, default=REPO_ROOT / "eval" / "config" / "pipeline.yaml",
                    help="eval.cli YAML config.")
    ap.add_argument("--api-key", default="EMPTY",
                    help="Reader API key. Local sglang usually uses EMPTY.")
    ap.add_argument("--judge-preset", default="remote", metavar="{remote,local,both,custom,none}",
                    help=(
                        "Primary judge placement. Use remote for a hosted judge, local for an "
                        "OpenAI-compatible local endpoint, or both for separate remote+local passes."
                    ))
    ap.add_argument("--openai-api-key", default=None,
                    help="OpenAI key for the default remote judge. Prefer OPENAI_API_KEY env for saved artifacts.")
    ap.add_argument("--remote-judge-model", default="gpt-4o-mini",
                    help="Model name for --judge-preset remote.")
    ap.add_argument("--remote-judge-endpoint", default=OPENAI_ENDPOINT,
                    help="OpenAI-compatible endpoint for --judge-preset remote.")
    ap.add_argument("--openai-judge-model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--local-judge-url", default=None,
                    help="OpenAI-compatible endpoint for --judge-preset local. Defaults to --sglang-url.")
    ap.add_argument("--judge-sglang-url", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--local-judge-model", default=None,
                    help="Model name for --judge-preset local. Defaults to --model-name or qwen3.")
    ap.add_argument("--qwen-judge-model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--judge-model", default=None,
                    help="Custom judge model for --judge-preset custom.")
    ap.add_argument("--judge-endpoint", default=None,
                    help="Custom judge endpoint for --judge-preset custom.")
    ap.add_argument("--judge-api-key", default=None,
                    help="API key for remote/custom judges. Local judges use --local-judge-api-key.")
    ap.add_argument("--local-judge-api-key", default="EMPTY",
                    help="API key for local OpenAI-compatible judge endpoints. Defaults to EMPTY.")
    ap.add_argument("--judge-tag", default=None,
                    help="Namespace tag for --judge-preset custom.")
    ap.add_argument("--secondary-judge-preset", default="none", metavar="{none,remote,local,custom}",
                    help="Optional eval.cli secondary judge in the same pass.")
    ap.add_argument("--hybrid-judge", action="store_true",
                    help="Use eval.cli's hybrid/voting judge path instead of the default judge scorer.")
    ap.add_argument("--judge-runs", type=int, default=3,
                    help="Voting runs when --hybrid-judge is enabled.")
    ap.add_argument("--judge-inline", action="store_true",
                    help=(
                        "Legacy one-shot mode: run LLM-as-judge inside run_accuracy.py. "
                        "Default is answer-only; use scripts/llmjudge.py to judge existing answers."
                    ))

    # Shared reader/eval controls.
    ap.add_argument("--sglang-url", default=None,
                    help="OpenAI-compatible reader endpoint, with or without /v1.")
    ap.add_argument("--model-name", default=None,
                    help="Reader model name exposed by the sglang server.")
    ap.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path,
                    default=Path("out") / "accuracy")
    ap.add_argument("--trial-name", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--qa-limit", type=int, default=None)
    ap.add_argument("--message-limit", type=int, default=None)
    ap.add_argument("--user-limit", type=int, default=None)
    ap.add_argument("--dimensions", nargs="+", default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--answer-concurrency", type=int, default=None)
    ap.add_argument("--eval-concurrency", type=int, default=None)
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-previous-context", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--d6-arm", default=None)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true",
                    help="Fixture mode: mock reader + judge. Production mode: pass --dry-run to eval.cli.")
    ap.add_argument("--test", action="store_true",
                    help="Production smoke mode: preserve outputs but replace LLM/OpenClaw calls with \"I don't know\".")

    # Legacy fixture mode.
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "config" / "backend.yaml",
                    help="Legacy fixture-mode backend config.")
    ap.add_argument("--instances", type=Path,
                    default=REPO_ROOT / "tests" / "fixtures" / "mini_eval_instances.jsonl",
                    help="Legacy fixture-mode JSONL eval instances.")
    ap.add_argument("--backend", choices=list(BUILDERS), default="vanilla",
                    help="Legacy fixture-mode backend.")
    ap.add_argument("--n", type=int, default=10,
                    help="Legacy fixture-mode max instances.")
    ap.add_argument("--full", action="store_true",
                    help="Legacy fixture-mode: disable --n cap.")
    return ap


def main(argv: list[str] | None = None) -> int:
    configure_live_output()
    args = build_arg_parser().parse_args(argv)
    if args.run_dir:
        return _run_production(args)
    return _run_fixture(args)


if __name__ == "__main__":
    raise SystemExit(main())
