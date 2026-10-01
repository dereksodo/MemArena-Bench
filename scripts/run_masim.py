#!/usr/bin/env python3
"""Run MASim end to end against an OpenAI-compatible SGLang endpoint.

This is the production launcher for data generation.  It takes a MASim YAML
configuration, overwrites its LLM endpoint with ``--sglang-url``, waits for the
server to become usable, runs ``python -m MASim run``, and then runs validation
and statistics collection without requiring manual intervention.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from memarena.runtime import configure_live_output, flush_standard_streams

DEFAULT_CONFIG = Path("MASim") / "configs" / "memarena_l.yaml"
DEFAULT_SGLANG_URL = "http://localhost:16000"
DEFAULT_RUNS_DIR = Path("MASim") / "runs"
DRY_TAG = "[dry-run]"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")

    base_name = data.pop("base", None)
    if not base_name:
        return data

    base_path = (path.parent / str(base_name)).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Config base file not found: {base_path}")
    return _deep_merge(_load_yaml_with_base(base_path), data)


def _normalise_sglang_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("--sglang-url cannot be empty")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def _slug(value: str) -> str:
    text = str(value or "").strip().lower()
    text = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_.-") or "run"


def _default_output_dir(config_path: Path, timestamp: str) -> Path:
    return (REPO_ROOT / DEFAULT_RUNS_DIR / f"{timestamp}_{_slug(config_path.stem)}").resolve()


def _wait_for_models(api_base: str, timeout_s: int, poll_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error = ""
    models_url = f"{api_base}/models"

    while time.time() < deadline:
        try:
            resp = requests.get(models_url, timeout=5)
            if resp.ok:
                payload = resp.json()
                if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                    return payload
                last_error = f"unexpected /models payload: {payload!r}"
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as exc:  # noqa: BLE001 - report endpoint readiness failures
            last_error = str(exc)
        time.sleep(poll_s)

    raise TimeoutError(
        f"SGLang endpoint did not become ready within {timeout_s}s: "
        f"{models_url} ({last_error})"
    )


def _smoke_chat(api_base: str, model: str, timeout_s: int) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with one word: ready"}],
        "temperature": 0,
        "max_tokens": 4,
    }
    resp = requests.post(f"{api_base}/chat/completions", json=payload, timeout=timeout_s)
    if not resp.ok:
        raise RuntimeError(
            f"SGLang chat smoke test failed for model {model!r}: "
            f"HTTP {resp.status_code}: {resp.text[:500]}"
        )


def _choose_model(config: dict[str, Any], models_payload: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    llm = config.setdefault("llm", {})
    if isinstance(llm, dict) and llm.get("model"):
        return str(llm["model"])
    models = models_payload.get("data") or []
    if models and isinstance(models[0], dict) and models[0].get("id"):
        return str(models[0]["id"])
    raise ValueError("No model specified in config.llm.model and no model id returned by /models")


def _prepare_output_dir(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output}. "
                "Pass --overwrite or choose a new --output."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)


def _run_and_tee(cmd: list[str], log_path: Path, cwd: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"[run-masim] $ {' '.join(cmd)}", flush=True)
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _smoke_call_llm(prompt: str, *, dry_run: bool, stage: str) -> str:
    if dry_run:
        return f"{DRY_TAG} {stage} :: " + prompt[:60].replace("\n", " ") + "..."
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"openai package not installed: {exc}")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set; omit --smoke-real or export a key")
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


def _smoke_personas(agents: int, dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(agents):
        desc = _smoke_call_llm(
            f"Write a 1-sentence persona for agent {idx}.",
            dry_run=dry_run,
            stage="persona",
        )
        rows.append({
            "agent_id": f"agent_{idx:03d}",
            "name": f"Smoke Agent {idx}",
            "persona": desc,
            "interests": ["memory", "benchmarks"],
        })
    return rows


def _smoke_schedules(personas: list[dict[str, Any]], days: int, dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for persona in personas:
        for day in range(days):
            plan = _smoke_call_llm(
                f"Write a one-line schedule for {persona['name']} on day {day}.",
                dry_run=dry_run,
                stage="schedule",
            )
            rows.append({
                "agent_id": persona["agent_id"],
                "day": day,
                "activities": [{"start": "09:00", "end": "10:00", "note": plan}],
            })
    return rows


def _smoke_dialogues(personas: list[dict[str, Any]], days: int, dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in range(days):
        for idx, persona in enumerate(personas):
            partner = personas[(idx + 1) % len(personas)]
            turn = _smoke_call_llm(
                f"{persona['name']} greets {partner['name']} in one short line.",
                dry_run=dry_run,
                stage="dialogue",
            )
            rows.append({
                "session_id": f"smoke_d{day}_s{idx:03d}",
                "day": day,
                "participants": [persona["agent_id"], partner["agent_id"]],
                "turns": [{"speaker": persona["agent_id"], "text": turn}],
            })
    return rows


def _smoke_eval_instances(dialogues: list[dict[str, Any]], dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in dialogues[:max(1, len(dialogues) // 2)]:
        turn_text = session["turns"][0]["text"] if session["turns"] else ""
        question = _smoke_call_llm(
            f"Write a yes/no factual question about: {turn_text[:100]}",
            dry_run=dry_run,
            stage="gt_qa",
        )
        rows.append({
            "instance_id": f"d7_qa::{session['session_id']}",
            "dimension": "d7_qa",
            "ego": session["participants"][0],
            "question": question,
            "gold_answer": "yes",
            "evidence_session_ids": [session["session_id"]],
        })
    return rows


def _validate_rows(rows: list[dict[str, Any]], required_keys: set[str], label: str) -> None:
    for idx, row in enumerate(rows):
        missing = required_keys - set(row.keys())
        if missing:
            raise SystemExit(f"[schema] {label} row {idx} missing required keys: {missing}")


def _run_smoke(args: argparse.Namespace, output_dir: Path, timestamp: str) -> int:
    dry_run = bool(args.dry_run or not args.smoke_real)
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    print(f"[run-masim] smoke=1 dry_run={dry_run} agents={args.agents} days={args.days}", flush=True)
    print(f"[run-masim] output={output_dir}", flush=True)
    t0 = time.time()

    personas = _smoke_personas(args.agents, dry_run)
    _validate_rows(personas, {"agent_id", "name", "persona"}, "persona")
    _write_jsonl(output_dir / "agents_personas.jsonl", personas)

    schedules = _smoke_schedules(personas, args.days, dry_run)
    _validate_rows(schedules, {"agent_id", "day", "activities"}, "schedule")
    _write_jsonl(output_dir / "agent_schedules.jsonl", schedules)

    dialogues = _smoke_dialogues(personas, args.days, dry_run)
    _validate_rows(dialogues, {"session_id", "day", "participants", "turns"}, "dialogue")
    _write_jsonl(output_dir / "corpus_sessions.jsonl", dialogues)

    evals = _smoke_eval_instances(dialogues, dry_run)
    _validate_rows(evals, {"instance_id", "dimension", "ego", "question", "gold_answer"}, "eval")
    _write_jsonl(output_dir / "eval_instances" / "d7_qa.jsonl", evals)

    summary = {
        "generated_at": timestamp,
        "smoke": True,
        "dry_run": dry_run,
        "counts": {
            "personas": len(personas),
            "schedules": len(schedules),
            "sessions": len(dialogues),
            "eval_instances": len(evals),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    _write_json(output_dir / "pipeline_report.json", summary)
    print(f"[run-masim] smoke ok counts={summary['counts']} elapsed={summary['elapsed_seconds']}s", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="MASim YAML config file",
    )
    parser.add_argument(
        "--sglang-url",
        default=DEFAULT_SGLANG_URL,
        help=(
            "Base URL for the SGLang OpenAI-compatible server, with or "
            "without /v1 (default: http://localhost:16000)"
        ),
    )
    parser.add_argument(
        "--output",
        "--out-dir",
        type=Path,
        default=None,
        help="MASim run directory. Default: MASim/runs/<timestamp>_<config-stem>",
    )
    parser.add_argument("--smoke", action="store_true", help="Run the toy MASim smoke pipeline instead of full MASim")
    parser.add_argument("--smoke-real", action="store_true", help="In --smoke mode, call OpenAI instead of deterministic stubs")
    parser.add_argument("--agents", type=int, default=2, help="--smoke only: number of toy agents")
    parser.add_argument("--days", type=int, default=1, help="--smoke only: number of toy days")
    parser.add_argument("--model", default=None, help="Override config.llm.model")
    parser.add_argument("--api-key", default="EMPTY", help="Override config.llm.api_key")
    parser.add_argument("--concurrency", type=int, default=None, help="Override config.llm.concurrency")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override config.llm.max_tokens")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Override config.llm.timeout_seconds")
    parser.add_argument("--wait-timeout", type=int, default=900, help="Seconds to wait for SGLang readiness")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="SGLang readiness poll interval")
    parser.add_argument("--skip-smoke-chat", action="store_true", help="Only check /models, not chat completions")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to MASim")
    parser.add_argument("--faster-batch", action="store_true", help="Pass --faster-batch to MASim")
    parser.add_argument("--stop-after", default=None, help="Pass --stop-after STAGE to MASim")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty output dir")
    parser.add_argument("--skip-validate", action="store_true", help="Do not run MASim validate after generation")
    parser.add_argument("--skip-stats", action="store_true", help="Do not run MASim stats after generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_live_output()
    args = build_arg_parser().parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.smoke:
        output_dir = _resolve_repo_path(args.output) if args.output else _default_output_dir(Path("smoke"), timestamp)
        return _run_smoke(args, output_dir, timestamp)

    config_path = _resolve_repo_path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    output_dir = (_resolve_repo_path(args.output) if args.output else _default_output_dir(config_path, timestamp))
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    effective_config = _load_yaml_with_base(config_path)
    api_base = _normalise_sglang_url(args.sglang_url)

    models_payload: dict[str, Any] = {"data": []}
    if args.dry_run:
        print("[run-masim] dry-run enabled; skipping SGLang readiness checks")
    else:
        print(f"[run-masim] waiting for SGLang at {api_base}/models", flush=True)
        models_payload = _wait_for_models(api_base, args.wait_timeout, args.poll_interval)

    model = _choose_model(effective_config, models_payload, args.model)
    llm_cfg = effective_config.setdefault("llm", {})
    if not isinstance(llm_cfg, dict):
        raise SystemExit("config.llm must be a mapping")
    llm_cfg["endpoint"] = api_base
    llm_cfg["api_key"] = args.api_key
    llm_cfg["model"] = model
    if args.concurrency is not None:
        llm_cfg["concurrency"] = args.concurrency
    if args.max_tokens is not None:
        llm_cfg["max_tokens"] = args.max_tokens
    if args.timeout_seconds is not None:
        llm_cfg["timeout_seconds"] = args.timeout_seconds

    if not args.dry_run and not args.skip_smoke_chat:
        print(f"[run-masim] smoke-testing chat completions with model={model}", flush=True)
        _smoke_chat(api_base, model, timeout_s=max(30, int(llm_cfg.get("timeout_seconds", 120))))

    effective_path = output_dir / "effective_config.yaml"
    effective_path.write_text(yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8")
    _write_json(
        output_dir / "run_masim_metadata.json",
        {
            "created_at": timestamp,
            "source_config": str(config_path),
            "effective_config": str(effective_path),
            "sglang_url": api_base,
            "model": model,
            "dry_run": args.dry_run,
        },
    )

    run_dir = output_dir
    cmd = [
        sys.executable,
        "-m",
        "MASim",
        "run",
        "--config",
        str(effective_path),
        "--output",
        str(run_dir),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.faster_batch:
        cmd.append("--faster-batch")
    if args.stop_after:
        cmd.extend(["--stop-after", str(args.stop_after)])

    rc = _run_and_tee(cmd, output_dir / "logs" / "masim.log", REPO_ROOT)
    if rc != 0:
        print(f"[run-masim] MASim failed with exit code {rc}; see {output_dir / 'logs' / 'masim.log'}")
        return rc

    if not args.skip_validate and not args.stop_after:
        rc = _run_and_tee(
            [sys.executable, "-m", "MASim", "validate", "--run", str(run_dir)],
            output_dir / "logs" / "validate.log",
            REPO_ROOT,
        )
        if rc != 0:
            print(f"[run-masim] validation failed; see {output_dir / 'logs' / 'validate.log'}")
            return rc

    if not args.skip_stats:
        rc = _run_and_tee(
            [sys.executable, "-m", "MASim", "stats", "--run", str(run_dir)],
            output_dir / "logs" / "stats.log",
            REPO_ROOT,
        )
        if rc != 0:
            print(f"[run-masim] stats failed; see {output_dir / 'logs' / 'stats.log'}")
            return rc

    print(f"[run-masim] completed: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
