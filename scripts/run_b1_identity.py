#!/usr/bin/env python3
"""b1_requester_identity ablation — Q3-8B on d4_permission.

Generates three identity-injection variants of the d4_permission queries:
  L0: original (no identity injection)
  L1: prepend "[Asker: <name>] " (just the asker's agent id)
  L2: prepend "[Asker: <name> | Asking about: <ego>] " (asker + target)

For each level, builds an alt run-dir that symlinks the original corpus
files and substitutes the modified d4_permission.jsonl. Then runs the
oracle backend on Qwen3-8B at trial s2, judging via OpenRouter
gpt-4o-mini. Compares answer accuracy + policy metrics across the three
levels to test whether explicit requester identity moves the needle.

The pre-registered wave-1 b1 ablation found Δ = 0.00 ± 0.87 pp (null) — this
script reproduces that test inside MemArena.

Output layout:
  out/ablations_l_b1_{L0,L1,L2}/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/benchmark"
EVAL_INSTANCES_DIR = DATA_DIR / "eval_instances"


def rewrite_query(inst: dict, level: str) -> str:
    asker = (inst.get("asker_agent_id")
             or (inst.get("metadata") or {}).get("query_agent")
             or "someone")
    target = (inst.get("ground_truth") or {}).get("target_agent") or inst.get("ego_agent_id") or "the user"
    q = (inst.get("query") or inst.get("question") or "").strip()
    if level == "L0":
        return q
    if level == "L1":
        return f"[Asker: {asker}] {q}"
    if level == "L2":
        return f"[Asker: {asker} | Asking about: {target}] {q}"
    raise ValueError(f"unknown level: {level}")


def build_alt_run_dir(level: str, alt_root: Path) -> Path:
    """Symlink corpus files + emit modified d4_permission.jsonl. Returns alt run-dir path."""
    alt_dir = alt_root / f"run_b1_{level}"
    alt_dir.mkdir(parents=True, exist_ok=True)

    # Symlink everything from data/benchmark/ except eval_instances (which we override)
    for entry in DATA_DIR.iterdir():
        target = alt_dir / entry.name
        if target.exists() or target.is_symlink():
            continue
        if entry.name == "eval_instances":
            # Build a fresh eval_instances subdir under alt_dir
            inst_dir = alt_dir / "eval_instances"
            inst_dir.mkdir(parents=True, exist_ok=True)
            # symlink all dim files except d4_permission, which we rewrite
            for f in (DATA_DIR / "eval_instances").iterdir():
                tgt = inst_dir / f.name
                if tgt.exists() or tgt.is_symlink():
                    continue
                if f.name == "d4_permission.jsonl":
                    rewritten = []
                    with f.open() as src:
                        for line in src:
                            inst = json.loads(line)
                            new_q = rewrite_query(inst, level)
                            inst["query"] = new_q
                            rewritten.append(inst)
                    with tgt.open("w") as out:
                        for inst in rewritten:
                            out.write(json.dumps(inst, ensure_ascii=False) + "\n")
                    print(f"[b1] {level}: wrote {len(rewritten)} d4 items → {tgt}")
                else:
                    tgt.symlink_to(f.resolve())
        else:
            target.symlink_to(entry.resolve())
    return alt_dir


def run_one_level(level: str, *, alt_run_dir: Path, out_dir: Path, sglang_url: str,
                  api_key: str, model_tag: str, model_name: str, trial: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    namespace = f"oracle_b1_{level}_{model_tag}_{trial}"

    # Run the answer + eval pipeline via eval.cli directly.
    cmd = [
        str(REPO_ROOT / ".venv/bin/python"), "-m", "eval.cli",
        "--run-dir", str(alt_run_dir),
        "--system", "oracle",
        "--output-dir", str(out_dir),
        "--namespace", namespace,
        "--config", str(REPO_ROOT / "eval/config/pipeline.yaml"),
        "--stages", "answer",
        "--model", model_name,
        "--endpoint", f"{sglang_url}/v1",
        "--api-key", "EMPTY",
        "--trial-name", trial,
        "--trial-seed", str({"s2": 1002, "s3": 1003, "s4": 1004}.get(trial, 1002)),
        "--answer-concurrency", "64",
        "--eval-concurrency", "512",
        "--temperature", "0.3",
        "--dimensions", "d4_permission",
        "--force",
        "--log-every", "100",
    ]
    print(f"[b1] running {level} → {out_dir}")
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    if rc != 0:
        return rc

    # Locate answer file
    answer_dir = out_dir / f"eval_results_{trial}" / "oracle"
    answer_paths = list(answer_dir.glob(f"answer_results_oracle_b1_{level}_{model_tag}_{trial}.json"))
    if not answer_paths:
        # Try fallback name
        answer_paths = list(answer_dir.glob(f"answer_results_*_{level}_{model_tag}_{trial}.json"))
    if not answer_paths:
        print(f"[b1] {level}: no answer file found in {answer_dir}")
        return 1

    judge_cmd = [
        str(REPO_ROOT / ".venv/bin/python"), str(REPO_ROOT / "scripts/llmjudge.py"),
        "--run-dir", str(alt_run_dir),
        "--answer-path", str(answer_paths[0]),
        "--judge-preset", "remote",
        "--judge-model", "openai/gpt-4o-mini-2024-07-18",
        "--judge-endpoint", "https://openrouter.ai/api/v1",
        "--judge-api-key", api_key,
        "--concurrency", "512",
        "--force",
    ]
    print(f"[b1] judging {level}")
    return subprocess.run(judge_cmd, cwd=str(REPO_ROOT)).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2"])
    ap.add_argument("--model-tag", default="8b")
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--sglang-url", default="http://localhost:16003")
    ap.add_argument("--trial", default="s2")
    ap.add_argument("--alt-root", type=Path, default=REPO_ROOT / "data/_b1_alt_runs")
    ap.add_argument("--out-prefix", type=Path, default=REPO_ROOT / "out/ablations_l_b1_")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")

    args.alt_root.mkdir(parents=True, exist_ok=True)

    rcs = {}
    for level in args.levels:
        alt_run_dir = build_alt_run_dir(level, args.alt_root)
        out_dir = Path(f"{args.out_prefix}{level}")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        rc = run_one_level(
            level,
            alt_run_dir=alt_run_dir,
            out_dir=out_dir,
            sglang_url=args.sglang_url,
            api_key=api_key,
            model_tag=args.model_tag,
            model_name=args.model_name,
            trial=args.trial,
        )
        rcs[level] = rc
        print(f"[b1] {level} rc={rc}")
    print(f"[b1] summary: {rcs}")
    return 0 if all(v == 0 for v in rcs.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
