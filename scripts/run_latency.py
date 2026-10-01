#!/usr/bin/env python3
"""Latency test — Spark-style on-device timing harness.

Measures answer-time latency for a backend × model cell on whatever
hardware the user has. Reports TTFT, decode throughput, and total
wall-clock per query; optionally also measures memory-cache ingest time.

Two modes
---------
  --dry-run   (default in CI)
      Reader inference is replaced by a deterministic fake-latency
      sampler that draws from `config/backend.yaml : latency.deterministic_fake_ms`
      (TTFT, decode tok/s). No network, no GPU. The pipeline still
      computes mean / p50 / p95 and emits both CSV and JSON, so the
      output schema is exercised.

  (no --dry-run)
      Real timing of sglang inference. The user must have an sglang
      server running (see config/backend.yaml : sglang.url).

Outputs under --out-dir (default ./out/latency/<timestamp>):
  latency_per_query.csv   — one row per query (backend, query_id, TTFT_ms,
                            decode_tok_per_s, total_ms, prompt_tokens,
                            completion_tokens)
  latency_summary.json    — mean / p50 / p95 per metric, overall rows

Usage
-----
  python scripts/run_latency.py --dry-run
  python scripts/run_latency.py --backend vanilla --n 10
  python scripts/run_latency.py --ingest        # also time Memobase cache build
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from memarena.runtime import configure_live_output

DRY_TAG = "[dry-run]"


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def resolve(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur = cfg
    for k in dotted.split("."):
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


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, int(round((p / 100) * (len(ys) - 1)))))
    return ys[k]


# ---------------------------------------------------------------------------
# Timed inference — real or fake.
# ---------------------------------------------------------------------------

def _fake_timing(prompt_tokens: int, completion_tokens: int,
                 fake_cfg: dict, rng: random.Random) -> dict:
    """Return deterministic fake per-query timings for --dry-run."""
    ttft_ms = float(fake_cfg.get("ttft", 50)) + rng.uniform(-5, 5)
    decode_tps = float(fake_cfg.get("decode_tok_per_s", 40)) + rng.uniform(-2, 2)
    decode_ms = 1000.0 * completion_tokens / max(decode_tps, 1)
    return {
        "ttft_ms": round(ttft_ms, 2),
        "decode_tok_per_s": round(decode_tps, 2),
        "total_ms": round(ttft_ms + decode_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "mode": "dry-run",
    }


def _real_timed_inference(prompt: str, *, sglang_cfg: dict,
                          max_tokens: int) -> dict:
    """Query the sglang server and time it end-to-end (no streaming)."""
    import requests
    url = sglang_cfg.get("url", "http://localhost:16000").rstrip("/")
    # Tolerate users passing the OpenAI-style /v1 suffix (e.g. when copying
    # an --sglang-url from an eval.cli invocation); we re-append /v1 below.
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    model = sglang_cfg.get("model_name", "Qwen/Qwen3-0.6B")
    timeout = sglang_cfg.get("timeout_seconds", 120)
    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {}) or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    # Without streaming we cannot isolate TTFT; report total_ms as upper
    # bound and estimate decode throughput from completion_tokens.
    decode_tps = (completion_tokens / (elapsed_ms / 1000.0)
                  if elapsed_ms > 0 else 0.0)
    return {
        "ttft_ms": round(elapsed_ms, 2),  # conservative upper bound (no stream)
        "decode_tok_per_s": round(decode_tps, 2),
        "total_ms": round(elapsed_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "mode": "real",
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    configure_live_output()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "config" / "backend.yaml")
    ap.add_argument("--queries", type=Path,
                    default=REPO_ROOT / "tests" / "fixtures" / "mini_latency_queries.jsonl",
                    help="JSONL file with `prompt` field per row")
    ap.add_argument("--backend", default="vanilla",
                    help="label recorded in output (reader is the same)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=None,
                    help="override warmup_queries from yaml")
    ap.add_argument("--ingest", action="store_true",
                    help="also time one round of memory-cache ingest (stub in dry-run)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("out") / "latency")
    ap.add_argument("--sglang-url", default=None)
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    sglang_cfg = dict(resolve(cfg, "sglang", {}))
    lat_cfg = dict(resolve(cfg, "latency", {}))
    if os.environ.get("MEMARENA_SGLANG_URL"):
        sglang_cfg["url"] = os.environ["MEMARENA_SGLANG_URL"]
    if args.sglang_url: sglang_cfg["url"] = args.sglang_url
    if args.model_name: sglang_cfg["model_name"] = args.model_name
    fake_cfg = lat_cfg.get("deterministic_fake_ms", {})
    warmup = args.warmup if args.warmup is not None else lat_cfg.get("warmup_queries", 3)

    device = detect_device() if not args.dry_run else "skipped(dry-run)"
    rng = random.Random(args.seed)

    # Load queries.
    queries: list[str] = []
    with args.queries.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries.append(row.get("prompt") or row.get("question") or "")
    queries = [q for q in queries if q]
    if not queries:
        raise SystemExit(f"no queries loaded from {args.queries}")
    # Cycle the fixture when N exceeds the number of distinct prompts so
    # callers can request tighter percentile estimates without growing the
    # fixture file.
    if args.n > len(queries):
        queries = [queries[i % len(queries)] for i in range(args.n)]
    else:
        queries = queries[: args.n]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_dir / ts
    out.mkdir(parents=True, exist_ok=True)
    print(f"[latency] dry_run={args.dry_run} backend={args.backend} "
          f"n={len(queries)} warmup={warmup} device={device}")
    print(f"[latency] sglang={sglang_cfg.get('url','<dry>')} "
          f"model={sglang_cfg.get('model_name','<dry>')} out={out}")

    # Warmup (not recorded).
    for i in range(warmup):
        if args.dry_run:
            _fake_timing(16, 16, fake_cfg, rng)
        else:
            _real_timed_inference(queries[i % len(queries)],
                                  sglang_cfg=sglang_cfg, max_tokens=16)

    # Timed pass.
    rows: list[dict] = []
    for i, q in enumerate(queries):
        if args.dry_run:
            stats = _fake_timing(
                prompt_tokens=min(1024, len(q) // 4),
                completion_tokens=64,
                fake_cfg=fake_cfg, rng=rng,
            )
        else:
            stats = _real_timed_inference(
                q, sglang_cfg=sglang_cfg, max_tokens=128,
            )
        rows.append({
            "backend": args.backend,
            "query_id": f"q{i:03d}",
            **stats,
        })

    # Optional ingest timing.
    ingest_ms: float | None = None
    if args.ingest:
        if args.dry_run:
            ingest_ms = 1000.0 * float(fake_cfg.get("ingest_s_per_day", 180))
        else:
            # Placeholder: users adapt this to their real Memobase/MemOS
            # ingest driver. We only time-probe the URL reachability here.
            import requests
            t0 = time.perf_counter()
            try:
                requests.get(resolve(cfg, "memobase.url",
                                     "http://localhost:8019"),
                             timeout=10)
            except Exception:
                pass
            ingest_ms = 1000.0 * (time.perf_counter() - t0)

    # Write per-query CSV.
    csv_path = out / "latency_per_query.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary JSON.
    metric_keys = ["ttft_ms", "decode_tok_per_s", "total_ms",
                    "prompt_tokens", "completion_tokens"]
    summary = {
        "generated_at": ts,
        "dry_run": args.dry_run,
        "backend": args.backend,
        "n": len(rows),
        "warmup": warmup,
        "seed": args.seed,
        "device": device,
        "metrics": {
            k: {
                "mean": round(mean([r[k] for r in rows]), 2),
                "p50":  round(percentile([r[k] for r in rows], 50), 2),
                "p95":  round(percentile([r[k] for r in rows], 95), 2),
            } for k in metric_keys
        },
        "ingest_ms": ingest_ms,
    }
    (out / "latency_summary.json").write_text(json.dumps(summary, indent=2))

    m = summary["metrics"]
    print(f"[latency] ttft  mean={m['ttft_ms']['mean']} p95={m['ttft_ms']['p95']} ms")
    print(f"[latency] decode mean={m['decode_tok_per_s']['mean']} tok/s")
    print(f"[latency] total mean={m['total_ms']['mean']} p95={m['total_ms']['p95']} ms"
          + (f"; ingest={ingest_ms:.0f} ms" if ingest_ms is not None else ""))
    print(f"[latency] wrote {csv_path.name} + latency_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
