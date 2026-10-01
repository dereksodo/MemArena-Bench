from __future__ import annotations

import argparse
import atexit
import asyncio
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
import yaml

from memarena.runtime import configure_live_output
from eval.src.adapters import build_adapter
from eval.src.answering import AnswerConfig
from eval.src.pipeline import EvalPipeline


class _TemporaryOpenClawSoulOverride:
    """Temporarily replace OpenClaw workspace SOUL.md and reliably restore it."""

    def __init__(self, *, source_soul: Path, target_soul: Path) -> None:
        self.source_soul = source_soul
        self.target_soul = target_soul
        self._armed = False
        self._had_original = False
        self._original_text: Optional[str] = None
        self._prev_sigint = None
        self._prev_sigterm = None

    def activate(self) -> None:
        if self._armed:
            return
        src = self.source_soul
        if not src.exists():
            raise FileNotFoundError(f"soul source not found: {src}")

        self.target_soul.parent.mkdir(parents=True, exist_ok=True)
        if self.target_soul.exists():
            self._had_original = True
            self._original_text = self.target_soul.read_text(encoding="utf-8")

        self.target_soul.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        self._armed = True

        atexit.register(self.restore)
        self._prev_sigint = signal.getsignal(signal.SIGINT)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_interrupt(signum, _frame):
            self.restore()
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handle_interrupt)
        signal.signal(signal.SIGTERM, _handle_interrupt)

    def restore(self) -> None:
        if not self._armed:
            return

        try:
            if self._had_original:
                self.target_soul.write_text(self._original_text or "", encoding="utf-8")
            else:
                if self.target_soul.exists():
                    self.target_soul.unlink()
        finally:
            try:
                if self._prev_sigint is not None:
                    signal.signal(signal.SIGINT, self._prev_sigint)
                if self._prev_sigterm is not None:
                    signal.signal(signal.SIGTERM, self._prev_sigterm)
            except Exception:
                pass
            self._armed = False


class _TemporaryOpenClawModelOverride:
    """Temporarily switch OpenClaw default model; restore on exit/interrupt."""

    def __init__(self, *, model_ref: str, openclaw_bin: str = "openclaw") -> None:
        self.model_ref = model_ref.strip()
        self.openclaw_bin = openclaw_bin
        self._armed = False
        self._prev_default: Optional[str] = None

    def _run(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, timeout=30)

    def _status_default(self) -> Optional[str]:
        proc = self._run([self.openclaw_bin, "models", "status", "--json"])
        if proc.returncode != 0:
            return None
        try:
            obj = json.loads(proc.stdout or "{}")
            v = obj.get("defaultModel")
            return str(v).strip() if v else None
        except Exception:
            return None

    def activate(self) -> None:
        if self._armed or not self.model_ref:
            return
        self._prev_default = self._status_default()

        proc = self._run([self.openclaw_bin, "models", "set", self.model_ref])
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"openclaw models set failed: {msg}")

        self._armed = True
        atexit.register(self.restore)

    def restore(self) -> None:
        if not self._armed:
            return
        try:
            if self._prev_default and self._prev_default != self.model_ref:
                self._run([self.openclaw_bin, "models", "set", self._prev_default])
        finally:
            self._armed = False


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _expand_env(value: Optional[str]) -> Optional[str]:
    """Expand ${VAR} or $VAR tokens in a string using os.environ."""
    if not value:
        return value
    import re
    def _sub(m):
        return os.getenv(m.group(1) or m.group(2), m.group(0))
    return re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", _sub, value)


def _resolve_judge_endpoint_and_key(
    explicit_endpoint: Optional[str],
    explicit_key: Optional[str],
) -> tuple[str, str]:
    """Resolve judge endpoint + API key with auto-detection.

    If both explicit values are provided, they win. Otherwise pick OpenAI
    when OPENAI_API_KEY is set, OpenRouter when OPENROUTER_API_KEY is set.
    """
    if explicit_endpoint and explicit_key:
        return explicit_endpoint, explicit_key
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if explicit_endpoint:
        if "openai.com" in explicit_endpoint:
            return explicit_endpoint, explicit_key or openai_key or openrouter_key or ""
        if "openrouter.ai" in explicit_endpoint:
            return explicit_endpoint, explicit_key or openrouter_key or ""
        return explicit_endpoint, explicit_key or ""
    if openai_key:
        return "https://api.openai.com/v1", openai_key
    if openrouter_key:
        return "https://openrouter.ai/api/v1", openrouter_key
    return "", ""


def _answer_cfg_from_yaml(doc: Dict[str, Any]) -> AnswerConfig:
    a = doc.get("answer") if isinstance(doc.get("answer"), dict) else {}
    raw_key = str(a.get("api_key") or "")
    raw_key = _expand_env(raw_key) or ""
    # Fall back to common env vars when YAML doesn't specify a key.
    api_key = raw_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY") or "EMPTY"
    raw_endpoint = _expand_env(str(a["endpoint"]) if a.get("endpoint") is not None else None)

    j = doc.get("judge") if isinstance(doc.get("judge"), dict) else {}
    j_model = str(j.get("model") or "gpt-4o-mini-2024-07-18")
    j_raw_endpoint = _expand_env(str(j["endpoint"]) if j.get("endpoint") is not None else None)
    j_raw_key = _expand_env(str(j["api_key"]) if j.get("api_key") is not None else None)
    j_endpoint, j_api_key = _resolve_judge_endpoint_and_key(j_raw_endpoint, j_raw_key)
    if j_endpoint and "openrouter.ai" in j_endpoint and "/" not in j_model:
        j_model = f"openai/{j_model}"

    return AnswerConfig(
        model=str(a.get("model", "Qwen3-Coder")),
        endpoint=(raw_endpoint if raw_endpoint else "http://127.0.0.1:8000/v1"),
        api_key=api_key,
        temperature=float(a.get("temperature", 0.0)),
        max_tokens=int(a.get("max_tokens", 400)),
        timeout_seconds=int(a.get("timeout_seconds", 300)),
        concurrency=max(1, int(a.get("concurrency", 32))),
        max_retries=max(1, int(a.get("max_retries", 5))),
        retry_base_delay=float(a.get("retry_base_delay", 1.0)),
        retry_max_delay=float(a.get("retry_max_delay", 60.0)),
        provider_order=(list(a.get("provider_order")) if isinstance(a.get("provider_order"), list) else None),
        allow_fallbacks=bool(a.get("allow_fallbacks", False)),
        cache_stats=bool(a.get("cache_stats", True)),
        judge_model=j_model,
        judge_endpoint=j_endpoint or None,
        judge_api_key=j_api_key or None,
    )


def _validate_env_for_system(system: str) -> None:
    s = (system or "").strip().lower()
    required = {
        "memos": ["MEMOS_BASE_URL", "MEMOS_API_KEY"],
        "mem0": ["MEM0_API_KEY"],
        # mem0sdk: no env vars required (uses local SGLang + local embedder)
        "memobase": ["MEMOBASE_BASE_URL", "MEMOBASE_API_TOKEN"],
        "zep": ["ZEP_API_KEY"],
    }
    if s not in required:
        return
    missing = [k for k in required[s] if not os.getenv(k)]
    if missing:
        raise ValueError(f"missing env for {s}: {', '.join(missing)}")


def _resolve_namespace(args: argparse.Namespace) -> str:
    run_id = str(getattr(args, "run_id", "") or "").strip()
    if run_id:
        return run_id
    return str(args.namespace)


_SECRET_ARG_SUFFIXES = ("api_key", "_token", "secret", "password")


def _redact_args(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Blank out credential-bearing args before they are written to disk."""
    redacted: Dict[str, Any] = {}
    for key, value in payload.items():
        if value and key.lower().endswith(_SECRET_ARG_SUFFIXES):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def _archive_run_artifacts(output_root: Path, namespace: str, args: argparse.Namespace, report: Dict[str, Any]) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / "runs" / f"{namespace}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy stage artifacts for this namespace
    patterns = [f"*_{namespace}.json", f"run_meta_{namespace}.json"]
    copied: List[str] = []
    for pat in patterns:
        for p in output_root.glob(pat):
            if not p.is_file():
                continue
            dst = run_dir / p.name
            shutil.copy2(p, dst)
            copied.append(p.name)

    args_payload = _redact_args(vars(args))
    (run_dir / "args.json").write_text(json.dumps(args_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "namespace": namespace,
                "system": str(getattr(args, "system", "")),
                "stages": list(getattr(args, "stages", []) or []),
                "copied_files": copied,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    latest = output_root / "runs" / f"latest_{namespace}.txt"
    latest.write_text(str(run_dir), encoding="utf-8")
    return run_dir


def _write_run_card(path: Path, report: Dict[str, Any], args: argparse.Namespace, archive_dir: Path, namespace: str) -> None:
    ev = report.get("evaluate") if isinstance(report.get("evaluate"), dict) else {}
    lines = [
        f"# Run Card - {getattr(args, 'system', '-')}",
        "",
        f"- namespace: `{namespace}`",
        f"- stages: `{', '.join(getattr(args, 'stages', []) or [])}`",
        f"- archive_dir: `{archive_dir}`",
        "",
        "## Metrics",
        f"- accuracy: {float(ev.get('accuracy', 0.0)):.4f}",
        f"- policy_accuracy: {float(ev.get('policy_accuracy', 0.0)):.4f}",
        f"- privacy_leakage_rate: {float(ev.get('privacy_leakage_rate', 0.0)):.4f}",
        f"- conflict_robustness_score: {float(ev.get('conflict_robustness_score', 0.0)):.4f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MemArena clean-room evaluation CLI")
    ap.add_argument("--messages", type=str, default=None, help="Path to transcript JSONL (e.g., transcript_final.jsonl)")
    ap.add_argument("--qa", type=str, default=None, help="Path to QA JSON")
    ap.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Path to a MASim run directory (e.g., MASim/runs/l_20260408_111046/). "
            "Auto-derives --messages and --qa from corpus_sessions.jsonl and eval_instances/. "
            "Converted files are cached in --output-dir for resumability."
        ),
    )
    ap.add_argument(
        "--dimensions",
        nargs="+",
        default=None,
        metavar="DIM",
        help=(
            "Restrict QA loading to specific MASim dimensions, e.g. "
            "--dimensions d4_permission d7_qa d11_exception. "
            "Default: all available dimensions."
        ),
    )
    ap.add_argument(
        "--system",
        type=str,
        choices=[
            "vanilla",
            "oracle",
            "oracle_retrieval",
            "oracle_with_distractors",
            "oracle_with_random_distractors",
            "omniscient",
            "inmem",
            "temporal",
            "llm",
            "baseline_simplerag",
            "baseline_session",
            "openclaw",
            "memos",
            "mem0",
            "mem0sdk",
            "memobase",
            "zep",
            "memory_cache",
            "dense_bge_m3",
            "dense_e5",
            "hybrid_bm25rerank",
            "hybrid_rrf",
            "inmem_text_sessions",
            "inmem_provenance",
            "oracle_gated",
        ],
        default="inmem",
        help=(
            "vanilla=recent ego-context (direct eval); "
            "oracle=evidence-session context (direct eval); "
            "inmem=lexical retriever (BM25); "
            "llm=oracle retrieval (legacy, use 'oracle' instead); "
            "baseline_simplerag=BM25 inmem retriever + LLM answerer; "
            "baseline_session=OpenClaw session-memory only (no retrieval); "
            "openclaw is kept as alias of baseline_simplerag; "
            "memos/mem0/memobase/zep=external memory backends; "
            "memory_cache=replay a frozen JSONL cache from scripts/build_memory_cache.py"
        ),
    )
    ap.add_argument("--cache-path", type=str, default=None,
                    help="Path to a memory-cache JSONL file (required when --system memory_cache)")
    ap.add_argument("--expected-extractor", type=str, default=None,
                    help="memory_cache: assert every cache row has this extractor_model (e.g. 'Qwen3-8B-AWQ')")
    ap.add_argument("--expected-memory-system", type=str, default=None,
                    choices=["mem0", "memobase", "memos", "memsearch"],
                    help="memory_cache: assert every cache row has this system (mem0|memobase|memos|memsearch)")
    ap.add_argument("--expected-config", type=str, default=None,
                    choices=["A_paired", "B_remote"],
                    help="memory_cache: assert every cache row has this config (A_paired|B_remote)")
    ap.add_argument("--cache-strict", action=argparse.BooleanOptionalAction, default=True,
                    help="memory_cache: raise on cache-miss lookups (default True)")
    ap.add_argument("--temporal-window-days", type=float, default=None,
                    help="temporal adapter: only consider messages within this many days of the most recent message (default: all)")
    ap.add_argument("--stages", nargs="+", default=["add"], choices=["add", "search", "answer", "evaluate"])
    ap.add_argument("--namespace", type=str, default="run")
    ap.add_argument("--run-id", type=str, default=None, help="Optional stable run id (overrides namespace)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume-friendly flag (wired for next pipeline step)")
    ap.add_argument("--force", action="store_true", help="Force recompute flag (wired for next pipeline step)")
    ap.add_argument("--judge-enabled", action=argparse.BooleanOptionalAction, default=False, help="Enable hybrid judge scoring path")
    ap.add_argument("--judge-runs", type=int, default=3, help="Judge voting runs for hybrid scoring")
    ap.add_argument("--output-dir", type=str, default="eval/results")
    ap.add_argument("--config", type=str, default="eval/config/pipeline.yaml")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--qa-limit", type=int, default=None)
    ap.add_argument("--message-limit", type=int, default=None)
    ap.add_argument("--user-limit", type=int, default=0,
                    help="Limit to first N ego agents (alphabetical). 0 = all. "
                         "For Spark 8-agent subset: --user-limit 8.")
    ap.add_argument("--limit-egos-from-memcache", type=str, default=None,
                    help="Path to a memcache JSONL; only keep QA rows whose ego "
                         "matches the distinct 'namespace' values in the cache file.")
    ap.add_argument("--prompt-variant", type=str, default=None,
                    choices=["paraphrased", "reordered"],
                    help="Prompt variant for robustness ablation (overrides dim hints)")
    ap.add_argument("--d6-arm", type=str, default="A", choices=["A", "B"],
                    help="D6 permission prompt arm: 'A' (default) keeps the "
                         "original D4 prompt; 'B' uses D6_ARM_B_SYSTEM "
                         "for d4_permission items only.")
    ap.add_argument("--d6-probe-mode", type=str, default="third_party",
                    choices=["third_party", "self_ego"],
                    help="D6 paired-probe protocol mode (only affects "
                         "d4_permission items): 'third_party' (default) keeps "
                         "the original asker; 'self_ego' overrides asker to "
                         "the ego agent. The self_ego pass is the second half "
                         "of the access-control behavioural test. Outputs from "
                         "self_ego runs SHOULD use a separate --output-dir so "
                         "they do not overwrite the third_party results.")
    ap.add_argument("--d6-inject-access", action=argparse.BooleanOptionalAction, default=False,
                    help="G3a norm-binding test: inject [access:DENY]/[access:ALLOW] "
                         "into the prompt for d4_permission items at scoring time. "
                         "Tests whether the reader can bind explicit access markers "
                         "to disclosure decisions.")
    ap.add_argument("--d6-owner-contacts", action=argparse.BooleanOptionalAction, default=True,
                    help="D6 prompts carry the fact owner's contact list with closeness labels "
                         "(the benchmark protocol); --no-d6-owner-contacts drops it (identity ablation).")
    ap.add_argument("--d6-show-asker", action=argparse.BooleanOptionalAction, default=True,
                    help="D6 prompts name who is asking (the benchmark protocol); --no-d6-show-asker "
                         "says only 'Someone' (identity ablation).")
    ap.add_argument("--fail-on-thinking", action=argparse.BooleanOptionalAction, default=False,
                    help="Abort the cell (ThinkingEnabledError) when more than "
                         "--think-max-fraction of its answers carry <think> reasoning although "
                         "enable_thinking=False is sent (a serving stack ignoring the flag). "
                         "Without it the engine only warns loudly.")
    ap.add_argument("--think-max-fraction", type=float, default=0.02,
                    help="Share of answers with <think> reasoning above which the thinking "
                         "guard fires (default 0.02).")

    # quick smoke switches
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-messages", type=int, default=200)
    ap.add_argument("--smoke-qa", type=int, default=20)

    # Offline smoke: forces the vanilla local adapter + empty API key so the
    # existing heuristic-offline fallback in AnswerEngine answers every QA
    # deterministically. No HTTP, no API keys, no sglang/OpenRouter; CI-safe.
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip all external calls; use local adapter + heuristic answerer.",
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help=(
            "Pipeline-shape smoke test: preserve system/stages/namespace and "
            "write normal artifacts, but replace answer/judge LLM calls with "
            "\"I don't know\" and avoid external memory services."
        ),
    )

    # Trial naming: when set, output lands at
    # <output-dir>/eval_results_<trial>/<system>/ instead of
    # <output-dir>/<system>/. Re-run with different --trial-name values to
    # build the trial set (s1, s2, s3, ...) that paper_data.SEEDS aggregates.
    # External users should pair this with MEMARENA_SEEDS=s1,s2,s3 when
    # invoking figure scripts.
    ap.add_argument(
        "--trial-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Trial identifier; routes outputs under eval_results_<trial>/.",
    )

    # answer overrides
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--endpoint", type=str, default=None)
    ap.add_argument("--api-key", type=str, default=None)
    # judge overrides (default: same as answering model/endpoint)
    ap.add_argument("--judge-model", type=str, default=None, help="Judge LLM model name (default: same as --model)")
    ap.add_argument("--judge-endpoint", type=str, default=None, help="Judge LLM base URL (default: same as --endpoint)")
    ap.add_argument("--judge-api-key", type=str, default=None, help="Judge LLM API key (default: same as --api-key)")
    # secondary judge for cross-judge validation
    ap.add_argument("--secondary-judge-model", type=str, default=None, help="Secondary judge model (e.g. gpt-4o-mini)")
    ap.add_argument("--secondary-judge-endpoint", type=str, default=None, help="Secondary judge base URL (e.g. https://api.openai.com/v1)")
    ap.add_argument("--secondary-judge-api-key", type=str, default=None, help="Secondary judge API key")
    ap.add_argument("--answer-concurrency", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Override the answering temperature from --config "
                         "(the paper's runs use 0.3 via scripts/run_eval_matrix.sh).")
    ap.add_argument("--trial-seed", type=int, default=None,
                    help="Integer seed recorded in output metadata for trial provenance. "
                         "Suggested: s1=1001, s2=1002, s3=1003.")
    ap.add_argument("--hw-marker-file", type=str, default=None,
                    help="Path to append JSONL hardware telemetry markers per LLM call "
                         "(ts_start_ms, ts_end_ms, instance_id, phase, llm_model, tokens). "
                         "Used by Spark edge-deployment measurements (spark_remote.md §8b).")
    ap.add_argument("--context-length", type=int, default=None, help="Max prompt token budget; context is truncated to fit (0=unlimited)")
    ap.add_argument("--max-tokens", type=int, default=None, help="Max LLM response tokens (default: from yaml or 8192)")
    ap.add_argument("--max-previous-context", type=int, default=4096,
                    help="Minimum tokens reserved for conversation history in direct eval (default: 4096)")
    ap.add_argument("--timing-fraction", type=float, default=0.0,
                    help="Fraction of queries per dimension to run at low concurrency for reliable timing "
                         "(e.g. 0.25). 0 = disabled, all queries run at --answer-concurrency.")
    ap.add_argument("--timing-concurrency", type=int, default=4,
                    help="Concurrency for the timing sample (default: 4)")
    ap.add_argument("--timing-only", action="store_true", default=False,
                    help="Run only the timing sample (per-dim stratified subset), skip bulk phase")
    ap.add_argument("--eval-concurrency", type=int, default=None, help="Concurrency for judge LLM calls in evaluate stage (default: same as answer-concurrency)")
    ap.add_argument(
        "--interleaved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interleave per-message ingestion and per-anchor QA (default: enabled)",
    )
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show eval progress bars for search/answer stages (default: enabled)",
    )
    ap.add_argument("--verbose", action="store_true", help="Enable detailed eval logs")
    ap.add_argument("--log-every", type=int, default=50, help="Verbose log cadence for stream/question loops")
    ap.add_argument("--print-full-io", action="store_true", help="Print full OpenClaw input/output for each call")

    # OpenClaw backend options (used by baseline_simplerag / baseline_session / openclaw)
    ap.add_argument("--openclaw-stateless", action="store_true", help="Use isolated OpenClaw session per question")
    ap.add_argument("--openclaw-session-id", type=str, default=None, help="Base OpenClaw session id for answer stage")
    ap.add_argument("--openclaw-new-session", action="store_true", help="Force a fresh OpenClaw session id even when --openclaw-session-id is fixed")
    ap.add_argument("--openclaw-model", type=str, default=None, help="Temporarily set OpenClaw default model for this benchmark run (e.g., sglang/qwen3), then restore")
    ap.add_argument("--openclaw-timeout", type=int, default=60, help="Per-question OpenClaw timeout seconds")
    ap.add_argument("--openclaw-ingest-chunk-lines", type=int, default=120, help="Messages per OpenClaw ingest chunk (baseline_session)")
    ap.add_argument("--openclaw-ingest-mode", type=str, choices=["ack", "chat"], default="ack", help="Ingest behavior: ack=reply ACK only; chat=reply naturally for every input")
    ap.add_argument("--openclaw-user-context-path", type=str, default=None, help="Optional JSON with per-user role/permission context")
    ap.add_argument("--openclaw-user-context-max-chars", type=int, default=12000, help="Max chars from user context JSON to load")
    ap.add_argument("--openclaw-console-log", type=str, default=None, help="Append raw openclaw subprocess stdout/stderr to this log file")
    ap.add_argument("--openclaw-bin", type=str, default="openclaw", help="OpenClaw CLI binary path")
    ap.add_argument(
        "--openclaw-inject-soul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Temporarily replace OpenClaw workspace SOUL.md during benchmark, then auto-restore (default: enabled)",
    )
    ap.add_argument(
        "--openclaw-soul-path",
        type=str,
        default="additional_SOUL.md",
        help="Path to benchmark SOUL markdown to inject into OpenClaw workspace",
    )
    ap.add_argument(
        "--openclaw-workspace-soul-path",
        type=str,
        default="~/.openclaw/workspace/SOUL.md",
        help="Target OpenClaw workspace SOUL.md path to override temporarily",
    )
    ap.add_argument(
        "--openclaw-cleanup-after-run",
        action="store_true",
        help="Best-effort reset OpenClaw test sessions after evaluation",
    )

    return ap.parse_args()


async def _main_async(args: argparse.Namespace) -> None:
    load_dotenv()

    if getattr(args, "dry_run", False):
        # Force a fully local, no-network config. Vanilla adapter reads ego
        # context from corpus_sessions in-process; empty api_key makes
        # AnswerEngine fall back to its heuristic offline path.
        if args.system not in ("vanilla", "baseline_session"):
            args.system = "vanilla"
        args.stages = ["answer"]
        args.judge_enabled = False
        args.api_key = ""
        args.interleaved = False
        args.qa_limit = min(args.qa_limit, 10) if args.qa_limit else 10
        args.message_limit = min(args.message_limit, 50) if args.message_limit else 50

    cfg_path = Path(args.config).resolve()
    cfg_doc = _read_yaml(cfg_path)
    if getattr(args, "test", False):
        cfg_doc["_test_mode"] = True

    answer_cfg = _answer_cfg_from_yaml(cfg_doc)
    answer_cfg.test_mode = bool(getattr(args, "test", False))
    if getattr(args, "dry_run", False):
        answer_cfg.api_key = None
    if args.model:
        answer_cfg.model = args.model
    if args.endpoint:
        answer_cfg.endpoint = args.endpoint
    if args.api_key:
        answer_cfg.api_key = args.api_key
    if args.judge_model:
        answer_cfg.judge_model = args.judge_model
    if args.judge_endpoint:
        answer_cfg.judge_endpoint = args.judge_endpoint
    if args.judge_api_key:
        answer_cfg.judge_api_key = args.judge_api_key
    if args.secondary_judge_model:
        answer_cfg.secondary_judge_model = args.secondary_judge_model
    if args.secondary_judge_endpoint:
        answer_cfg.secondary_judge_endpoint = args.secondary_judge_endpoint
    if args.secondary_judge_api_key:
        answer_cfg.secondary_judge_api_key = args.secondary_judge_api_key
    if args.answer_concurrency is not None:
        answer_cfg.concurrency = max(1, int(args.answer_concurrency))
    if args.temperature is not None:
        answer_cfg.temperature = float(args.temperature)
    if args.trial_seed is not None:
        # Trial seed is purely provenance metadata — AnswerConfig may or may
        # not have a matching field; stash it directly so the output JSON
        # writer can emit it. Safe on dataclass (setattr works).
        try:
            setattr(answer_cfg, "trial_seed", int(args.trial_seed))
        except Exception:
            pass
    if args.hw_marker_file:
        answer_cfg.hw_marker_file = args.hw_marker_file
    if args.max_tokens is not None:
        answer_cfg.max_tokens = max(1, int(args.max_tokens))
    if args.context_length is not None:
        answer_cfg.context_length = max(0, int(args.context_length))
    answer_cfg.timing_fraction = max(0.0, float(args.timing_fraction))
    answer_cfg.timing_concurrency = max(1, int(args.timing_concurrency))
    answer_cfg.timing_only = bool(args.timing_only)
    answer_cfg.prompt_variant = args.prompt_variant
    answer_cfg.d6_arm = str(getattr(args, "d6_arm", "A") or "A").upper()
    answer_cfg.d6_probe_mode = str(getattr(args, "d6_probe_mode", "third_party") or "third_party").lower()
    answer_cfg.d6_inject_access_marker = bool(getattr(args, "d6_inject_access", False))
    answer_cfg.d6_owner_contacts = bool(getattr(args, "d6_owner_contacts", True))
    answer_cfg.d6_show_asker = bool(getattr(args, "d6_show_asker", True))
    answer_cfg.fail_on_thinking = bool(getattr(args, "fail_on_thinking", False))
    answer_cfg.think_max_fraction = max(0.0, float(getattr(args, "think_max_fraction", 0.02)))
    answer_cfg.interleaved_mode = bool(args.interleaved)
    answer_cfg.show_progress = bool(args.progress)
    answer_cfg.verbose = bool(args.verbose)
    answer_cfg.log_every = max(1, int(args.log_every))
    answer_cfg.print_full_io = bool(args.print_full_io)
    # SOUL override is handled at CLI/process level (temporary OpenClaw workspace SOUL.md replacement),
    # not via per-session bootstrap prompts.
    answer_cfg.openclaw_inject_soul = False
    answer_cfg.openclaw_soul_path = (str(args.openclaw_soul_path).strip() if args.openclaw_soul_path else None)
    answer_cfg.openclaw_cleanup_after_run = bool(args.openclaw_cleanup_after_run)
    answer_cfg.openclaw_ingest_mode = str(args.openclaw_ingest_mode or "ack").strip().lower()
    answer_cfg.openclaw_user_context_path = (str(args.openclaw_user_context_path).strip() if args.openclaw_user_context_path else None)
    answer_cfg.openclaw_user_context_max_chars = max(1000, int(args.openclaw_user_context_max_chars))
    answer_cfg.openclaw_console_log_path = (str(args.openclaw_console_log).strip() if args.openclaw_console_log else None)

    if answer_cfg.api_key in {"", "null", "None"}:
        answer_cfg.api_key = None
    if answer_cfg.test_mode:
        answer_cfg.api_key = None
        answer_cfg.judge_api_key = None
        answer_cfg.secondary_judge_api_key = None

    selected_system = args.system

    # vanilla/oracle: force non-interleaved mode and answer-only stage
    # (context is built internally from corpus_sessions, not from adapter search).
    # inmem supports ego/time-scoped retrieval in the adapter, and memory_cache
    # already contains frozen search hits. Both should use the normal
    # add/search -> batched answer path rather than streaming corpus ingestion.
    if selected_system in ("vanilla", "oracle"):
        args.interleaved = False
        answer_cfg.interleaved_mode = False
        if not any(s in args.stages for s in ["answer"]):
            args.stages = ["answer"]
    elif selected_system in ("inmem", "baseline_simplerag", "inmem_text_sessions", "memory_cache"):
        args.interleaved = False
        answer_cfg.interleaved_mode = False

    if selected_system == "openclaw":
        selected_system = "baseline_simplerag"
    if not answer_cfg.test_mode:
        _validate_env_for_system(selected_system)
    resolved_namespace = _resolve_namespace(args)

    openclaw_session_id = args.openclaw_session_id
    if args.openclaw_new_session:
        suffix = str(int(time.time()))
        openclaw_session_id = f"{openclaw_session_id}-{suffix}" if openclaw_session_id else f"memarena-openclaw-{suffix}"

    # system-specific answer backend wiring
    if selected_system == "baseline_simplerag":
        # Historical name for the simple RAG baseline: BM25 in-memory search
        # with the configured LLM answerer. This path must not call OpenClaw.
        answer_cfg.backend = "llm"
    elif selected_system == "baseline_session":
        answer_cfg.backend = "openclaw-session"
        answer_cfg.model = "baseline_session"
        answer_cfg.openclaw_stateless = False
        answer_cfg.openclaw_session_id = openclaw_session_id
        answer_cfg.openclaw_model = None  # strict mode: no per-session /new pre-prompt
        answer_cfg.openclaw_timeout_seconds = max(1, int(args.openclaw_timeout))
        answer_cfg.openclaw_ingest_chunk_lines = max(1, int(args.openclaw_ingest_chunk_lines))
        answer_cfg.openclaw_bin = str(args.openclaw_bin)
    else:
        answer_cfg.backend = "llm"

    search_cfg = cfg_doc.get("search") if isinstance(cfg_doc.get("search"), dict) else {}
    top_k = int(args.top_k) if args.top_k is not None else int(search_cfg.get("top_k", 10))

    output_root = Path(args.output_dir).resolve()
    if getattr(args, "trial_name", None):
        trial = str(args.trial_name).strip()
        if not trial:
            raise ValueError("--trial-name must be non-empty")
        output_root = output_root / f"eval_results_{trial}"
    output_root = output_root / selected_system

    # ------------------------------------------------------------------
    # MASim run-dir auto-loading: convert corpus + eval_instances to the
    # flat transcript + QA files the existing pipeline expects.
    # ------------------------------------------------------------------
    if getattr(args, "run_dir", None):
        from eval.src.masim_loader import (
            load_masim_messages,
            load_masim_qa,
            load_corpus_sessions_dict,
            dump_masim_transcript,
            dump_masim_qa,
        )

        masim_run_dir = Path(args.run_dir).resolve()
        if not masim_run_dir.is_dir():
            raise ValueError(f"--run-dir does not exist or is not a directory: {masim_run_dir}")

        transcript_path = output_root / f"masim_transcript_{resolved_namespace}.jsonl"
        qa_auto_path = output_root / f"masim_qa_{resolved_namespace}.json"

        if not transcript_path.exists() or bool(getattr(args, "force", False)):
            print(f"[eval.cli] converting MASim corpus → {transcript_path}")
            msgs = load_masim_messages(masim_run_dir)
            dump_masim_transcript(msgs, transcript_path)
            print(f"[eval.cli] wrote {len(msgs)} turns")

        if not qa_auto_path.exists() or bool(getattr(args, "force", False)):
            dimensions = list(args.dimensions) if getattr(args, "dimensions", None) else None
            print(f"[eval.cli] converting MASim eval_instances → {qa_auto_path} (dims={dimensions or 'all'})")
            corpus_sessions = load_corpus_sessions_dict(masim_run_dir)
            qas = load_masim_qa(masim_run_dir, dimensions=dimensions, corpus_sessions=corpus_sessions)
            dump_masim_qa(qas, qa_auto_path)
            print(f"[eval.cli] wrote {len(qas)} QA items")

        if not args.messages:
            args.messages = str(transcript_path)
        if not args.qa:
            args.qa = str(qa_auto_path)

    if selected_system == "temporal" and getattr(args, "temporal_window_days", None) is not None:
        temporal_cfg = cfg_doc.get("temporal") if isinstance(cfg_doc.get("temporal"), dict) else {}
        temporal_cfg["window_days"] = float(args.temporal_window_days)
        cfg_doc["temporal"] = temporal_cfg

    if selected_system == "oracle_gated":
        # oracle_gated behaves like oracle for staging purposes
        args.interleaved = False
        answer_cfg.interleaved_mode = False
        if not any(s in args.stages for s in ["answer"]):
            args.stages = ["answer"]

    if selected_system == "memory_cache":
        if not args.cache_path:
            raise ValueError("--cache-path is required when --system memory_cache")
        mc_cfg = cfg_doc.get("memory_cache") if isinstance(cfg_doc.get("memory_cache"), dict) else {}
        mc_cfg["cache_path"] = str(Path(args.cache_path).resolve())
        if args.expected_extractor:
            mc_cfg["expected_extractor"] = args.expected_extractor
        if args.expected_memory_system:
            mc_cfg["expected_system"] = args.expected_memory_system
        if args.expected_config:
            mc_cfg["expected_config"] = args.expected_config
        mc_cfg["strict"] = bool(args.cache_strict)
        cfg_doc["memory_cache"] = mc_cfg

    adapter = build_adapter(selected_system, cfg=cfg_doc, output_dir=output_root)
    pipeline = EvalPipeline(
        adapter=adapter,
        answer_cfg=answer_cfg,
        output_dir=output_root,
        masim_run_dir=masim_run_dir if getattr(args, "run_dir", None) else None,
    )

    # --user-limit: filter QA to first N ego agents (alphabetical)
    if getattr(args, "user_limit", 0) and args.user_limit > 0 and args.qa:
        import tempfile
        with open(args.qa) as _f:
            _raw = json.load(_f)
        # Handle both formats: {"qars": [...]} and flat [...]
        if isinstance(_raw, dict) and "qars" in _raw:
            _all_qas = _raw["qars"]
            _wrapper = _raw
        else:
            _all_qas = _raw
            _wrapper = None

        def _get_ego(q):
            meta = q.get("meta", {}) or q.get("metadata", {}) or {}
            return (meta.get("ego_agent_id", "")
                    or meta.get("instance_query_agent", "")
                    or meta.get("query_agent", ""))

        _agents = sorted({_get_ego(q) for q in _all_qas} - {""})
        _keep = set(_agents[:args.user_limit])
        _filtered = [q for q in _all_qas if _get_ego(q) in _keep]

        _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        if _wrapper is not None:
            _wrapper["qars"] = _filtered
            json.dump(_wrapper, _tmp, ensure_ascii=False, indent=2)
        else:
            json.dump(_filtered, _tmp, ensure_ascii=False, indent=2)
        _tmp.close()
        print(f"[eval.cli] --user-limit {args.user_limit}: {len(_filtered)}/{len(_all_qas)} QAs "
              f"from agents {sorted(_keep)[:3]}...")
        args.qa = _tmp.name

    # --limit-egos-from-memcache: filter QA to egos present in a memcache JSONL
    if getattr(args, "limit_egos_from_memcache", None) and args.qa:
        import tempfile as _tmpmod2
        mc_path = Path(args.limit_egos_from_memcache).resolve()
        mc_egos: set[str] = set()
        with open(mc_path) as _mf:
            for _line in _mf:
                _row = json.loads(_line)
                mc_egos.add(_row["namespace"])
        with open(args.qa) as _f2:
            _raw2 = json.load(_f2)
        if isinstance(_raw2, dict) and "qars" in _raw2:
            _all_qas2 = _raw2["qars"]
            _wrapper2 = _raw2
        else:
            _all_qas2 = _raw2
            _wrapper2 = None

        def _get_ego2(q):
            meta = q.get("meta", {}) or q.get("metadata", {}) or {}
            return (meta.get("ego_agent_id", "")
                    or meta.get("instance_query_agent", "")
                    or meta.get("query_agent", ""))

        _filtered2 = [q for q in _all_qas2 if _get_ego2(q) in mc_egos]
        _tmp2 = _tmpmod2.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        if _wrapper2 is not None:
            _wrapper2["qars"] = _filtered2
            json.dump(_wrapper2, _tmp2, ensure_ascii=False, indent=2)
        else:
            json.dump(_filtered2, _tmp2, ensure_ascii=False, indent=2)
        _tmp2.close()
        print(f"[eval.cli] --limit-egos-from-memcache: {len(_filtered2)}/{len(_all_qas2)} QAs, "
              f"{len(mc_egos)} egos: {sorted(mc_egos)}")
        args.qa = _tmp2.name

    message_limit = args.message_limit
    qa_limit = args.qa_limit
    if args.smoke:
        message_limit = args.smoke_messages if message_limit is None else min(message_limit, args.smoke_messages)
        qa_limit = args.smoke_qa if qa_limit is None else min(qa_limit, args.smoke_qa)

    model_override: Optional[_TemporaryOpenClawModelOverride] = None
    if selected_system in {"baseline_simplerag", "baseline_session"} and bool(args.openclaw_model) and not answer_cfg.test_mode:
        model_override = _TemporaryOpenClawModelOverride(model_ref=str(args.openclaw_model), openclaw_bin=str(args.openclaw_bin))
        model_override.activate()

    soul_override: Optional[_TemporaryOpenClawSoulOverride] = None
    if selected_system in {"baseline_simplerag", "baseline_session"} and bool(args.openclaw_inject_soul) and not answer_cfg.test_mode:
        repo_root = Path(__file__).resolve().parents[1]
        src = Path(str(args.openclaw_soul_path or "additional_SOUL.md").strip())
        if not src.is_absolute():
            src = repo_root / src
        target = Path(str(args.openclaw_workspace_soul_path or "~/.openclaw/workspace/SOUL.md").strip()).expanduser().resolve()
        soul_override = _TemporaryOpenClawSoulOverride(source_soul=src, target_soul=target)
        soul_override.activate()

    if args.verbose:
        print(
            "[eval.cli] "
            f"backend={answer_cfg.backend} model={answer_cfg.model} "
            f"openclaw_model={str(args.openclaw_model or '-')} model_override={'on' if model_override else 'off'} ingest_mode={answer_cfg.openclaw_ingest_mode} "
            f"interleaved={answer_cfg.interleaved_mode} progress={answer_cfg.show_progress} "
            f"soul_override={'on' if soul_override else 'off'} user_ctx_path={answer_cfg.openclaw_user_context_path or '-'} "
            f"console_log={answer_cfg.openclaw_console_log_path or '-'} print_full_io={answer_cfg.print_full_io}"
        )

    try:
        report = await pipeline.run(
            stages=args.stages,
            namespace=resolved_namespace,
            messages_path=Path(args.messages).resolve() if args.messages else None,
            qa_path=Path(args.qa).resolve() if args.qa else None,
            top_k=top_k,
            qa_limit=qa_limit,
            message_limit=message_limit,
            resume=bool(args.resume),
            force=bool(args.force),
            judge_enabled=bool(args.judge_enabled),
            judge_runs=max(1, int(args.judge_runs)),
            eval_concurrency=(int(args.eval_concurrency) if args.eval_concurrency is not None else None),
        )
    finally:
        if soul_override is not None:
            soul_override.restore()
        if model_override is not None:
            model_override.restore()

    archive_dir = _archive_run_artifacts(output_root, resolved_namespace, args, report)
    run_card_path = archive_dir / "RUN_CARD.md"
    _write_run_card(run_card_path, report, args, archive_dir, resolved_namespace)
    report["artifacts"] = {
        "archive_dir": str(archive_dir),
        "run_card": str(run_card_path),
    }

    print("\n=== MemArena Eval Summary ===")
    print(f"system: {selected_system}")
    print(f"stages: {args.stages}")
    print(f"namespace: {resolved_namespace}")
    print(f"output_dir: {output_root}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    configure_live_output()
    args = parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
