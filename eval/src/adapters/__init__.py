from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import PromptStyle, RetrievalAdapter
from .inmemory import InMemoryAdapter
from .vanilla import VanillaAdapter
from .oracle_retrieval import OracleRetrievalAdapter
# Backward compat alias
FullContextAdapter = OracleRetrievalAdapter
from .oracle_gated import OracleGatedAdapter
from .oracle_with_distractors import OracleWithDistractorsAdapter
from .oracle_with_random_distractors import OracleWithRandomDistractorsAdapter
from .noop import NoopAdapter
from .memos_adapter import MemosAdapter
from .mem0_adapter import Mem0Adapter
from .mem0_sdk_adapter import Mem0SdkAdapter
from .memobase_adapter import MemobaseAdapter
from .memsearch_adapter import MemsearchAdapter
from .zep_adapter import ZepAdapter
from .omniscient import OmniscientAdapter
from .temporal import TemporalAdapter
from .memory_cache import MemoryCacheAdapter
from .dense_bge_m3 import DenseBgeM3Adapter
from .dense_e5 import DenseE5Adapter
from .hybrid_bm25rerank import HybridBm25RerankAdapter
from .hybrid_rrf import HybridRrfAdapter
from .inmem_provenance import InMemProvenanceAdapter
from .inmem_text_sessions import InMemTextSessionsAdapter


def build_adapter(system: str, *, cfg: Dict[str, Any], output_dir: Path) -> RetrievalAdapter:
    s = (system or "").strip().lower()
    llm_context_cfg = cfg.get("llm_context") if isinstance(cfg.get("llm_context"), dict) else {}
    vanilla_cfg = cfg.get("vanilla") if isinstance(cfg.get("vanilla"), dict) else {}
    test_mode = bool(cfg.get("_test_mode"))

    if s == "inmem":
        return InMemoryAdapter()
    if s == "vanilla":
        return VanillaAdapter(
            max_tokens=int(vanilla_cfg.get("max_tokens", 8192)),
            max_messages=int(vanilla_cfg.get("max_messages", 2000)),
        )
    if s in {"llm", "oracle", "oracle_retrieval"}:
        return OracleRetrievalAdapter(max_messages=int(llm_context_cfg.get("max_messages", 2000)))
    if s == "oracle_gated":
        return OracleGatedAdapter(max_messages=int(llm_context_cfg.get("max_messages", 2000)))
    if s == "oracle_with_distractors":
        return OracleWithDistractorsAdapter(max_messages=int(llm_context_cfg.get("max_messages", 2000)))
    if s == "oracle_with_random_distractors":
        return OracleWithRandomDistractorsAdapter(max_messages=int(llm_context_cfg.get("max_messages", 2000)))
    if s == "baseline_session":
        return NoopAdapter()
    if s in {"baseline_simplerag", "openclaw"}:
        return InMemoryAdapter()
    if test_mode and s in {"memos", "mem0", "mem0sdk", "memobase", "memsearch", "zep"}:
        return InMemoryAdapter()
    if s == "memos":
        return MemosAdapter(cfg=cfg.get("memos") if isinstance(cfg.get("memos"), dict) else {})
    if s == "mem0":
        return Mem0Adapter(cfg=cfg.get("mem0") if isinstance(cfg.get("mem0"), dict) else {})
    if s == "mem0sdk":
        sdk_cfg = cfg.get("mem0sdk") if isinstance(cfg.get("mem0sdk"), dict) else {}
        # Allow env vars to configure embedder
        import os
        if os.getenv("MEM0_EMBEDDER"):
            sdk_cfg.setdefault("embedder_provider", os.getenv("MEM0_EMBEDDER"))
        if os.getenv("MEM0_EMBEDDER_MODEL"):
            sdk_cfg.setdefault("embedder_model", os.getenv("MEM0_EMBEDDER_MODEL"))
        return Mem0SdkAdapter(cfg=sdk_cfg)
    if s == "memobase":
        return MemobaseAdapter(cfg=cfg.get("memobase") if isinstance(cfg.get("memobase"), dict) else {})
    if s == "memsearch":
        return MemsearchAdapter(cfg=cfg.get("memsearch") if isinstance(cfg.get("memsearch"), dict) else {})
    if s == "zep":
        return ZepAdapter(cfg=cfg.get("zep") if isinstance(cfg.get("zep"), dict) else {})
    if s == "omniscient":
        return OmniscientAdapter()
    if s == "temporal":
        temporal_cfg = cfg.get("temporal") if isinstance(cfg.get("temporal"), dict) else {}
        return TemporalAdapter(
            half_life_days=float(temporal_cfg.get("half_life_days", 30)),
            temporal_alpha=float(temporal_cfg.get("temporal_alpha", 0.5)),
            window_days=temporal_cfg.get("window_days"),
        )
    if s == "memory_cache":
        mc_cfg = cfg.get("memory_cache") if isinstance(cfg.get("memory_cache"), dict) else {}
        if not mc_cfg.get("cache_path"):
            raise ValueError("memory_cache adapter requires cfg.memory_cache.cache_path")
        return MemoryCacheAdapter(
            cache_path=mc_cfg["cache_path"],
            expected_extractor=mc_cfg.get("expected_extractor"),
            expected_system=mc_cfg.get("expected_system"),
            expected_config=mc_cfg.get("expected_config"),
            strict=bool(mc_cfg.get("strict", True)),
        )

    if s == "dense_bge_m3":
        return DenseBgeM3Adapter()
    if s == "dense_e5":
        return DenseE5Adapter()
    if s == "hybrid_bm25rerank":
        return HybridBm25RerankAdapter()
    if s == "hybrid_rrf":
        return HybridRrfAdapter()
    if s == "inmem_provenance":
        return InMemProvenanceAdapter()

    if s == "inmem_text_sessions":
        return InMemTextSessionsAdapter()

    raise ValueError(f"unsupported system: {system}")


__all__ = [
    "PromptStyle",
    "RetrievalAdapter",
    "InMemoryAdapter",
    "VanillaAdapter",
    "OracleRetrievalAdapter",
    "OracleGatedAdapter",
    "OracleWithDistractorsAdapter",
    "OracleWithRandomDistractorsAdapter",
    "FullContextAdapter",
    "NoopAdapter",
    "MemosAdapter",
    "Mem0Adapter",
    "MemobaseAdapter",
    "MemsearchAdapter",
    "ZepAdapter",
    "OmniscientAdapter",
    "TemporalAdapter",
    "MemoryCacheAdapter",
    "DenseBgeM3Adapter",
    "DenseE5Adapter",
    "HybridBm25RerankAdapter",
    "build_adapter",
]
