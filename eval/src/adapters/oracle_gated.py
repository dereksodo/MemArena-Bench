from __future__ import annotations

from typing import Any, List, Optional

from .oracle_retrieval import OracleRetrievalAdapter
from ..types import SearchHit


class OracleGatedAdapter(OracleRetrievalAdapter):
    """Oracle + retrieval-time access-control filter.

    For D6 DENY items (policy_expected=="DENY_NO_ACCESS"), returns no evidence
    so the reader never sees the restricted session.  ALLOW items pass through
    identically to the base Oracle adapter.  This gives the upper-bound F1pu
    gain from retrieval-side enforcement vs prompt-only gating (A7 ablation).

    Like Oracle it is TEXT_SESSIONS, so the answer prompt is built from the
    evidence session ids, not from search(); the gate itself is applied there
    (AnswerEngine._build_text_prompt, adapter_name "oracle_gated"), which
    withholds every evidence route for DENY items. search() gates too, for
    callers that use the hits.
    """

    async def search(
        self,
        *,
        namespace: str,
        query: str,
        top_k: int,
        policy_expected: Optional[str] = None,
        **kwargs: Any,
    ) -> List[SearchHit]:
        if policy_expected == "DENY_NO_ACCESS":
            return []
        return await super().search(namespace=namespace, query=query, top_k=top_k, **kwargs)
