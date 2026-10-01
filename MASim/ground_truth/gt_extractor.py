"""Extract ground truth from simulation logs for evaluated dimensions.

Central coordinator that calls dimension-specific extractors and
builds the complete set of evaluation instances.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from MASim.core.schema import (
    AnaphoraGT, AutonomousPrivacyGT, ConfabulationGT, ConflictGT,
    DialogueCorpus, Dimension, EvalInstance, MetadataGT, PermissionGT, QAGT,
)
from MASim.ground_truth import d1_conflict, d2_anaphora, d3_confabulation
from MASim.ground_truth import d4_permission, d5_cloze, d6_metadata, d7_qa
from MASim.ground_truth.query_hardener import HardeningConfig, QueryHardener
from MASim.utils.logging import get_logger

log = get_logger(__name__)


class GTExtractor:
    """Extract ground truth and generate evaluation instances for all dimensions."""

    def __init__(
        self,
        llm_client: Any = None,
        hardening_cfg: Optional[HardeningConfig] = None,
        graph_layer_weights: Optional[Sequence[float]] = None,
    ):
        self.llm_client = llm_client
        self.hardening_cfg = hardening_cfg
        self.graph_layer_weights = graph_layer_weights

    def extract_all(
        self,
        corpus: DialogueCorpus,
        conflict_gts: Optional[List[ConflictGT]] = None,
        anaphora_gts: Optional[List[AnaphoraGT]] = None,
        permission_gts: Optional[List[PermissionGT]] = None,
        autonomous_privacy_gts: Optional[List[AutonomousPrivacyGT]] = None,
        max_per_dimension: int = 200,
        d6_probes: str = "exclude",
    ) -> Dict[Dimension, List[EvalInstance]]:
        """Extract evaluation instances for evaluated dimensions.

        Args:
            corpus: The full dialogue corpus.
            conflict_gts: Pre-computed conflict ground truths from injection.
            anaphora_gts: Pre-computed anaphora ground truths from tracking.
            permission_gts: Permission ground truths from injection (not used:
                D4 reads the injected turns and the social graph from ``corpus``).
            autonomous_privacy_gts: Autonomous privacy ground truths (not used, as above).
            max_per_dimension: Maximum instances per dimension.

        Returns:
            Dict mapping each Dimension to its list of EvalInstances.
        """
        results: Dict[Dimension, List[EvalInstance]] = {}

        # D1: Conflict preservation
        log.info("Extracting D1: Conflict preservation instances...")
        results[Dimension.D1_CONFLICT] = d1_conflict.generate_instances(
            corpus, conflict_gts or [], max_instances=max_per_dimension,
        )

        # D2: Cross-session anaphora
        log.info("Extracting D2: Cross-session anaphora instances...")
        results[Dimension.D2_ANAPHORA] = d2_anaphora.generate_instances(
            corpus, anaphora_gts or [], max_instances=max_per_dimension,
        )

        # D3: Confabulation resistance
        log.info("Extracting D3: Confabulation resistance instances...")
        results[Dimension.D3_CONFABULATION] = d3_confabulation.generate_instances(
            corpus, max_instances=max_per_dimension,
        )

        # D4 (paper D6): permission-aware access. The builder reads the injected
        # permission / autonomous-privacy turns from the corpus and friends_only
        # access from the Dunbar graph (corpus.social_graph_edges);
        # permission_gts / autonomous_privacy_gts
        # (the injectors' co-attendance rule) are not used for it.
        log.info("Extracting D4: Permission management instances...")
        results[Dimension.D4_PERMISSION] = d4_permission.generate_instances(
            corpus, max_instances=max_per_dimension, llm_client=self.llm_client,
            layer_weights=self.graph_layer_weights, probes=d6_probes,
        )

        # D5: Cloze-deletion fidelity
        log.info("Extracting D5: Cloze-deletion instances...")
        results[Dimension.D5_CLOZE] = d5_cloze.generate_instances(
            corpus, max_instances=max_per_dimension, llm_client=self.llm_client,
        )

        # D5: Next-turn prediction (from truncated racing sessions)
        next_turn_instances = d5_cloze.generate_next_turn_instances(
            corpus, max_instances=max_per_dimension, llm_client=self.llm_client,
        )
        if next_turn_instances:
            log.info("D5 next-turn-prediction: %d instances", len(next_turn_instances))
            results[Dimension.D5_CLOZE].extend(next_turn_instances)

        # D6: Metadata completeness
        log.info("Extracting D6: Metadata completeness instances...")
        results[Dimension.D6_METADATA] = d6_metadata.generate_instances(
            corpus, max_instances=max_per_dimension,
        )

        # D7: Standard QA + temporal reasoning
        log.info("Extracting D7: Standard QA instances...")
        results[Dimension.D7_QA] = d7_qa.generate_instances(
            corpus, max_instances=max_per_dimension, llm_client=self.llm_client,
        )

        # --- Query hardening pass (D8/D10 + paraphrase + perspective) ---
        if self.llm_client is not None and self.hardening_cfg is not None:
            hardener = QueryHardener(self.llm_client, self.hardening_cfg)
            results = hardener.harden_all(results, corpus)

        total = sum(len(v) for v in results.values())
        log.info(
            "Extracted %d total eval instances: %s",
            total,
            {d.value: len(v) for d, v in results.items()},
        )
        return results
