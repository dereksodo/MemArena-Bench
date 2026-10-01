"""LLMJudge: rubric-based scoring engine using LLM-as-a-Judge.

Implements Prometheus-style rubric evaluation with bias mitigation
(position swap, reference support, CoT scoring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from MASim.core.schema import Dimension, EvalInstance
from MASim.evaluation.bias_mitigation import batch_position_swap_score, position_swap_score
from MASim.prompts import RUBRIC_JUDGE_SYSTEM as JUDGE_SYSTEM_PROMPT
from MASim.prompts import RUBRIC_JUDGE_USER as JUDGE_USER_TEMPLATE
from MASim.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class JudgeResult:
    """Result from a single evaluation judgment."""
    instance_id: str
    dimension: Dimension
    score: float
    reasoning: str = ""
    raw_scores: List[float] = field(default_factory=list)  # for position swap
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMJudge:
    """Score system responses using dimension-specific rubrics."""

    def __init__(
        self,
        llm_client: Any,
        rubrics: Optional[Dict[Dimension, str]] = None,
        use_position_swap: bool = True,
    ):
        self.llm_client = llm_client
        self.rubrics = rubrics or self._default_rubrics()
        self.use_position_swap = use_position_swap

    def evaluate(
        self,
        instance: EvalInstance,
        response: str,
        reference: Optional[str] = None,
    ) -> JudgeResult:
        """Evaluate a single system response.

        Args:
            instance: The evaluation instance with query and ground truth.
            response: The system's response to judge.
            reference: Optional reference answer (derived from GT if not provided).

        Returns:
            JudgeResult with score and reasoning.
        """
        if reference is None:
            reference = self._derive_reference(instance)

        rubric = self.rubrics.get(instance.dimension, self._generic_rubric())

        if self.use_position_swap:
            score, reasoning, raw_scores = position_swap_score(
                self.llm_client,
                query=instance.query,
                response=response,
                reference=reference,
                rubric=rubric,
            )
        else:
            score, reasoning = self._single_judge(
                instance.query, response, reference, rubric
            )
            raw_scores = [score]

        return JudgeResult(
            instance_id=instance.instance_id,
            dimension=instance.dimension,
            score=score,
            reasoning=reasoning,
            raw_scores=raw_scores,
        )

    def evaluate_batch(
        self,
        instances: List[EvalInstance],
        responses: List[str],
        references: Optional[List[str]] = None,
    ) -> List[JudgeResult]:
        """Evaluate a batch of responses via bulk LLM calls."""
        if references is None:
            references = [self._derive_reference(inst) for inst in instances]

        rubrics = [
            self.rubrics.get(inst.dimension, self._generic_rubric())
            for inst in instances
        ]

        if self.use_position_swap:
            queries = [inst.query for inst in instances]
            swap_results = batch_position_swap_score(
                self.llm_client, queries, responses, references, rubrics,
            )
            results = []
            for inst, (score, reasoning, raw_scores) in zip(instances, swap_results):
                results.append(JudgeResult(
                    instance_id=inst.instance_id,
                    dimension=inst.dimension,
                    score=score,
                    reasoning=reasoning,
                    raw_scores=raw_scores,
                ))
            return results
        else:
            # Batch single-judge calls
            prompts = [
                {
                    "system": JUDGE_SYSTEM_PROMPT,
                    "user": JUDGE_USER_TEMPLATE.format(
                        query=inst.query, reference=ref,
                        response=resp, rubric=rubric,
                    ),
                    "tags": {"phase": "eval_judge"},
                }
                for inst, resp, ref, rubric in zip(instances, responses, references, rubrics)
            ]
            outputs = self.llm_client.generate_batch(prompts)
            results = []
            for inst, output in zip(instances, outputs):
                score, reasoning = self._parse_judge_output(output)
                results.append(JudgeResult(
                    instance_id=inst.instance_id,
                    dimension=inst.dimension,
                    score=score,
                    reasoning=reasoning,
                    raw_scores=[score],
                ))
            return results

    def _single_judge(
        self, query: str, response: str, reference: str, rubric: str
    ) -> Tuple[float, str]:
        """Single pass of LLM judge."""
        prompt = JUDGE_USER_TEMPLATE.format(
            query=query,
            reference=reference,
            response=response,
            rubric=rubric,
        )
        output = self.llm_client.generate(JUDGE_SYSTEM_PROMPT, prompt, tags={"phase": "eval_judge"})
        score, reasoning = self._parse_judge_output(output)
        return score, reasoning

    def _parse_judge_output(self, output: str) -> Tuple[float, str]:
        """Parse judge output into score and reasoning."""
        reasoning = ""
        score = 3.0  # default

        lines = output.strip().split("\n")
        for line in lines:
            if line.strip().startswith("REASONING:"):
                reasoning = line.split("REASONING:", 1)[1].strip()
            elif line.strip().startswith("SCORE:"):
                try:
                    score = float(line.split("SCORE:", 1)[1].strip())
                    score = max(1.0, min(5.0, score))
                except ValueError:
                    pass

        if not reasoning:
            reasoning = output

        return score, reasoning

    def _derive_reference(self, instance: EvalInstance) -> str:
        """Derive a reference answer from ground truth."""
        gt = instance.ground_truth
        if "answer" in gt:
            return str(gt["answer"])
        if "original_fact" in gt:
            return f"Original: {gt['original_fact']}. Source: {gt.get('source_agent', 'unknown')}"
        if "expected_action" in gt:
            return f"Expected: {gt['expected_action']}"
        if "deleted_span" in gt:
            return gt["deleted_span"]
        return str(gt)

    def _default_rubrics(self) -> Dict[Dimension, str]:
        """Load default rubrics for all dimensions."""
        from MASim.evaluation.rubrics import d1_conflict, d3_confabulation, d4_privacy, d5_cloze

        return {
            Dimension.D1_CONFLICT: d1_conflict.RUBRIC,
            Dimension.D3_CONFABULATION: d3_confabulation.RUBRIC,
            Dimension.D4_PERMISSION: d4_privacy.RUBRIC,
            Dimension.D5_CLOZE: d5_cloze.RUBRIC,
        }

    def _generic_rubric(self) -> str:
        return """Score 1-5:
5: Response is completely accurate and comprehensive
4: Response is mostly accurate with minor omissions
3: Response is partially correct but has notable gaps
2: Response has significant errors or omissions
1: Response is incorrect or irrelevant"""
