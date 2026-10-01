"""Bias mitigation for LLM-as-a-Judge: position swap, reference support, CoT scoring."""

from __future__ import annotations

from typing import Any, List, Tuple

from MASim.prompts import (
    BIAS_JUDGE_AB as JUDGE_TEMPLATE_AB,
    BIAS_JUDGE_BA as JUDGE_TEMPLATE_BA,
    BIAS_JUDGE_SYSTEM as JUDGE_SYSTEM_PROMPT,
)


def position_swap_score(
    llm_client: Any,
    query: str,
    response: str,
    reference: str,
    rubric: str,
) -> Tuple[float, str, list]:
    """Evaluate with position swap to mitigate positional bias.

    Runs evaluation twice with response and reference in swapped positions,
    then averages the scores.

    Returns (avg_score, combined_reasoning, [score1, score2]).
    """
    prompt_ab = JUDGE_TEMPLATE_AB.format(
        query=query, response_a=response, response_b=reference, rubric=rubric,
    )
    prompt_ba = JUDGE_TEMPLATE_BA.format(
        query=query, response_a=reference, response_b=response, rubric=rubric,
    )

    responses = llm_client.generate_batch([
        {"system": JUDGE_SYSTEM_PROMPT, "user": prompt_ab, "tags": {"phase": "eval_bias_ab"}},
        {"system": JUDGE_SYSTEM_PROMPT, "user": prompt_ba, "tags": {"phase": "eval_bias_ba"}},
    ])

    score_ab, reasoning_ab = _parse_output(responses[0])
    score_ba, reasoning_ba = _parse_output(responses[1])

    avg_score = (score_ab + score_ba) / 2.0
    combined = f"[Pos A] {reasoning_ab}\n[Pos B] {reasoning_ba}"

    return avg_score, combined, [score_ab, score_ba]


def batch_position_swap_score(
    llm_client: Any,
    queries: List[str],
    responses: List[str],
    references: List[str],
    rubrics: List[str],
) -> List[Tuple[float, str, list]]:
    """Batch position-swap evaluation for multiple instances.

    Sends all 2*N prompts in one generate_batch() call.
    Returns list of (avg_score, combined_reasoning, [score_ab, score_ba]).
    """
    prompts = []
    for query, response, reference, rubric in zip(queries, responses, references, rubrics):
        prompt_ab = JUDGE_TEMPLATE_AB.format(
            query=query, response_a=response, response_b=reference, rubric=rubric,
        )
        prompt_ba = JUDGE_TEMPLATE_BA.format(
            query=query, response_a=reference, response_b=response, rubric=rubric,
        )
        prompts.append({"system": JUDGE_SYSTEM_PROMPT, "user": prompt_ab, "tags": {"phase": "eval_bias_ab"}})
        prompts.append({"system": JUDGE_SYSTEM_PROMPT, "user": prompt_ba, "tags": {"phase": "eval_bias_ba"}})

    raw_responses = llm_client.generate_batch(prompts)

    results = []
    for i in range(len(queries)):
        score_ab, reasoning_ab = _parse_output(raw_responses[2 * i])
        score_ba, reasoning_ba = _parse_output(raw_responses[2 * i + 1])
        avg_score = (score_ab + score_ba) / 2.0
        combined = f"[Pos A] {reasoning_ab}\n[Pos B] {reasoning_ba}"
        results.append((avg_score, combined, [score_ab, score_ba]))

    return results


def _parse_output(output: str) -> Tuple[float, str]:
    """Parse judge output for score and reasoning."""
    reasoning = ""
    score = 3.0

    for line in output.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("REASONING:"):
            reasoning = stripped.split("REASONING:", 1)[1].strip()
        elif stripped.startswith("SCORE:"):
            try:
                score = float(stripped.split("SCORE:", 1)[1].strip())
                score = max(1.0, min(5.0, score))
            except ValueError:
                pass

    if not reasoning:
        reasoning = output

    return score, reasoning
