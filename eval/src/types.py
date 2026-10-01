from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MessageEntry:
    msg_id: str
    occur_ts: str
    deliver_ts: str
    user_id: Optional[int]
    thread_id: str
    text: str
    refs: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAItem:
    question_id: str
    question: str
    answer: str
    question_type: str = "open_ended"  # open_ended | multiple_choice
    options: Optional[List[str]] = None
    correct_option: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    msg_id: str
    score: float
    text: str
    occur_ts: str
    thread_id: str
    user_id: Optional[int]


@dataclass
class SearchRecord:
    question_id: str
    query: str
    hits: List[SearchHit]
    search_time_ms: Optional[float] = None


@dataclass
class AnswerRecord:
    question_id: str
    question: str
    answer: str
    prediction: Optional[str]   # None when LLM call failed after all retries
    model: str
    raw_response: str
    answer_time_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    search_time_ms: Optional[float] = None
    timing_reliable: Optional[bool] = None
    # Reasoning in the raw output (Qwen3 <think>, or reasoning_content from a
    # server-side reasoning parser). think_truncated: nothing was left outside
    # the reasoning, so the prediction is recorded as empty.
    think_detected: Optional[bool] = None
    think_truncated: Optional[bool] = None


@dataclass
class EvalRecord:
    question_id: str
    question_type: str
    gold_answer: str
    prediction: str
    correct: bool
    score: float
    reason: str
    answer_scored: bool = True
    policy_expected: Optional[str] = None
    expected_answer_mode: Optional[str] = None
    # D4_permission ArmB judge output. Populated only on D4 records; the
    # category (REFUSAL/NONE/COMPLY/PARSE_ERROR) drives `correct` for D4
    # via `_armB_correctness()`. Other dimensions leave these as None.
    policy_category: Optional[str] = None
    leaked_fact_in_output: Optional[bool] = None
    # secondary judge fields (populated when --secondary-judge-model is set)
    secondary_correct: Optional[bool] = None
    secondary_score: Optional[float] = None
    secondary_reason: Optional[str] = None
    # The answer was cut inside <think> (see AnswerRecord.think_truncated);
    # it was scored as an empty prediction.
    think_truncated: Optional[bool] = None
