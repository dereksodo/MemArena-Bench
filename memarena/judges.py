"""Judge registry shared by the grid runners, the judges and the figure loaders.

Standard library only: ``scripts/grid_helper.py`` imports it before any venv
exists.

Tags
----
``gpt4omini``  openai/gpt-4o-mini-2024-07-18 via OpenRouter; the judge of the
               submitted paper and the primary (default) judge.
``deepseek``   deepseek/deepseek-v4.1-flash via OpenRouter; the secondary judge,
               for agreement / kappa.

File names
----------
A judged cell is ``<system>/evaluation_results_<ns>_judge_<file_tag>.json``.
``gpt4omini`` keeps the historical file tag ``remote`` (every gpt-4o-mini file
ever written is ``*_judge_remote.json``); any other judge uses its own tag, so
``deepseek`` writes ``*_judge_deepseek.json`` next to it. The D6 relabel backs
the step-1 file up as ``*_judge_<file_tag>_legacy.json``.

progress.json
-------------
Per-judge state lives under ``judges.<tag>`` (status, stage, judged, total,
accuracy, model, eval_file, step1_done, ...). ``gpt4omini`` additionally keeps
the flat ``judge_*`` fields the first grid runners wrote, and a reader falls
back to them, so cells judged before this registry existed stay valid.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class JudgeSpec:
    tag: str
    model: str
    file_tag: str


JUDGES: Dict[str, JudgeSpec] = {
    "deepseek": JudgeSpec("deepseek", "deepseek/deepseek-v4.1-flash", "deepseek"),
    "gpt4omini": JudgeSpec("gpt4omini", "openai/gpt-4o-mini-2024-07-18", "remote"),
}
PRIMARY_JUDGE = "gpt4omini"
SECONDARY_JUDGE = "deepseek"

# Flat progress.json keys the first runners wrote for the (then only) gpt-4o-mini judge.
LEGACY_JUDGE = "gpt4omini"
LEGACY_PROGRESS_KEYS: Dict[str, str] = {
    "status": "judge_status",
    "stage": "judge_stage",
    "judged": "judged",
    "total": "judge_total",
    "accuracy": "judge_accuracy",
    "model": "judge_model",
    "step1_done": "judge_step1_done",
    "message": "judge_message",
    "started_at": "judge_started_at",
    "ended_at": "judge_ended_at",
    "elapsed_s": "judge_elapsed_s",
    "eval_file": "eval_file",
}


def judge_spec(tag: str) -> JudgeSpec:
    try:
        return JUDGES[tag]
    except KeyError:
        raise ValueError(f"unknown judge tag {tag!r} (valid: {', '.join(JUDGES)})") from None


def eval_filename(namespace: str, tag: str) -> str:
    """``evaluation_results_<ns>_judge_<file_tag>.json`` for a judge tag."""
    return f"evaluation_results_{namespace}_judge_{judge_spec(tag).file_tag}.json"


DEEPSEEK_PROVIDER = "deepseek"  # OpenRouter provider slug of DeepSeek's first-party endpoint
# Reasoning plus the JSON verdict. At 4096 a few records reasoned past the cap
# every time and came back with an empty reply; a verdict that fits in 4096 is
# unchanged by the higher cap.
DEEPSEEK_MAX_TOKENS = 8192


def is_deepseek(model: str) -> bool:
    return "deepseek" in str(model or "").lower()


def judge_request_kwargs(model: str) -> dict:
    """Keyword arguments for ``chat.completions.create`` with this judge, incl. ``max_tokens``.

    Judges answer in a short JSON verdict, so the budget is 200 tokens. DeepSeek
    v4.1 flash is run with reasoning ON and a ``DEEPSEEK_MAX_TOKENS`` budget: with reasoning
    off it often graded the gold against the evidence instead of the prediction
    (61% agreement with rule-checkable truth on the human sample, against 87%
    with reasoning on; kappa 0.70 with gpt-4o-mini), and with reasoning on but
    200 tokens the reasoning used up the budget and the reply came back empty.
    """
    if is_deepseek(model):
        return {"max_tokens": DEEPSEEK_MAX_TOKENS, "extra_body": {
            "reasoning": {"enabled": True},
            # OpenRouter's default routing spread one batch of requests over five
            # hosts, some quantised (fp4/fp8), and the slow ones held replies for
            # minutes; pinned to DeepSeek's own endpoint (full precision, 24/24
            # parseable, median 2 s on real D6 requests), with no fallback, so
            # every verdict comes from the same weights.
            "provider": {"order": [DEEPSEEK_PROVIDER], "allow_fallbacks": False},
        }}
    return {"max_tokens": 200}


# Wall-clock limit of one judge request. The client's read timeout does not
# bound it: OpenRouter keeps a slow request's connection alive with padding
# bytes, so a reply stuck reasoning held a request for over half an hour.
JUDGE_DEADLINE_S = 120


def judge_create(client, deadline_s: float = JUDGE_DEADLINE_S, **kwargs):
    """``client.chat.completions.create(**kwargs)``, abandoned after ``deadline_s``.

    Raises TimeoutError past the deadline, so the caller's retry resends the
    request; the abandoned call finishes in a daemon thread and is discarded.
    """
    box: dict = {}

    def run() -> None:
        try:
            box["resp"] = client.chat.completions.create(**kwargs)
        except BaseException as exc:  # handed to the caller
            box["exc"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(deadline_s)
    if t.is_alive():
        raise TimeoutError(f"judge request exceeded {deadline_s:.0f} s")
    if "exc" in box:
        raise box["exc"]
    return box["resp"]


def judge_state(progress: Mapping, tag: str) -> dict:
    """The progress.json state of one judge, with the legacy fallback for gpt4omini."""
    judges = progress.get("judges") if isinstance(progress, Mapping) else None
    state = dict((judges or {}).get(tag) or {})
    if tag == LEGACY_JUDGE:
        legacy_model = progress.get("judge_model")
        # the flat fields describe gpt-4o-mini only (the one judge the first runners knew)
        if legacy_model in (None, "", JUDGES[LEGACY_JUDGE].model):
            for key, flat in LEGACY_PROGRESS_KEYS.items():
                if flat in progress and progress[flat] is not None:
                    state[key] = progress[flat]
    return state


# ---------------------------------------------------------------------------
# agreement statistics
# ---------------------------------------------------------------------------

def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa for two raters over any label set (nan when undefined)."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    labels = sorted(set(a) | set(b), key=str)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((sum(1 for x in a if x == lab) / n) * (sum(1 for y in b if y == lab) / n) for lab in labels)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def agreement(a: Sequence, b: Sequence) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else float("nan")


def fmt(x: Optional[float], digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{digits}f}"


def majority(values: Iterable) -> Optional[object]:
    """Strict-majority value (None on a tie or with no values)."""
    vals: List = [v for v in values if v is not None]
    if not vals:
        return None
    counts: Dict = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    top = [v for v, c in counts.items() if c == best]
    return top[0] if len(top) == 1 else None
