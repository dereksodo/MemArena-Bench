"""Build the prompt material for the cold-prefill TTFT calibration (App. C).

Prefill time on a dense reader depends on the number of prompt tokens, not on
their content, so the calibration only needs prompts of controlled length. We
still build them from benchmark text in the answer-phase message layout
(system -> conversation_context -> ack -> question payload, see
eval/src/answering.py) so the chat template and token mix match evaluation.

Output: out/ttft_calib/prompt_bundle.json
    system_prompt    ANSWERING_SYSTEM_PARTS joined as in _build_llm_system_prompt
    ack              the assistant acknowledgement turn
    context_pool     corpus turns as conversation_context entries
    payloads         answer-phase payloads for real benchmark questions
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

from MASim.prompts import ANSWERING_SYSTEM_PARTS

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "benchmark"
OUT = REPO / "out" / "ttft_calib" / "prompt_bundle.json"

ACK = "I have read and memorized the full conversation context above. Ready for questions."
N_TURNS = 4000     # ~5 x 12K-token prompts of distinct text
N_PAYLOADS = 40


def context_pool(rng: random.Random) -> list[dict]:
    sessions = []
    with gzip.open(BENCH / "corpus_sessions.jsonl.gz", "rt") as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    rng.shuffle(sessions)
    pool = []
    for s in sessions:
        for t in s.get("turns") or []:
            pool.append({
                "msg_id": t.get("turn_id"),
                "thread_id": s.get("session_id"),
                "occur_ts": s.get("start_time"),
                "user_id": t.get("speaker_id"),
                "text": t.get("text"),
            })
        if len(pool) >= N_TURNS:
            break
    return pool[:N_TURNS]


def payloads(rng: random.Random) -> list[dict]:
    rows = []
    for p in sorted((BENCH / "eval_instances").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    rng.shuffle(rows)
    out = []
    for r in rows[:N_PAYLOADS]:
        out.append({
            "task": "memarena_answer",
            "question_type": r.get("question_type") or "open",
            "question": r["query"],
            "options": r.get("options"),
            "output_format": {"answer": "string", "reason": "short string"},
        })
    return out


def main() -> None:
    rng = random.Random(20260929)
    bundle = {
        "system_prompt": "\n".join(ANSWERING_SYSTEM_PARTS),
        "ack": ACK,
        "context_pool": context_pool(rng),
        "payloads": payloads(rng),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, ensure_ascii=False))
    chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in bundle["context_pool"])
    print(f"wrote {OUT}: {len(bundle['context_pool'])} turns ({chars} chars), "
          f"{len(bundle['payloads'])} payloads")


if __name__ == "__main__":
    main()
