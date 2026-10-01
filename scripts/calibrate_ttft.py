"""Cold-prefill TTFT and decode calibration for one reader (App. C latency model).

With the prefix cache off, a dense reader's time-to-first-token depends only
on the prompt length N_p, so one curve per reader, TTFT_r(N_p), replaces
per-backend timing runs. Every request here is therefore a cold prefill:
the server runs with --disable-radix-cache, each prompt starts with a unique
nonce, and usage.prompt_tokens_details.cached_tokens is logged so a hit
would be visible.

TTFT is measured exactly as in eval/src/answering.py (request start to the
first non-empty streamed content chunk). Each request also decodes a fixed
MAX_TOKENS with ignore_eos, which gives the per-token decode time at that
context length.

Prompt lengths are set in characters; a linear chars->tokens map is fitted
from the warm-up requests (which are discarded), and the fit downstream uses
the server-reported prompt_tokens, not the targets.

Usage (on the Spark host, sglang already serving the reader):
    python calibrate_ttft.py --bundle prompt_bundle.json --reader 32b \
        --served-model Qwen/Qwen3-32B-AWQ --endpoint http://localhost:17100/v1 \
        --out out/ttft_calib/calib_32b.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
import uuid

import numpy as np
from openai import OpenAI

TARGETS = [192, 256, 384, 512, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000]
REPS = 5
MAX_TOKENS = 64
WARMUP_CHARS = [1500, 12000, 30000, 45000]   # also fits the chars->tokens map


def build_messages(bundle: dict, ctx_chars: int, rng: random.Random) -> list[dict]:
    pool = bundle["context_pool"]
    i = rng.randrange(len(pool))
    kept, used = [], len('{"conversation_context": []}')
    while used < ctx_chars:
        m = pool[i % len(pool)]
        kept.append(m)
        used += len(json.dumps(m, ensure_ascii=False)) + 2
        i += 1
    payload = rng.choice(bundle["payloads"])
    messages = [{"role": "system", "content": f"[request {uuid.uuid4().hex}]\n{bundle['system_prompt']}"}]
    if kept:   # answering.py also omits the context turn when there is no context
        messages += [
            {"role": "user", "content": json.dumps({"conversation_context": kept}, ensure_ascii=False)},
            {"role": "assistant", "content": bundle["ack"]},
        ]
    return messages + [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def one_request(client: OpenAI, model: str, messages: list[dict]) -> dict:
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True},
    )
    t_first, usage = None, None
    for chunk in stream:
        if chunk.choices:
            if (chunk.choices[0].delta.content or "") and t_first is None:
                t_first = time.perf_counter()
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
    t_end = time.perf_counter()
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    return {
        "prompt_chars": sum(len(m["content"]) for m in messages),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "cached_tokens": cached,
        "ttft_ms": None if t_first is None else round((t_first - t0) * 1000.0, 2),
        "decode_ms": None if t_first is None else round((t_end - t_first) * 1000.0, 2),
        "total_ms": round((t_end - t0) * 1000.0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--targets", default=",".join(map(str, TARGETS)))
    args = ap.parse_args()

    bundle = json.load(open(args.bundle))
    client = OpenAI(base_url=args.endpoint, api_key="EMPTY", timeout=600)
    rng = random.Random(f"ttft-calib-{args.reader}")
    out = open(args.out, "w")

    def emit(rec: dict) -> None:
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[{rec['phase']:6s}] target={rec.get('target_tokens')} N_p={rec['prompt_tokens']} "
              f"cached={rec['cached_tokens']} ttft={rec['ttft_ms']}ms decode={rec['decode_ms']}ms "
              f"n_c={rec['completion_tokens']}", flush=True)

    # Warm-up: kernel JIT for a spread of prefill shapes; fits chars -> tokens.
    ctx_len, n_tok = [], []
    for c in WARMUP_CHARS:
        msgs = build_messages(bundle, c, rng)
        r = one_request(client, args.served_model, msgs)
        emit({"reader": args.reader, "phase": "warmup", "target_tokens": None, **r})
        ctx_len.append(len(msgs[1]["content"]) if len(msgs) == 4 else 0)
        n_tok.append(r["prompt_tokens"])
    slope, intercept = np.polyfit(ctx_len, n_tok, 1)   # tokens per context char, fixed tokens

    plan = [(t, k) for t in map(int, args.targets.split(",")) for k in range(args.reps)]
    random.Random(f"order-{args.reader}").shuffle(plan)
    for idx, (target, rep) in enumerate(plan):
        ctx_chars = max(0, int((target - intercept) / slope))
        r = one_request(client, args.served_model, build_messages(bundle, ctx_chars, rng))
        emit({"reader": args.reader, "phase": "timed", "order": idx, "target_tokens": target,
              "rep": rep, **r})
    out.close()


if __name__ == "__main__":
    main()
