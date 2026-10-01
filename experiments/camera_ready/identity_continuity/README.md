# Cross-session identity-continuity ablation

Holds each item's BM25 retrieval pool byte-fixed (same sessions, sentences, gold,
order and token count) and varies only whether an agent keeps one label across
sessions:

- `raw`: the corpus as released;
- `sham`: every agent gets one global single-token pseudonym (the renaming control);
- `break`: query and gold keep `sham`'s mapping, each non-gold session is remapped
  independently, so cross-session identity is destroyed and nothing else is.

`raw − sham` bounds the renaming artefact; `break − sham` is the effect. The
paired bootstrap is clustered by ego, over (mapping seed, item) cells.

No model is called; everything runs on the released dataset.

```bash
export MEMARENA_DATASET_DIR=/path/to/memarena_l          # default: data/benchmark
export MEMARENA_TOKEN_CACHE=out/identity_continuity/session_tokens.json
python experiments/camera_ready/identity_continuity/build_length_cache.py   # ~20 s, Qwen3-8B tokenizer
python experiments/camera_ready/identity_continuity/e6_identity.py \
    --budget 100 --seeds 5 --out out/identity_continuity/identity_results_bm25.json   # ~10-20 min
python experiments/camera_ready/identity_continuity/e6_analyze.py \
    --data out/identity_continuity/identity_results_bm25.json
```

The paper table is rendered from that results file by
`python -m memarena.figures.gen_identity_continuity_table`.
