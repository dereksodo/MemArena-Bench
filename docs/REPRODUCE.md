# Reproducing MemArena

Detailed instructions behind the [README](../README.md): the grid runners and judges used for the
paper, manual serving of readers and memory backends, regenerating every table and figure, building
new worlds with MASim, and auxiliary tools.

- [1. Run the main grid](#1-run-the-main-grid)
- [2. Serve readers and memory backends manually](#2-serve-readers-and-memory-backends-manually)
- [3. Regenerate the paper's tables and figures](#3-regenerate-the-papers-tables-and-figures)
- [4. Generate a new world with MASim](#4-generate-a-new-world-with-masim)
- [5. Tools](#5-tools)
- [6. D6 scoring and item construction](#6-d6-scoring-and-item-construction)

## 1. Run the main grid

Four scripts run the main grid one cell (reader × backend × seed) at a time, each cell on a single GPU, with the paper's settings. They need Docker with GPU access, the Python environment above and the dataset (downloaded on first use if `data/benchmark` is missing).

| Script | What it does |
|---|---|
| `scripts/test_cell.sh -model M -backend B -gpu G -seed S [-limit N] [-dataset DIR] [-out DIR] [-force]` | Answers one cell: starts SGLang on GPU `G`, prepares memory (a Memobase v0.0.42 stack per reader × seed, or the one MemSearch cache shared by all cells), runs `eval.cli` with `--fail-on-thinking`, and removes everything it started, also on error or Ctrl-C. No judging. |
| `scripts/test.sh -seed S -gpu G [-limit N]` | All 25 cells of one seed on one GPU (models outer, backends inner); continues past failures and prints a summary. |
| `scripts/judge_cell.sh -model M -backend B -seed S [-judge TAG] [-limit N] [-workers N] [-force]` | Judges one answered cell as the paper did, with judge `TAG` (below) on OpenRouter: `scripts/llmjudge.py` (512 requests in flight, `-concurrency`), then the D6 relabel `scripts/rerun_d6_judge.py` (128 in flight, `-workers`). Reads `OPENROUTER_API_KEY` from the environment or `.env`. No GPU. |
| `scripts/judge.sh -seed "S ..." [-judge TAG] [-parallel N] [-limit N]` | `judge_cell.sh` over all models × backends of one or more seeds, `N` cells at a time (default 4); summary table at the end. |

Models: `0_6b llama3b 7b 8b 32b`. Backends: `vanilla rag memobase memsearch oracle` (`rag` is BM25, `inmem` in file names). Seeds: `s1` (T=0.0) and `s2 s3 s4` (T=0.3; the paper's mean ± std). Every script takes `-dry-run` (print the commands, run nothing) and `-help`.

```bash
scripts/test_cell.sh -model 8b -backend oracle -gpu 0 -seed s2 -limit 5   # smoke test, a few minutes
scripts/test.sh -seed s2 -gpu 0 &  scripts/test.sh -seed s3 -gpu 1 &  scripts/test.sh -seed s4 -gpu 2 &
scripts/judge.sh -seed s2,s3,s4                     # primary judge (gpt-4o-mini)
scripts/judge.sh -seed s2,s3,s4 -judge deepseek     # secondary judge, for agreement
python3 scripts/judge_agreement.py --runs-dir out/runs   # agreement / Cohen's kappa, gpt4omini vs deepseek
```

Judges (`-judge`, defined in `memarena/judges.py` and `scripts/grid_common.sh`). Each judge has its own files, so both can be run on the same cells, even at the same time; neither touches the other's output or progress.

| Tag | Model | Role | Eval file | Log, manifest |
|---|---|---|---|---|
| `gpt4omini` (default) | `openai/gpt-4o-mini-2024-07-18` | primary (the submitted paper's judge) | `evaluation_results_<ns>_judge_remote.json` (the historical name) | `logs/judge.log`, `logs/llmjudge_manifest.json` |
| `deepseek` | `deepseek/deepseek-v4.1-flash` | secondary | `evaluation_results_<ns>_judge_deepseek.json` | `logs/judge_deepseek.log`, `logs/llmjudge_manifest_deepseek.json` |

Both use the same prompts, temperature 0 and parsing. DeepSeek is called with reasoning on (OpenRouter `reasoning: {enabled: true}`), an 8,192-token budget, and OpenRouter routing pinned to DeepSeek's own endpoint with no fallback; with reasoning off it often graded the gold instead of the prediction, and smaller budgets left some replies empty. The D6 relabel backs each eval file up once as `*_legacy.json` (its state before the relabel).

`scripts/judge_agreement.py --runs-dir DIR [--judges A B] [--seeds s2,s3,s4]` pairs the two judges' records of every cell both finished and reports agreement and Cohen's κ per cell, per paper dimension and pooled: binary correct/incorrect on the records the LLM judge scored (deterministically scored records such as `mcq_exact`, `refusal_detection` or cloze are excluded and listed with counts), and for D6 the 5-label rubric and the binary leak (`DISCLOSE_CORRECT`). It writes `DIR/_judge_agreement/agreement_<A>_vs_<B>.{md,csv}`. Judge-vs-human κ on the 500-item calibration sample: `python3 -m memarena.human_calibration.rejudge_sample --judge TAG` re-judges `memarena/human_calibration/sample.json` (local) with the gold passed, then `python3 -m memarena.human_calibration.analyze_judge_human --pool memarena/human_calibration/sample_judge_TAG.json` compares it with `labels.json` (majority of the annotators).

Output of a cell, under `out/runs` (or `-out`):

```text
<out>/<backend>/<model>/<seed>/            e.g. out/runs/rag/8b/s2/
  progress.json                            status, stage, items, ETA (below)
  args.json                                every setting and the exact eval.cli command (no keys)
  <system>/answer_results_<ns>.json        answers; <system> is vanilla|oracle|inmem|memory_cache
  <system>/evaluation_results_<ns>_judge_deepseek.json after judge_cell.sh -judge deepseek
  <system>/evaluation_results_<ns>_judge_remote.json   after judge_cell.sh -judge gpt4omini
  memory/                                  memobase: cache JSONL, summary, the stack's database
  logs/  sglang.log  memory_build.log  answer.log  judge_deepseek.log  judge.log  test_cell.log ...
  limitN/                                  -limit N runs; never counted as the finished cell
<out>/_shared/memsearch/                   the MemSearch cache shared by every reader and seed
<out>/_grid_logs/                          summaries of test.sh and judge.sh (+ per-cell console output of judge.sh -parallel)
```

`progress.json` holds `status` (`running`, `done`, `failed` with `message`), `stage` (`1/4`…`4/4`: sglang, memory, answer, teardown) with `stage_name`, `items_done`/`items_total`/`items_unit` (memory-build days or answered items), `eta_seconds`/`eta`, `started_at`, `gpu`, a per-stage history with elapsed times, and after judging `judges.<tag>` per judge: `status`, `stage`, `judged`/`total`, `accuracy`, `model`, `eval_file`, timings (for `gpt4omini` also mirrored in the flat `judge_status`, `judge_accuracy`, … fields of the first runners). A cell whose status is `done` is skipped unless `-force`; `judge_cell.sh` skips a cell that judge has finished, per judge. Ports are `20000 + 100·GPU + offset` (`MEMARENA_PORT_BASE`), and containers carry a per-cell name prefix, so cells on different GPUs run side by side; `scripts/test_cell.sh -help` lists the other overrides.

## 2. Serve readers and memory backends manually

The paper's grid is 5 readers × 5 memory conditions × 3 seeds:

```text
readers:   Qwen3-0.6B, Llama-3.2-3B, Mistral-7B-Instruct-v0.3, Qwen3-8B, Qwen3-32B-AWQ
backends:  vanilla (recent context), inmem (BM25 RAG), memobase, memsearch, oracle (evidence only)
seeds:     s2, s3, s4   (answer temperature 0.3)
```

### Readers

Put checkpoints under `~/models/{0_6b,llama3b,7b,8b,32b}` (override with `MODEL_ROOT`) and start SGLang:

```bash
MODEL_ROOT=~/models ./start_sglang_servers.sh --image lmsysorg/sglang:latest start 0_6b llama3b 7b 8b 32b
./start_sglang_servers.sh status
```

| Reader tag | Model | Endpoint | GPUs |
|---|---|---|---|
| `0_6b` | Qwen/Qwen3-0.6B | http://localhost:16000 | 0 |
| `llama3b` | meta-llama/Llama-3.2-3B-Instruct | http://localhost:16001 | 1 |
| `7b` | mistralai/Mistral-7B-Instruct-v0.3 | http://localhost:16002 | 2,3 |
| `8b` | Qwen/Qwen3-8B | http://localhost:16003 | 4,5 |
| `32b` | Qwen/Qwen3-32B-AWQ | http://localhost:16004 | 6,7 |

SGLang runs with `--context-length 16384 --mem-fraction-static 0.85 --max-running-requests 64`; multi-GPU readers use data parallelism.

### Memory services

Memobase (pinned `v0.0.42`) runs as one isolated stack per reader; reserve ~150 GB of disk for images, volumes and caches (`$MEMARENA_MEMORY_SERVICES_ROOT`, default `~/memarena-memory-services`):

```bash
scripts/setup_memory_backends.sh restart 0_6b llama3b 7b 8b 32b
scripts/check_service_health.sh
```

MemSearch runs in-process on milvus-lite and embeds with `nomic-embed-text` through Ollama (`ollama pull nomic-embed-text`; point `OLLAMA_HOST` or `MEMSEARCH_EMBED_BASE_URL` at it). MemOS (`v2.0.13`, same script) is supported for ablations.

### Run the matrix

The per-cell runners in [section 1](#1-run-the-main-grid) replace this path for the main grid.

```bash
scripts/run_eval_matrix.sh --paper-full --run-dir data/benchmark --out-dir out/accuracy_memarena_l
```

`--paper-full` runs `vanilla,inmem,oracle,memobase,memos`. The paper's MemSearch cells were produced in two phases instead: one reader-independent memory cache built with `scripts/reproduce/build_memory_cache.py --system memsearch`, then all 15 cells answered from it with `scripts/run_memsearch_phase2.sh`. The cache used for the paper was not archived, so a re-run rebuilds it.

Useful commands: `python scripts/summarize_eval_progress.py out --paths-output progress.json` (find finished cells), `scripts/llmjudge.py --from-progress-json progress.json ...` (judge them), `./start_sglang_servers.sh logs 8b`.

## 3. Regenerate the paper's tables and figures

Every table and figure in the paper is regenerated from the per-item judge outputs with one command. The result files behind the paper are not distributed; point `MEMARENA_RESULTS_DIR` at the outputs of your own runs:

```bash
export MEMARENA_RESULTS_DIR=/path/to/results      # holds out/ and MASim/runs/ from the runs above
python scripts/reproduce_figures.py --list
python scripts/reproduce_figures.py --all --out-dir out/paper_artifacts
python scripts/reproduce_figures.py --check path/to/paper   # regenerate and diff against a paper checkout
```

`experiments_index.csv` maps each cell to its result file under `$MEMARENA_RESULTS_DIR`. To build the main-grid tables and figures from the grid runners' tree instead, pass `--runs-dir out/runs` (or set `MEMARENA_RUNS_DIR`) and choose the judge with `--judge gpt4omini|deepseek` (`MEMARENA_JUDGE`, default `gpt4omini`); cells the judge has not finished are left out. Artifacts built from ablation rows, latency logs or the dataset keep their own inputs. The realism tables also read four external corpora from `$MEMARENA_EXTERNAL_CORPORA` (REALTALK, DailyDialog, PERSONA-CHAT, LoCoMo; see `memarena/realism.py` for the layout and download URLs). The identity-continuity ablation calls no model and runs from the dataset alone; see `experiments/camera_ready/identity_continuity/README.md`.

## 4. Generate a new world with MASim

Smaller worlds reproduce the full pipeline end to end on one machine. For 5 agents × 10 days:

```bash
export RUN=$PWD/MASim/runs/5a10d
python run_masim.py --config MASim/configs/memarena_5a10d_5k.yaml --sglang-url http://localhost:16000 --output "$RUN" --overwrite
scripts/run_eval_matrix.sh --paper-full --run-dir "$RUN" --out-dir out/accuracy_5a10d
python scripts/summarize_eval_progress.py out/accuracy_5a10d --paths-output progress_5a10d.json
python scripts/llmjudge.py --from-progress-json progress_5a10d.json --run-dir "$RUN" \
    --judge-preset openrouter --judge-model openai/gpt-4o-mini --concurrency 4 --keep-going
```

`MASim/configs/memarena_10a5d_5k.yaml` gives a 10-agent × 5-day world; `memarena_l.yaml` is the configuration of MemArena-L.

To rebuild only the D6 items of an existing run (the LLM settings come from the run's `effective_config.yaml`; `--dry-run` uses fixed category phrases instead):

```bash
python -m MASim.ground_truth.d4_permission --run-dir "$RUN" --out "$RUN/eval_instances/d4_permission.jsonl" --overwrite \
    --config "$RUN/effective_config.yaml"
```

## 5. Tools

```bash
python -m memarena.tools.validate_croissant docs/memarena-croissant.json
python -m memarena.tools.scan_corpus_non_ascii path/to/corpus.jsonl
python -m memarena.tools.bundle_memarena --version 1.0.0 --out out/bundle/
python memarena/human_calibration/server.py --port 8080    # annotation UI; serves a sample.json built by build_human_calibration.py
```

## 6. D6 scoring and item construction

**Scoring D6.** F1_PU is the harmonic mean of privacy (1 − share of DENY items whose response discloses the protected fact) and utility (share of ALLOW items that disclose it), both judged with the five-label rubric in `eval/src/scoring.py`. Every item carries a protected fact; the generator's 56 fact-less behavioural probes (known-requester ALLOW, anonymous-querier DENY) are not released, since a fact-level judge cannot score them. The canonical implementation is `eval/src/permission_metrics.py`.

**How D6 items are built.** `MASim/ground_truth/d4_permission.py` builds them from the injected permission and autonomous-privacy turns: every human agent outside a fact's session is a candidate asker, `friends_only` access follows the Dunbar graph (intimate or close layer), and a 200-item draw is sampled 40% DENY / 40% ALLOW / 20% public; dropping the 56 fact-less probes leaves the released 73 DENY, 31 friends-only ALLOW and 40 public ALLOW items. Each fact question names the fact's topic category but none of its content (checks in `MASim/ground_truth/d4_permission_checks.py`), the ALLOW and DENY items of a fact share one template pool, and every item carries `metadata.query_timestamp` after the last session.

The world was generated by a locally served open-weight Qwen3-family model; its exact checkpoint was not recorded (see the paper's limitations section).

The world was generated by a locally served open-weight Qwen3-family model; its exact checkpoint was not recorded (see the paper's limitations section).
