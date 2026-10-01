# MASim: Multi-Agent Simulation for Ego-Centric Memory Evaluation

MASim is the data generation and evaluation framework for **MemArena**, an ego-centric memory benchmark. It orchestrates multi-agent dialogues over Dunbar-layered social graphs, injects controlled information asymmetries, and produces ground-truth evaluation instances across 7 memory dimensions.

## Architecture

```
                 ┌─────────────┐
                 │  YAML Config │
                 └──────┬──────┘
                        ▼
              ┌──────────────────┐
              │   Orchestrator   │  pipeline/orchestrator.py
              └──────┬───────────┘
                     │
    ┌────────┬───────┼────────┬──────────┐
    ▼        ▼       ▼        ▼          ▼
 Social   Persona  Event   Dialogue   Ego
 Graph    Factory  Factory  Engine    Projection
 (core/)  (gen/)   (gen/)   (core/)   (core/)
    │        │       │        │          │
    └────────┴───────┴────┬───┴──────────┘
                          ▼
              ┌──────────────────┐
              │  Dialogue Corpus │
              └──────┬───────────┘
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
 Conflict      Permission      Coreference
 Injector      Injector        Tracker
 (gen/)        (gen/)          (gen/)
                     │
                     ▼
              ┌──────────────────┐
              │  GT Extractor    │  ground_truth/
              │  D1–D7 modules   │
              └──────┬───────────┘
                     ▼
              ┌──────────────────┐
              │  Eval Instances  │  JSONL output
              │  + Ego Contexts  │
              └──────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install networkx tiktoken rich openai pyyaml numpy

# Dry-run MemArena-L without LLM calls
python -m MASim run --config MASim/configs/memarena_l.yaml --output runs/l_dryrun/ --dry-run

# With a running vLLM/SGLang server
python run_masim.py \
  --output runs/l

# Inspect results
python -m MASim stats --run runs/l/run/
python -m MASim validate --run runs/l/run/
```

## Directory Structure

```
MASim/
├── configs/                  # YAML configurations
│   ├── memarena_base.yaml    #   Shared MemArena-L generation settings
│   ├── memarena_l.yaml       #   Public MemArena-L configuration
│   └── graph_presets/        #   Dunbar-WS, Erdős-Rényi, Barabási-Albert
├── core/                     # Core simulation
│   ├── schema.py             #   All dataclasses (PersonaCard, DialogueTurn, EvalInstance, ...)
│   ├── social_graph.py       #   Dunbar-layered graph construction (WS/ER/BA)
│   ├── agent.py              #   EntityAgent: persona + memory + knowledge
│   ├── world_broadcaster.py  #   Event visibility by category (GLOBAL/COMMUNITY/DYADIC/PRIVATE)
│   ├── dialogue_engine.py    #   Turn-by-turn LLM dialogue generation
│   ├── ego_projection.py     #   π(u_i, D): ego-centric corpus filtering
│   └── knowledge_state.py    #   Per-agent fact tracking with conflict detection
├── generation/               # Data generation pipeline
│   ├── persona_factory.py    #   LLM-based diverse persona generation
│   ├── event_factory.py      #   World event generation with visibility assignment
│   ├── conflict_injector.py  #   Controlled contradiction injection (D1 GT)
│   ├── permission_injector.py#   Privacy scenario injection (D4 GT)
│   ├── coreference_tracker.py#   Cross-session reference tracking (D2 GT)
│   └── multimodal/           #   MAGID-style multimodal augmentation
│       ├── scanner.py        #     Identify augmentable utterances
│       ├── image_generator.py#     Text-to-image (SDXL/FLUX interface)
│       ├── audio_generator.py#     TTS with persona voice profiles
│       └── quality_filter.py #     CLIP scoring + aesthetic filtering
├── inference/                # LLM backend
│   ├── llm_client.py        #   Thread-safe OpenAI-compatible client (vLLM/SGLang)
│   └── batch_scheduler.py   #   Wave-parallel batch execution
├── ground_truth/             # Ground truth extraction (7 dimensions)
│   ├── gt_extractor.py       #   Coordinator for all dimensions
│   ├── d1_conflict.py        #   Conflict preservation
│   ├── d2_anaphora.py        #   Cross-session anaphora resolution
│   ├── d3_confabulation.py   #   Confabulation resistance (fabricated queries)
│   ├── d4_permission.py      #   Permission management (paper D6); CLI: python -m MASim.ground_truth.d4_permission
│   ├── d4_permission_checks.py #  Query checks for D6 fact items
│   ├── d5_cloze.py           #   Cloze-deletion fidelity
│   ├── d6_metadata.py        #   Metadata completeness
│   └── d7_qa.py              #   Standard QA + temporal reasoning
├── evaluation/               # LLM-as-a-Judge framework
│   ├── judge.py              #   Rubric-based scoring engine
│   ├── rubrics/              #   5-level Prometheus rubrics (D1, D3, D4, D5)
│   ├── metrics.py            #   CPS, MCE, PU-AUC, Memory Lift, Set F1
│   ├── bias_mitigation.py    #   Position swap, reference support
│   └── nugget_scorer.py      #   Atomic nugget decomposition + coverage
├── pipeline/                 # Orchestration + CLI
│   ├── orchestrator.py       #   11-stage end-to-end pipeline
│   └── cli.py                #   CLI: run / evaluate / validate / stats
└── utils/
    ├── io.py                 #   JSONL/YAML/JSON I/O with numpy serialization
    ├── tokens.py             #   tiktoken counting + truncation
    └── logging.py            #   Rich structured logging
```

## Pipeline Stages

The orchestrator runs 10 stages sequentially:

| # | Stage | Description |
|---|-------|-------------|
| 1 | **Build social graph** | Dunbar-layered Watts-Strogatz (or ER/BA baseline) |
| 2 | **Generate personas** | LLM-created diverse persona cards with Dunbar layer assignment |
| 3 | **Generate events** | World events with category-based visibility (GLOBAL/COMMUNITY/DYADIC/PRIVATE) |
| 4 | **Run simulation** | Wave-parallel dialogue generation across agent pairs |
| 5 | **Inject conflicts** | Controlled contradictions with (fact, source, time) ground truth |
| 6 | **Inject permissions** | Privacy scenarios with access control ground truth |
| 7 | **Track coreferences** | Cross-session entity reference annotation |
| 8 | **Extract ground truth** | Generate eval instances for all 7 dimensions (D1–D7) |
| 9 | **Compute ego projections** | Per-agent ego-centric memory views |
| 10 | **Write outputs** | JSONL corpus, eval instances, graph edges, reports |

## Evaluation Dimensions

| Dim | Name | Metric | What it tests |
|-----|------|--------|---------------|
| D1 | Conflict Preservation | CPS (harmonic mean) | Detect and preserve contradictory facts from different sources |
| D2 | Cross-Session Anaphora | Nugget Recall | Resolve references to entities across conversation sessions |
| D3 | Confabulation Resistance | PU-AUC | Abstain on queries about non-existent events |
| D4 | Permission Management | Accuracy | Respect privacy levels (private / friends_only / public) |
| D5 | Cloze Fidelity | Memory Lift | Reconstruct deleted dialogue spans from memory |
| D6 | Metadata Completeness | Set F1 | Exhaustive retrieval of entity attributes |
| D7 | Standard QA | Rubric score | Factual recall + temporal reasoning |

## Configuration

All parameters are controlled via YAML. Key sections:

```yaml
graph:
  n_agents: 50
  model: watts_strogatz       # or erdos_renyi, barabasi_albert
  layers: [...]               # Dunbar layer sizes + WS params

llm:
  endpoint: "http://127.0.0.1:8000/v1"
  model: "qwen3"
  concurrency: 32

n_events: 100
max_sessions: 200
conflict_inject_prob: 0.1
permission_inject_prob: 0.08
```

See `configs/memarena_l.yaml` and `configs/memarena_base.yaml` for all options.

## Output Format

A completed run produces:

```
output_dir/
├── corpus_sessions.jsonl      # All dialogue sessions with turns
├── events.jsonl               # World events with visibility masks
├── graph_edges.jsonl          # Social graph edge list
├── ego_projections.json       # Per-agent ego memory stats
├── pipeline_report.json       # Run metadata and statistics
└── eval_instances/
    ├── d1_conflict.jsonl      # One file per dimension
    ├── d2_anaphora.jsonl
    ├── d3_confabulation.jsonl
    ├── d4_permission.jsonl
    ├── d5_cloze.jsonl
    ├── d6_metadata.jsonl
    └── d7_qa.jsonl
```

Each eval instance is a JSON object with `instance_id`, `dimension`, `query`, `ground_truth`, `difficulty`, and `metadata`.

## LLM Backend

MASim uses an OpenAI-compatible API. MemArena does not start or manage the
inference server; start your preferred SGLang/vLLM server yourself, then pass
its base URL to the runner. Accepted forms include `http://localhost:16000` and
`http://localhost:16000/v1`; the default is `http://localhost:16000`.

For unattended runs, prefer the repository-root launcher:

```bash
python run_masim.py
```

If `--config` is omitted, the launcher uses `MASim/configs/memarena_l.yaml`.
For smaller integration tests, use `MASim/configs/memarena_5a10d_5k.yaml`.
The launcher writes a self-contained `effective_config.yaml`, points
`llm.endpoint` to the supplied server, waits for readiness, runs MASim, and
then runs `validate` and `stats`.
