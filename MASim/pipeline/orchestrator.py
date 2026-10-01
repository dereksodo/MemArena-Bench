"""End-to-end pipeline orchestrator: graph -> agents -> sim -> GT -> eval.

Coordinates all stages of the MASim pipeline from config to output.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from MASim.core.agent import EntityAgent
from MASim.core.dialogue_engine import DialogueEngine
from MASim.core.ego_projection import ego_project
from MASim.core.schema import (
    ActivityEntry, AgentState, ConflictGT, DialogueCorpus, DialogueTurn,
    Dimension, EncounterWindow, EvalInstance,
    Group, Location, PersonaCard, PersonRole, Session, WorldEvent,
    is_assistant,
)
from MASim.core.social_graph import GraphConfig, build_graph, graph_properties, relabel_graph
from MASim.core.world_broadcaster import WorldBroadcaster
from MASim.generation.conflict_injector import ConflictInjector, ConflictInjectorConfig
from MASim.generation.coreference_tracker import CoreferenceTracker
from MASim.generation.event_factory import EventFactory
from MASim.generation.permission_injector import (
    AutonomousPrivacyConfig, AutonomousPrivacyInjector,
    PermissionInjector, PermissionInjectorConfig,
)
from MASim.core.memory_writer import MemoryWriter
from MASim.generation.group_factory import GroupFactory
from MASim.generation.location_factory import LocationFactory
from MASim.generation.persona_factory import PersonaFactory, personas_to_slugs
from MASim.generation.schedule_factory import ScheduleFactory
from MASim.generation.social_knowledge_injector import inject_social_knowledge
from MASim.ground_truth.gt_extractor import GTExtractor
from MASim.ground_truth.query_time import query_timestamp, session_end_times
from MASim.ground_truth.query_hardener import HardeningConfig
from MASim.inference.batch_scheduler import BatchScheduler
from MASim.inference.llm_client import LLMClient, LLMClientConfig
from MASim.utils.io import read_jsonl, read_yaml, write_json_report, write_jsonl
from MASim.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Corpus token counter
# ---------------------------------------------------------------------------

def _count_corpus_tokens(sessions: List[Session]) -> int:
    """Count corpus tokens over turn text.

    We prefer the Qwen3 tokenizer used by the generator.  If it is not
    available locally or from Hugging Face, fall back to cl100k_base before
    using the old character estimate.  Counting each turn separately avoids
    adding artificial newline tokens between turns.
    """
    texts = [t.text for s in sessions for t in s.turns if t.text]
    try:
        from transformers import AutoTokenizer

        candidates: List[Path | str] = []
        if os.getenv("MEMARENA_TOKENIZER"):
            candidates.append(os.environ["MEMARENA_TOKENIZER"])
        candidates.append(Path("~/models/Qwen3-8B").expanduser())
        for candidate in candidates:
            try:
                if isinstance(candidate, Path) and not candidate.exists():
                    continue
                tok = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                return sum(len(tok.encode(text)) for text in texts)
            except Exception:
                continue
    except Exception:
        pass

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(text)) for text in texts)
    except Exception:
        # Last-resort estimate for environments without tokenizer packages.
        return int(sum(len(text) for text in texts) / 3.5)


# ---------------------------------------------------------------------------
# Stage timer
# ---------------------------------------------------------------------------

class _StageTimer:
    """Simple wall-clock timer for pipeline stages."""

    def __init__(self) -> None:
        self._timings: Dict[str, float] = {}
        self._current_name: Optional[str] = None
        self._current_start: float = 0.0

    def start(self, name: str) -> None:
        if self._current_name is not None:
            self.stop()
        self._current_name = name
        self._current_start = time.monotonic()

    def stop(self) -> None:
        if self._current_name is not None:
            elapsed = time.monotonic() - self._current_start
            self._timings[self._current_name] = round(elapsed, 2)
            log.info("  ⏱ %s finished in %.1fs", self._current_name, elapsed)
            self._current_name = None

    def to_dict(self) -> Dict[str, float]:
        if self._current_name is not None:
            self.stop()
        return dict(self._timings)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MultimodalConfig:
    """Configuration for multimodal pipeline."""
    enable_image: bool = False
    enable_audio: bool = False
    image_model: str = "~/models/FLUX.1-dev"
    image_endpoint: str = ""  # optional API endpoint
    audio_model: str = ""
    max_image_candidates: int = 500
    clip_threshold: float = 0.25


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*. Lists are replaced, not extended."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_yaml(path: Path) -> dict:
    """Load a YAML file, recursively applying any ``base:`` inheritance chain."""
    doc = read_yaml(path)
    base_name = doc.pop("base", None)
    if base_name:
        base_path = path.parent / base_name
        base_doc = _resolve_yaml(base_path)
        doc = _deep_merge(base_doc, doc)
    return doc


@dataclass
class PersonAgentConfig:
    """Configuration for person-agent dialogue sessions."""
    enabled: bool = False
    ratio: float = 1.5                     # person-agent sessions per person-person session
    activity_reflection_split: float = 0.6  # fraction that are activity narration (legacy, used when probe not configured)
    activity_reflection_probe_split: Optional[List[float]] = None  # [activity, reflection, probe] fractions
    max_turns: int = 30
    min_turns: int = 15
    probe_max_turns: int = 10
    probe_min_turns: int = 4


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    # Graph
    graph: GraphConfig = field(default_factory=GraphConfig)
    # Simulation
    n_events: int = 100
    max_sessions: int = 200               # legacy — ignored when sessions_per_day_per_agent > 0
    max_sessions_per_day: int = 200        # legacy — ignored when sessions_per_day_per_agent > 0
    sessions_per_day_per_agent: int = 20   # per-day target: n_agents * this = total sessions/day
    pp_day_proportions: Dict[str, float] = field(default_factory=lambda: {
        "facet_routine": 0.40, "event_triggered": 0.15,
        "encounter": 0.25, "remote": 0.20,
    })
    pa_day_ratio_range: tuple = (1.5, 2.0)  # PA count per day = PP * U(lo, hi)
    target_tokens: int = 0                  # 0 = use time_range; >0 = stop at token count
    max_turns_per_session: int = 20
    min_turns_per_session: int = 4
    no_token_limit: bool = False
    time_range: tuple = (0.0, 1000.0)
    # Injection
    conflict_inject_prob: float = 0.1
    max_conflicts: Optional[int] = None
    permission_inject_prob: float = 0.08
    max_permissions: Optional[int] = None
    # Autonomous privacy
    autonomous_privacy_inject_prob: float = 0.06
    max_autonomous_privacy: Optional[int] = None
    # Modality weights: probability distribution over session modalities
    online_prob: float = 0.05  # legacy; prefer modality_weights
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "face_to_face": 0.85, "voice_message": 0.05, "text_message": 0.10,
    })
    # GT & Eval
    max_instances_per_dim: int = 200
    gt_cutoff_ratio: float = 0.85  # keep 85% when a dimension hits its cap
    # LLM
    llm: LLMClientConfig = field(default_factory=LLMClientConfig)
    # Multimodal
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    # Query hardening
    hardening: HardeningConfig = field(default_factory=HardeningConfig)
    # Person-agent sessions
    person_agent: PersonAgentConfig = field(default_factory=PersonAgentConfig)
    # Scheduler-driven simulation (Phase 3 lifecycle)
    scheduler_driven: bool = True      # use encounter-based scheduler vs legacy event-driven
    base_encounter_prob: float = 0.15  # base probability for casual encounter → conversation
    # Day-batching: all sessions within the same day are batched together;
    # knowledge updates apply only at day boundaries.
    # If 0 (default), auto-computed as time_range / n_events * 2.
    day_length: float = 0.0
    # General
    seed: int = 42
    dry_run: bool = False
    memory_context_sessions: bool = False  # inject past dialogue transcripts into agent prompts
    faster_batch: bool = False       # alternate generation pipeline (--faster-batch)
    overprovision_ratio: float = 1.5  # extra pairwise configs for racing (--faster-batch)
    cutoff_ratio: float = 0.1        # fraction of n_target to keep as truncated sessions
    day_completion_ratio: float = 1.0  # keep this fraction of PP+PA sessions (1.0 = all, 0.8 = fastest 80%)
    log_level: str = "INFO"
    enable_multimodal: bool = False  # legacy flag, prefer multimodal.enable_*
    # Interest-facet sub-agents
    facets_enabled: bool = False     # enable interest-domain faceted sessions

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load config from YAML file, with optional base-config inheritance.

        If the YAML contains ``base: <filename.yaml>`` the named file is loaded
        first (resolved relative to the current file's directory) and the
        current file's keys are deep-merged on top. This keeps MemArena-L
        overrides small while inheriting shared LLM and simulator settings.
        """
        doc = _resolve_yaml(path)
        cfg = cls()

        # Graph config
        if "graph" in doc:
            cfg.graph = GraphConfig.from_dict(doc["graph"])

        # LLM config
        if "llm" in doc:
            cfg.llm = LLMClientConfig.from_dict(doc["llm"])

        # Multimodal config
        if "multimodal" in doc:
            mm = doc["multimodal"]
            cfg.multimodal = MultimodalConfig(
                enable_image=mm.get("enable_image", False),
                enable_audio=mm.get("enable_audio", False),
                image_model=mm.get("image_model", cfg.multimodal.image_model),
                image_endpoint=mm.get("image_endpoint", ""),
                audio_model=mm.get("audio_model", ""),
                max_image_candidates=mm.get("max_image_candidates", 500),
                clip_threshold=mm.get("clip_threshold", 0.25),
            )

        # Hardening config
        if "hardening" in doc:
            cfg.hardening = HardeningConfig.from_dict(doc["hardening"])

        # Person-agent config
        if "person_agent" in doc:
            pa = doc["person_agent"]
            cfg.person_agent = PersonAgentConfig(
                enabled=pa.get("enabled", False),
                ratio=pa.get("ratio", 1.5),
                activity_reflection_split=pa.get("activity_reflection_split", 0.6),
                activity_reflection_probe_split=pa.get("activity_reflection_probe_split"),
                max_turns=pa.get("max_turns", 30),
                min_turns=pa.get("min_turns", 15),
                probe_max_turns=pa.get("probe_max_turns", 10),
                probe_min_turns=pa.get("probe_min_turns", 4),
            )

        # Simulation params
        for key in [
            "n_events", "max_sessions", "max_sessions_per_day",
            "sessions_per_day_per_agent", "target_tokens",
            "max_turns_per_session", "min_turns_per_session",
            "no_token_limit", "seed", "dry_run", "log_level", "enable_multimodal",
            "conflict_inject_prob", "max_conflicts",
            "permission_inject_prob", "max_permissions",
            "autonomous_privacy_inject_prob", "max_autonomous_privacy",
            "max_instances_per_dim", "gt_cutoff_ratio",
            "scheduler_driven", "base_encounter_prob", "day_length",
            "day_completion_ratio", "memory_context_sessions", "facets_enabled",
        ]:
            if key in doc:
                setattr(cfg, key, doc[key])

        if "pp_day_proportions" in doc:
            cfg.pp_day_proportions = doc["pp_day_proportions"]
        if "pa_day_ratio_range" in doc:
            cfg.pa_day_ratio_range = tuple(doc["pa_day_ratio_range"])

        if "time_range" in doc:
            cfg.time_range = tuple(doc["time_range"])

        # Modality weights (new) vs online_prob (legacy backward compat)
        if "modality_weights" in doc:
            cfg.modality_weights = doc["modality_weights"]
        elif "online_prob" in doc:
            op = float(doc["online_prob"])
            cfg.online_prob = op
            cfg.modality_weights = {
                "face_to_face": 1.0 - op,
                "voice_message": op,
                "text_message": 0.0,
            }

        return cfg


def _roll_modality(rng: np.random.Generator, weights: Dict[str, float]) -> str:
    """Sample a session modality from the weight distribution."""
    modalities = list(weights.keys())
    probs = np.array([weights[m] for m in modalities], dtype=float)
    probs /= probs.sum()
    return rng.choice(modalities, p=probs)


def _apply_gt_cutoff(
    eval_instances: Dict,
    max_per_dim: int,
    cutoff_ratio: float,
) -> Dict:
    """Trim dimensions that hit their cap to cutoff_ratio * max_per_dim.

    If a dimension produced exactly max_per_dim instances (i.e. it hit the
    cap), keep only the first ``int(max_per_dim * cutoff_ratio)`` instances.
    Dimensions below the cap are left untouched.
    """
    cutoff = int(max_per_dim * cutoff_ratio)
    for dim, instances in eval_instances.items():
        if dim == Dimension.D4_PERMISSION:
            continue  # D6 is sampled 40/40/20 by its builder; trimming would unbalance it
        if len(instances) >= max_per_dim:
            before = len(instances)
            eval_instances[dim] = instances[:cutoff]
            log.info(
                "GT cutoff: %s hit cap (%d) — trimmed to %d (%.0f%%)",
                dim.value, before, cutoff, cutoff_ratio * 100,
            )
    return eval_instances


class Orchestrator:
    """Coordinate the full MASim pipeline."""

    def __init__(self, cfg: PipelineConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, stop_after: int | None = None) -> Dict[str, Any]:
        """Execute the complete pipeline.

        Args:
            stop_after: If set, stop after this stage number (2 = checkpoint, 4 = events).

        Returns a report dict with statistics and paths.
        """
        setup_logging(self.cfg.log_level)
        start_time = time.monotonic()
        timer = _StageTimer()
        report: Dict[str, Any] = {"config": str(self.cfg)}

        # --- Check for existing checkpoint (skip Stages 1-2 if present) ---
        _checkpoint_files = [
            self.output_dir / "agents_personas.jsonl",
            self.output_dir / "locations.json",
            self.output_dir / "groups.json",
        ]
        _has_checkpoint = all(f.exists() for f in _checkpoint_files) and not self.cfg.dry_run

        if _has_checkpoint:
            timer.start("stage_1_graph")
            timer.start("stage_2_personas")
            log.info("Stages 1-2: Loading from checkpoint in %s ...", self.output_dir)
            from MASim.utils.io import read_jsonl
            import networkx as nx

            # Load agents
            agents_raw = read_jsonl(self.output_dir / "agents_personas.jsonl")
            agents: Dict[str, EntityAgent] = {}
            for row in agents_raw:
                aid = row["agent_id"]
                persona = PersonaCard.from_dict(row["persona"])
                neighbors = row.get("social_neighbors", {})
                agents[aid] = EntityAgent(
                    agent_id=aid,
                    persona=persona,
                    social_neighbors=neighbors,
                    seed=self.cfg.seed + sorted(r["agent_id"] for r in agents_raw).index(aid),
                    memory_context_sessions=self.cfg.memory_context_sessions,
                )
            agent_ids = sorted(agents.keys())

            # Reconstruct graph from agent neighbors
            graph = nx.Graph()
            for aid in agent_ids:
                graph.add_node(aid)
            for aid, agent in agents.items():
                for nid, weight in agent.social_neighbors.items():
                    if nid in agents:
                        graph.add_edge(aid, nid, weight=weight)
            props = graph_properties(graph)
            report["graph"] = props

            # Inject social knowledge
            persona_map = {aid: a.persona for aid, a in agents.items()}
            inject_social_knowledge(agents, graph, persona_map)

            # Load locations
            loc_data = json.loads((self.output_dir / "locations.json").read_text())
            locations = [Location(**{k: v for k, v in d.items() if k in Location.__dataclass_fields__}) for d in loc_data]

            # Load transit matrix
            transit_path = self.output_dir / "transit_matrix.json"
            transit_matrix = json.loads(transit_path.read_text()) if transit_path.exists() else {}

            # Load groups
            grp_data = json.loads((self.output_dir / "groups.json").read_text())
            groups = [Group(**{k: v for k, v in d.items() if k in Group.__dataclass_fields__}) for d in grp_data]

            # Load person roles
            roles_path = self.output_dir / "person_roles.jsonl"
            person_roles = [PersonRole(**{k: v for k, v in d.items() if k in PersonRole.__dataclass_fields__}) for d in read_jsonl(roles_path)] if roles_path.exists() else []

            report["n_agents"] = len(agents)
            report["n_locations"] = len(locations)
            report["n_groups"] = len(groups)
            report["n_roles"] = len(person_roles)
            log.info(
                "Checkpoint loaded: %d agents, %d locations, %d groups",
                len(agents), len(locations), len(groups),
            )

            # LLM client still needed for later stages
            llm_client = LLMClient(self.cfg.llm)
            llm_client.set_detail_log(self.output_dir / "llm_calls_detail.jsonl")
        else:
            # --- Stage 1: Build social graph ---
            timer.start("stage_1_graph")
            log.info("Stage 1: Building social graph...")
            graph = build_graph(self.cfg.graph)
            props = graph_properties(graph)
            report["graph"] = props
            log.info("Graph: %d nodes, %d edges", props["n_nodes"], props["n_edges"])

            # --- Stage 2: Generate personas + create agents ---
            timer.start("stage_2_personas")
            log.info("Stage 2: Generating personas...")
            llm_client = LLMClient(self.cfg.llm)
            llm_client.set_detail_log(self.output_dir / "llm_calls_detail.jsonl")
            persona_factory = PersonaFactory(llm_client, seed=self.cfg.seed)
            personas = persona_factory.generate_personas(
                n_agents=self.cfg.graph.n_agents, dry_run=self.cfg.dry_run,
            )

            # Relabel graph nodes from agent_XXXX to persona name slugs
            slugs = personas_to_slugs(personas)
            # When two LLM personas share a name the slug gets a numeric suffix (_1, _2…).
            # Propagate that disambiguation to the display name so schedules / outputs
            # don't show four agents all named "Elias Thorne".
            for slug, persona in zip(slugs, personas):
                m = re.match(r'^(.+)_(\d+)$', slug)
                if m:
                    persona.name = f"{persona.name} ({m.group(2)})"
            graph = relabel_graph(graph, slugs)

            agents: Dict[str, EntityAgent] = {}
            agent_ids = list(graph.nodes)
            for i, (agent_id, persona) in enumerate(zip(agent_ids, personas)):
                neighbors = {
                    n: graph[agent_id][n].get("weight", 1.0)
                    for n in graph.neighbors(agent_id)
                }
                agents[agent_id] = EntityAgent(
                    agent_id=agent_id,
                    persona=persona,
                    social_neighbors=neighbors,
                    seed=self.cfg.seed + i,
                    memory_context_sessions=self.cfg.memory_context_sessions,
                )
            report["n_agents"] = len(agents)

            # --- Stage 2d: Inject social knowledge (public profiles) ---
            log.info("Stage 2d: Injecting social knowledge into agents...")
            persona_map = {aid: a.persona for aid, a in agents.items()}
            inject_social_knowledge(agents, graph, persona_map)

            # --- Stage 2b: Generate location graph ---
            log.info("Stage 2b: Generating locations...")
            location_factory = LocationFactory(llm_client, seed=self.cfg.seed)
            locations, transit_matrix = location_factory.generate_locations(
                personas=[a.persona for a in agents.values()],
                agent_ids=list(agents.keys()),
                dry_run=self.cfg.dry_run,
            )

            # --- Stage 2c: Infer groups and role assignments ---
            log.info("Stage 2c: Inferring groups...")
            group_factory = GroupFactory()
            groups, person_roles = group_factory.generate_groups(
                personas=[a.persona for a in agents.values()],
                agent_ids=list(agents.keys()),
                locations=locations,
            )
            report["n_locations"] = len(locations)
            report["n_groups"] = len(groups)
            report["n_roles"] = len(person_roles)

            # -- Checkpoint: flush static data --
            log.info("Checkpoint: writing agents/locations/groups to disk...")
            self._write_static_checkpoint(agents, locations, groups, person_roles, transit_matrix)

        if stop_after is not None and stop_after <= 2:
            report["stage_timings"] = timer.to_dict()
            report["stopped_after"] = 2
            elapsed = time.monotonic() - start_time
            report["elapsed_seconds"] = round(elapsed, 1)
            log.info("Stopped after Stage 2 (--stop-after %d). Elapsed: %.1fs", stop_after, elapsed)
            return report

        # --- Check for existing Stages 3-4 checkpoint ---
        _sched_path = self.output_dir / "agent_schedules.jsonl"
        _events_path = self.output_dir / "events.jsonl"
        _has_stage34_checkpoint = (
            _sched_path.exists() and _events_path.exists() and not self.cfg.dry_run
        )

        if _has_stage34_checkpoint:
            timer.start("stage_3_skeleton_schedules")
            timer.start("stage_4_events")
            log.info("Stages 3-4: Loading from checkpoint in %s ...", self.output_dir)

            # Reconstruct skeleton_logs from agent_schedules.jsonl
            # Format: {agent_id, day_index, slots: [{start_hour, end_hour, activity_type, location_id, ...}]}
            sched_raw = read_jsonl(_sched_path)
            sim_start = self.cfg.time_range[0]
            skeleton_logs: Dict[str, List] = {}
            for row in sched_raw:
                aid = row["agent_id"]
                day_idx = row.get("day_index", 0)
                day_offset = sim_start + float(day_idx)
                if aid not in skeleton_logs:
                    skeleton_logs[aid] = []
                for slot in row.get("slots", []):
                    # Convert day-relative hours to simulation time
                    start_frac = slot["start_hour"] / 24.0
                    end_frac = slot["end_hour"] / 24.0
                    skeleton_logs[aid].append(ActivityEntry(
                        agent_id=aid,
                        activity_type=slot.get("activity_type", "solo"),
                        start_time=day_offset + start_frac,
                        end_time=day_offset + end_frac,
                        location_id=slot.get("location_id", ""),
                        group_id=slot.get("group_id", ""),
                        role_name=slot.get("role_name", ""),
                        participants=slot.get("participants", []),
                    ))

            # Re-derive encounter_windows deterministically (no LLM calls)
            from MASim.generation.schedule_factory import _detect_encounters
            encounter_windows = _detect_encounters(skeleton_logs, self.cfg.time_range)
            report["n_encounter_windows"] = len(encounter_windows)

            # Reconstruct events from events.jsonl
            events_raw = read_jsonl(_events_path)
            events = [WorldEvent.from_dict(e) for e in events_raw]
            report["n_events"] = len(events)

            log.info(
                "Stages 3-4 checkpoint loaded: %d encounter windows, %d events",
                len(encounter_windows), len(events),
            )
        else:
            # --- Stage 3: Build skeleton schedules + detect encounter windows ---
            timer.start("stage_3_skeleton_schedules")
            log.info("Stage 3: Building skeleton schedules...")
            memory_writer = MemoryWriter(llm_client)
            schedule_factory = ScheduleFactory(memory_writer, transit_matrix)
            skeleton_logs, encounter_windows = schedule_factory.build_skeleton(
                agents=agents,
                locations=locations,
                time_range=self.cfg.time_range,
                groups=groups,
                llm_client=llm_client,
                dry_run=self.cfg.dry_run,
            )
            report["n_encounter_windows"] = len(encounter_windows)

            if stop_after is not None and stop_after <= 3:
                # Write skeleton schedules for inspection
                write_jsonl(
                    self.output_dir / "agent_schedules.jsonl",
                    [{"agent_id": aid, "entries": [e.to_dict() for e in entries]}
                     for aid, entries in skeleton_logs.items()],
                )
                report["stage_timings"] = timer.to_dict()
                report["stopped_after"] = 3
                elapsed = time.monotonic() - start_time
                report["elapsed_seconds"] = round(elapsed, 1)
                log.info("Stopped after Stage 3 (--stop-after %d). Elapsed: %.1fs", stop_after, elapsed)
                return report

            # --- Stage 4: Generate event schedule (grounded in encounter windows) ---
            timer.start("stage_4_events")
            log.info("Stage 4: Generating events...")
            event_factory = EventFactory(llm_client, seed=self.cfg.seed)
            agent_names = {aid: a.persona.name for aid, a in agents.items()}
            events = event_factory.generate_events(
                n_events=self.cfg.n_events,
                social_graph=graph,
                time_range=self.cfg.time_range,
                dry_run=self.cfg.dry_run,
                agent_names=agent_names,
                encounter_windows=encounter_windows,
            )
            # Attach a plausible location_id to each event
            _assign_event_locations(events, locations)
            report["n_events"] = len(events)

            # -- Checkpoint: flush events so a crash during Stage 5 leaves recoverable data --
            log.info("Checkpoint: writing events to disk...")
            write_jsonl(self.output_dir / "events.jsonl", [e.to_dict() for e in events])

            if stop_after is not None and stop_after <= 4:
                report["stage_timings"] = timer.to_dict()
                report["stopped_after"] = 4
                elapsed = time.monotonic() - start_time
                report["elapsed_seconds"] = round(elapsed, 1)
                log.info("Stopped after Stage 4 (--stop-after %d). Elapsed: %.1fs", stop_after, elapsed)
                return report

        # --- Stage 5: Run simulation ---
        timer.start("stage_5_simulation")
        dialogue_engine = DialogueEngine(
            llm_client,
            max_turns_per_session=self.cfg.max_turns_per_session,
            min_turns_per_session=self.cfg.min_turns_per_session,
            no_token_limit=self.cfg.no_token_limit,
        )
        scheduler = BatchScheduler(max_concurrent=self.cfg.llm.concurrency)

        if self.cfg.faster_batch:
            log.info("Stage 5: Running faster-batch simulation...")
            from MASim.core.scheduler import DailyScheduler
            daily_scheduler = DailyScheduler(
                groups=groups,
                agents=agents,
                locations=locations,
                rng=np.random.default_rng(self.cfg.seed),
                modality_weights=self.cfg.modality_weights,
                base_encounter_prob=self.cfg.base_encounter_prob,
                graph=graph,
            )
            sessions = self._run_faster_batch_simulation(
                daily_scheduler, dialogue_engine, scheduler, agents,
                encounter_windows, events,
                skeleton_logs=skeleton_logs,
                llm_client=llm_client,
            )
        elif self.cfg.scheduler_driven:
            log.info("Stage 5: Running scheduler-driven simulation...")
            from MASim.core.scheduler import DailyScheduler
            daily_scheduler = DailyScheduler(
                groups=groups,
                agents=agents,
                locations=locations,
                rng=np.random.default_rng(self.cfg.seed),
                modality_weights=self.cfg.modality_weights,
                base_encounter_prob=self.cfg.base_encounter_prob,
                graph=graph,
            )
            sessions = self._run_scheduled_simulation(
                daily_scheduler, dialogue_engine, scheduler, agents,
                encounter_windows, events,
                skeleton_logs=skeleton_logs,
                llm_client=llm_client,
            )
        else:
            log.info("Stage 5: Running legacy event-driven simulation...")
            broadcaster = WorldBroadcaster(events, agents, graph)
            sessions = self._run_simulation_legacy(
                broadcaster, dialogue_engine, scheduler, agents, events,
            )
        # Propagate location_id from triggering event to session
        # Remote sessions (voice/text) have no physical location
        event_location_map = {e.event_id: e.location_id for e in events}
        for sess in sessions:
            if sess.modality in ("voice_message", "text_message"):
                sess.location_id = ""
            elif not sess.location_id and sess.triggering_events:
                sess.location_id = event_location_map.get(sess.triggering_events[0], "")
        report["n_sessions"] = len(sessions)
        report["n_turns"] = sum(len(s.turns) for s in sessions)
        report["corpus_tokens"] = _count_corpus_tokens(sessions)
        if self.cfg.scheduler_driven:
            report["n_group_meeting_sessions"] = sum(
                1 for s in sessions if s.metadata.get("trigger") == "group_meeting"
            )
            report["n_event_sessions"] = sum(
                1 for s in sessions if s.metadata.get("trigger") == "event"
            )
            report["n_encounter_sessions"] = sum(
                1 for s in sessions if s.metadata.get("trigger") == "encounter"
            )
            report["n_remote_sessions"] = sum(
                1 for s in sessions if s.metadata.get("trigger") == "remote"
            )

        # --- Stage 5a: Process inline images from text_message turns ---
        timer.start("stage_5a_inline_images")
        log.info("Stage 5a: Processing inline images...")
        inline_image_entries = self._process_inline_images(sessions)
        report["n_inline_images"] = len(inline_image_entries)

        # --- Stage 5b: Augment skeleton logs with sessions ---
        timer.start("stage_5b_activity_logs")
        log.info("Stage 5b: Augmenting skeleton logs with sessions...")
        activity_logs, daily_schedules = schedule_factory.augment_with_sessions(
            skeleton_logs=skeleton_logs,
            sessions=sessions,
            agents=agents,
            locations=locations,
            time_range=self.cfg.time_range,
            dry_run=self.cfg.dry_run,
        )
        report["n_activity_entries"] = sum(len(v) for v in activity_logs.values())

        # --- Stage 5c: Generate voice TTS for voice_message turns ---
        timer.start("stage_5c_voice_tts")
        log.info("Stage 5c: Processing voice messages...")
        voice_audio_entries = self._process_voice_messages(sessions)
        report["n_voice_audio"] = len(voice_audio_entries)

        # --- Stage 5d: Person-agent sessions ---
        # For scheduler-driven path, PA sessions are interleaved in the day loop
        # and already included in `sessions`. For legacy path, run separately.
        if self.cfg.person_agent.enabled:
            if self.cfg.scheduler_driven:
                # PA already generated inside _run_scheduled_simulation
                report["n_person_agent_sessions"] = sum(
                    1 for s in sessions
                    if s.metadata.get("session_type", "").startswith("person_agent")
                )
                report["n_person_agent_activity"] = sum(
                    1 for s in sessions
                    if s.metadata.get("session_type") == "person_agent_activity"
                )
                report["n_person_agent_reflection"] = sum(
                    1 for s in sessions
                    if s.metadata.get("session_type") == "person_agent_reflection"
                )
                report["n_person_agent_probe"] = sum(
                    1 for s in sessions
                    if s.metadata.get("session_type") == "person_agent_probe"
                )
            else:
                timer.start("stage_5d_person_agent")
                log.info("Stage 5d: Generating person-agent sessions (legacy)...")
                person_agent_sessions = self._run_person_agent_sessions(
                    agents, sessions, llm_client, scheduler, rng=np.random.default_rng(self.cfg.seed + 777),
                )
                sessions.extend(person_agent_sessions)
                with (self.output_dir / "corpus_sessions.jsonl").open("a", encoding="utf-8") as fh:
                    for sess in person_agent_sessions:
                        fh.write(json.dumps(sess.to_dict(), ensure_ascii=False) + "\n")
                report["n_person_agent_sessions"] = len(person_agent_sessions)
                report["n_person_agent_activity"] = sum(
                    1 for s in person_agent_sessions
                    if s.metadata.get("session_type") == "person_agent_activity"
                )
                report["n_person_agent_reflection"] = sum(
                    1 for s in person_agent_sessions
                    if s.metadata.get("session_type") == "person_agent_reflection"
                )
                report["n_person_agent_probe"] = sum(
                    1 for s in person_agent_sessions
                    if s.metadata.get("session_type") == "person_agent_probe"
                )
            log.info(
                "Person-agent: %d sessions (%d activity, %d reflection, %d probe)",
                report.get("n_person_agent_sessions", 0),
                report.get("n_person_agent_activity", 0),
                report.get("n_person_agent_reflection", 0),
                report.get("n_person_agent_probe", 0),
            )

        # Build corpus
        corpus = DialogueCorpus(
            sessions=sessions,
            agents={aid: a.to_state() for aid, a in agents.items()},
            events=events,
            social_graph_edges=[
                {"source": u, "target": v, **d}
                for u, v, d in graph.edges(data=True)
            ],
            locations=locations,
            groups=groups,
            person_roles=person_roles,
            activity_logs=activity_logs,
        )

        # --- Stage 5: Inject conflicts ---
        timer.start("stage_5_conflicts")
        log.info("Stage 5: Injecting conflicts...")
        conflict_cfg = ConflictInjectorConfig(
            inject_prob=self.cfg.conflict_inject_prob,
            max_injections=self.cfg.max_conflicts,
            seed=self.cfg.seed,
        )
        conflict_injector = ConflictInjector(llm_client, conflict_cfg)
        corpus, conflict_gts = conflict_injector.inject(corpus, dry_run=self.cfg.dry_run)
        report["n_injected_conflicts"] = len(conflict_gts)

        # Detect organic conflicts from probe restatements
        from MASim.ground_truth.d1_conflict import detect_organic_conflicts
        organic_gts = detect_organic_conflicts(corpus, agents)
        conflict_gts = conflict_gts + organic_gts
        report["n_organic_conflicts"] = len(organic_gts)
        report["n_conflicts"] = len(conflict_gts)

        # --- Stage 6: Inject permissions ---
        timer.start("stage_6_permissions")
        log.info("Stage 6: Injecting permission scenarios...")
        perm_cfg = PermissionInjectorConfig(
            inject_prob=self.cfg.permission_inject_prob,
            max_injections=self.cfg.max_permissions,
            seed=self.cfg.seed,
        )
        perm_injector = PermissionInjector(llm_client, perm_cfg)
        corpus, permission_gts = perm_injector.inject(corpus, dry_run=self.cfg.dry_run)
        report["n_permissions"] = len(permission_gts)

        # --- Stage 6b: Inject autonomous privacy scenarios ---
        timer.start("stage_6b_autonomous_privacy")
        log.info("Stage 6b: Injecting autonomous privacy scenarios...")
        auto_priv_cfg = AutonomousPrivacyConfig(
            inject_prob=self.cfg.autonomous_privacy_inject_prob,
            max_injections=self.cfg.max_autonomous_privacy,
            seed=self.cfg.seed,
        )
        auto_priv_injector = AutonomousPrivacyInjector(llm_client, auto_priv_cfg)
        corpus, auto_priv_gts = auto_priv_injector.inject(corpus, dry_run=self.cfg.dry_run)
        report["n_autonomous_privacy"] = len(auto_priv_gts)

        timer.start("stage_7_multimodal")
        # --- Stage 7: Multimodal pipeline (optional) ---
        if self.cfg.multimodal.enable_image or self.cfg.multimodal.enable_audio:
            log.info("Stage 7: Running multimodal pipeline...")
            self._run_multimodal_pipeline(corpus, llm_client)
        else:
            log.info("Stage 7: Multimodal pipeline skipped (disabled in config).")

        # --- Stage 8: Track coreferences ---
        timer.start("stage_8_coreferences")
        log.info("Stage 8: Tracking cross-session references...")
        coref_tracker = CoreferenceTracker()
        cross_refs, anaphora_gts = coref_tracker.track(corpus)
        corpus = coref_tracker.annotate_corpus(corpus, cross_refs)
        report["n_cross_refs"] = len(cross_refs)

        # --- Stage 9: Derive ground truth for evaluated dimensions ---
        timer.start("stage_9_ground_truth")
        log.info("Stage 9: Extracting ground truth...")
        gt_extractor = GTExtractor(
            llm_client=llm_client if not self.cfg.dry_run else None,
            hardening_cfg=self.cfg.hardening,
            graph_layer_weights=[layer.weight for layer in self.cfg.graph.layers],
        )
        eval_instances = gt_extractor.extract_all(
            corpus,
            conflict_gts=conflict_gts,
            anaphora_gts=anaphora_gts,
            permission_gts=permission_gts,
            autonomous_privacy_gts=auto_priv_gts,
            max_per_dimension=self.cfg.max_instances_per_dim,
        )
        eval_instances = _apply_gt_cutoff(
            eval_instances, self.cfg.max_instances_per_dim, self.cfg.gt_cutoff_ratio,
        )
        report["eval_instances"] = {d.value: len(v) for d, v in eval_instances.items()}

        # --- Stage 10: Compute ego projections + populate eval instance session refs ---
        timer.start("stage_10_ego_projections")
        log.info("Stage 10: Computing ego-centric projections...")

        # Context tier classification per dimension
        _FULL_EGO_DIMS = {Dimension.D3_CONFABULATION}
        _CROSS_SESSION_DIMS = {Dimension.D1_CONFLICT, Dimension.D2_ANAPHORA, Dimension.D8_TEMPORAL}
        # Everything else is single_session (D5, D6, D7, D10)

        # Build ego session map: agent_id -> [session_id, ...]
        ego_session_map: Dict[str, List[str]] = {}
        ego_data = {}
        for agent_id in agent_ids:
            ego_corpus = ego_project(agent_id, corpus)
            ego_sids = [s.session_id for s in ego_corpus.sessions]
            ego_session_map[agent_id] = ego_sids
            ego_data[agent_id] = {
                "n_sessions": len(ego_corpus.sessions),
                "n_events": len(ego_corpus.events),
                "n_turns": sum(len(s.turns) for s in ego_corpus.sessions),
            }
        report["ego_projections"] = {
            "n_agents": len(ego_data),
            "avg_sessions": sum(v["n_sessions"] for v in ego_data.values()) / max(len(ego_data), 1),
            "avg_turns": sum(v["n_turns"] for v in ego_data.values()) / max(len(ego_data), 1),
        }

        # Populate session ID references into eval instances (no ego_context embedding)
        session_participants = {
            s.session_id: s.participants for s in corpus.sessions
        }
        session_ends = session_end_times(corpus.sessions)
        for dim, instances in eval_instances.items():
            if dim == Dimension.D4_PERMISSION:
                # D6 items carry their own ego (the fact owner), evidence
                # ([source session]), context tier and query_timestamp (after
                # the last session); see ground_truth/d4_permission.py.
                continue
            # Determine context tier for this dimension
            if dim in _FULL_EGO_DIMS:
                tier = "full_ego"
            elif dim in _CROSS_SESSION_DIMS:
                tier = "cross_session"
            else:
                tier = "single_session"

            for inst in instances:
                # Resolve ego agent
                ego_agent = inst.metadata.get("query_agent")
                evidence = inst.metadata.get("evidence_sessions", [])
                if not ego_agent:
                    if inst.answerer_agent_id and not is_assistant(inst.answerer_agent_id):
                        ego_agent = inst.answerer_agent_id
                    elif evidence:
                        participants = session_participants.get(evidence[0], [])
                        ego_agent = participants[0] if participants else None

                inst.ego_agent_id = ego_agent or ""
                inst.context_tier = tier

                # Populate evidence_session_ids
                if tier == "full_ego" and ego_agent and ego_agent in ego_session_map:
                    inst.evidence_session_ids = list(ego_session_map[ego_agent])
                else:
                    inst.evidence_session_ids = list(evidence)

                # Ask the question after the last evidence session (rule in
                # MASim/ground_truth/query_time.py)
                qts, qts_rule = query_timestamp(evidence, ego_session_map.get(inst.ego_agent_id), session_ends)
                inst.metadata["query_timestamp"] = qts
                inst.metadata["query_timestamp_rule"] = qts_rule
                # All eval queries are presented as online (voice/text) interactions
                inst.metadata["query_modality"] = "online"

        # --- Stage 11: Write outputs ---
        timer.start("stage_11_write_outputs")
        log.info("Stage 11: Writing outputs...")
        self._write_outputs(corpus, eval_instances, ego_data, report,
                            locations=locations, groups=groups,
                            person_roles=person_roles, transit_matrix=transit_matrix,
                            activity_logs=activity_logs, daily_schedules=daily_schedules,
                            inline_image_entries=inline_image_entries,
                            voice_audio_entries=voice_audio_entries,
                            ego_session_map=ego_session_map)

        timer.stop()
        duration = time.monotonic() - start_time
        report["duration_seconds"] = round(duration, 2)
        report["llm_calls"] = llm_client.call_count
        report["token_usage"] = llm_client.get_usage_stats()
        report["stage_timings"] = timer.to_dict()
        report["generated_at"] = datetime.now(timezone.utc).isoformat()

        llm_client.write_timing_log(self.output_dir / "llm_timing.jsonl")

        write_json_report(self.output_dir / "pipeline_report.json", report)
        log.info("Pipeline complete in %.1fs. Output: %s", duration, self.output_dir)
        return report

    # ------------------------------------------------------------------
    # Resume from a specific stage
    # ------------------------------------------------------------------

    def resume_from(self, stage: str) -> Dict[str, Any]:
        """Resume a crashed/interrupted pipeline from a given stage.

        Loads all checkpoint data from ``self.output_dir`` and continues
        from the requested stage.  Currently supported stages:

        - ``"5d"`` — person-agent sessions + all post-5d processing
        - ``"post_sim"`` — everything after simulation (conflicts → output)
        - ``"regen_eval"`` — regenerate eval instances only (no corpus changes)

        Returns the same report dict as ``run()``.
        """
        from MASim.utils.io import read_jsonl
        import networkx as nx

        setup_logging(self.cfg.log_level)
        start_time = time.monotonic()
        timer = _StageTimer()
        report: Dict[str, Any] = {"config": str(self.cfg), "resumed_from": stage}

        # ---- Load checkpoint data from disk ----
        timer.start("load_checkpoint")
        log.info("Loading checkpoint data from %s ...", self.output_dir)

        # Agents
        agents_raw = read_jsonl(self.output_dir / "agents_personas.jsonl")
        agents: Dict[str, EntityAgent] = {}
        for row in agents_raw:
            aid = row["agent_id"]
            persona = PersonaCard.from_dict(row["persona"])
            neighbors = row.get("social_neighbors", {})
            agents[aid] = EntityAgent(
                agent_id=aid,
                persona=persona,
                social_neighbors=neighbors,
                seed=self.cfg.seed + list(sorted(a["agent_id"] for a in agents_raw)).index(aid),
                memory_context_sessions=self.cfg.memory_context_sessions,
            )
        agent_ids = sorted(agents.keys())
        log.info("Loaded %d agents", len(agents))

        # Sessions (person-person only — PA sessions may not exist yet)
        sessions_raw = read_jsonl(self.output_dir / "corpus_sessions.jsonl")
        sessions = [Session.from_dict(d) for d in sessions_raw]
        # Split into PP (non-person-agent) and existing PA
        pp_sessions = [s for s in sessions if not s.metadata.get("session_type", "").startswith("person_agent")]
        existing_pa = [s for s in sessions if s.metadata.get("session_type", "").startswith("person_agent")]
        log.info("Loaded %d sessions (%d PP, %d existing PA)", len(sessions), len(pp_sessions), len(existing_pa))

        # Events
        events_raw = read_jsonl(self.output_dir / "events.jsonl")
        events = [WorldEvent.from_dict(d) for d in events_raw]
        log.info("Loaded %d events", len(events))

        # Reconstruct social graph from agents' social_neighbors
        graph = nx.Graph()
        for aid in agent_ids:
            graph.add_node(aid)
        for aid, agent in agents.items():
            for nid, weight in agent.social_neighbors.items():
                if nid in agents:
                    graph.add_edge(aid, nid, weight=weight)

        # LLM client + scheduler
        llm_client = LLMClient(self.cfg.llm)
        scheduler = BatchScheduler(max_concurrent=self.cfg.llm.concurrency)
        persona_map = {aid: a.persona for aid, a in agents.items()}

        report["n_agents"] = len(agents)
        report["n_events"] = len(events)

        # ---- Stage 5d (if requested) ----
        if stage in ("5d",):
            timer.start("stage_5d_person_agent")
            log.info("Stage 5d: Generating person-agent sessions...")
            person_agent_sessions = self._run_person_agent_sessions(
                agents, pp_sessions, llm_client, scheduler,
                rng=np.random.default_rng(self.cfg.seed + 777),
            )
            # Remove any existing PA sessions and replace
            sessions = pp_sessions + person_agent_sessions
            # Overwrite corpus_sessions.jsonl with PP + new PA
            from MASim.utils.io import write_jsonl as _write_jsonl
            _write_jsonl(
                self.output_dir / "corpus_sessions.jsonl",
                [s.to_dict() for s in sessions],
            )
            report["n_person_agent_sessions"] = len(person_agent_sessions)
            report["n_person_agent_activity"] = sum(
                1 for s in person_agent_sessions
                if s.metadata.get("session_type") == "person_agent_activity"
            )
            report["n_person_agent_reflection"] = sum(
                1 for s in person_agent_sessions
                if s.metadata.get("session_type") == "person_agent_reflection"
            )
            report["n_person_agent_probe"] = sum(
                1 for s in person_agent_sessions
                if s.metadata.get("session_type") == "person_agent_probe"
            )
            log.info(
                "Person-agent: %d sessions (%d activity, %d reflection, %d probe)",
                report["n_person_agent_sessions"],
                report["n_person_agent_activity"],
                report["n_person_agent_reflection"],
                report.get("n_person_agent_probe", 0),
            )

        report["n_sessions"] = len(sessions)
        report["n_turns"] = sum(len(s.turns) for s in sessions)
        report["corpus_tokens"] = _count_corpus_tokens(sessions)

        # ---- Load lifecycle data if available ----
        from MASim.core.schema import ActivityEntry, Location, Group, PersonRole
        locations: List[Location] = []
        groups: List[Group] = []
        person_roles: List[PersonRole] = []
        activity_logs: Dict[str, List[ActivityEntry]] = {}

        loc_path = self.output_dir / "locations.json"
        if loc_path.exists():
            import json as _json
            with loc_path.open() as _fh:
                for _d in _json.load(_fh):
                    locations.append(Location(**{k: v for k, v in _d.items() if k in Location.__dataclass_fields__}))
            log.info("Loaded %d locations from checkpoint", len(locations))

        grp_path = self.output_dir / "groups.json"
        if grp_path.exists():
            with grp_path.open() as _fh:
                for _d in _json.load(_fh):
                    groups.append(Group(**{k: v for k, v in _d.items() if k in Group.__dataclass_fields__}))
            log.info("Loaded %d groups from checkpoint", len(groups))

        roles_path = self.output_dir / "person_roles.jsonl"
        if roles_path.exists():
            roles_raw = read_jsonl(roles_path)
            for _d in roles_raw:
                person_roles.append(PersonRole(**{k: v for k, v in _d.items() if k in PersonRole.__dataclass_fields__}))
            log.info("Loaded %d person_roles from checkpoint", len(person_roles))

        logs_path = self.output_dir / "activity_logs.jsonl"
        if logs_path.exists():
            logs_raw = read_jsonl(logs_path)
            for _d in logs_raw:
                entry = ActivityEntry(**{k: v for k, v in _d.items() if k in ActivityEntry.__dataclass_fields__})
                activity_logs.setdefault(entry.agent_id, []).append(entry)
            log.info("Loaded %d activity_log entries from checkpoint", sum(len(v) for v in activity_logs.values()))

        # ---- Build corpus from loaded data ----
        corpus = DialogueCorpus(
            sessions=sessions,
            agents={aid: a.to_state() for aid, a in agents.items()},
            events=events,
            social_graph_edges=[
                {"source": u, "target": v, **d}
                for u, v, d in graph.edges(data=True)
            ],
            locations=locations,
            groups=groups,
            person_roles=person_roles,
            activity_logs=activity_logs,
        )

        # ---- regen_eval: skip injection, reconstruct GTs, re-run stages 9-11 ----
        if stage == "regen_eval":
            return self._regen_eval_only(
                corpus, agents, agent_ids, persona_map,
                llm_client, timer, report, start_time,
            )

        # ---- Post-simulation stages (conflicts → output) ----

        # Conflicts
        timer.start("stage_5_conflicts")
        log.info("Injecting conflicts...")
        conflict_cfg = ConflictInjectorConfig(
            inject_prob=self.cfg.conflict_inject_prob,
            max_injections=self.cfg.max_conflicts,
            seed=self.cfg.seed,
        )
        conflict_injector = ConflictInjector(llm_client, conflict_cfg)
        corpus, conflict_gts = conflict_injector.inject(corpus, dry_run=self.cfg.dry_run)
        report["n_injected_conflicts"] = len(conflict_gts)

        # Detect organic conflicts from probe restatements
        from MASim.ground_truth.d1_conflict import detect_organic_conflicts
        organic_gts = detect_organic_conflicts(corpus, agents)
        conflict_gts = conflict_gts + organic_gts
        report["n_organic_conflicts"] = len(organic_gts)
        report["n_conflicts"] = len(conflict_gts)

        # Permissions
        timer.start("stage_6_permissions")
        log.info("Injecting permissions...")
        perm_cfg = PermissionInjectorConfig(
            inject_prob=self.cfg.permission_inject_prob,
            max_injections=self.cfg.max_permissions,
            seed=self.cfg.seed,
        )
        perm_injector = PermissionInjector(llm_client, perm_cfg)
        corpus, permission_gts = perm_injector.inject(corpus, dry_run=self.cfg.dry_run)
        report["n_permissions"] = len(permission_gts)

        # Autonomous privacy
        timer.start("stage_6b_autonomous_privacy")
        log.info("Injecting autonomous privacy...")
        auto_priv_cfg = AutonomousPrivacyConfig(
            inject_prob=self.cfg.autonomous_privacy_inject_prob,
            max_injections=self.cfg.max_autonomous_privacy,
            seed=self.cfg.seed,
        )
        auto_priv_injector = AutonomousPrivacyInjector(llm_client, auto_priv_cfg)
        corpus, auto_priv_gts = auto_priv_injector.inject(corpus, dry_run=self.cfg.dry_run)
        report["n_autonomous_privacy"] = len(auto_priv_gts)

        # Coreferences
        timer.start("stage_8_coreferences")
        log.info("Tracking cross-session references...")
        coref_tracker = CoreferenceTracker()
        cross_refs, anaphora_gts = coref_tracker.track(corpus)
        corpus = coref_tracker.annotate_corpus(corpus, cross_refs)
        report["n_cross_refs"] = len(cross_refs)

        # Ground truth for evaluated dimensions
        timer.start("stage_9_ground_truth")
        log.info("Extracting ground truth...")
        gt_extractor = GTExtractor(
            llm_client=llm_client if not self.cfg.dry_run else None,
            hardening_cfg=self.cfg.hardening,
            graph_layer_weights=[layer.weight for layer in self.cfg.graph.layers],
        )
        eval_instances = gt_extractor.extract_all(
            corpus,
            conflict_gts=conflict_gts,
            anaphora_gts=anaphora_gts,
            permission_gts=permission_gts,
            autonomous_privacy_gts=auto_priv_gts,
            max_per_dimension=self.cfg.max_instances_per_dim,
        )
        eval_instances = _apply_gt_cutoff(
            eval_instances, self.cfg.max_instances_per_dim, self.cfg.gt_cutoff_ratio,
        )
        report["eval_instances"] = {d.value: len(v) for d, v in eval_instances.items()}

        # Ego projections
        timer.start("stage_10_ego_projections")
        log.info("Computing ego-centric projections...")
        _FULL_EGO_DIMS = {Dimension.D3_CONFABULATION}
        _CROSS_SESSION_DIMS = {Dimension.D1_CONFLICT, Dimension.D2_ANAPHORA, Dimension.D8_TEMPORAL}

        ego_session_map: Dict[str, List[str]] = {}
        ego_data = {}
        for agent_id in agent_ids:
            ego_corpus = ego_project(agent_id, corpus)
            ego_sids = [s.session_id for s in ego_corpus.sessions]
            ego_session_map[agent_id] = ego_sids
            ego_data[agent_id] = {
                "n_sessions": len(ego_corpus.sessions),
                "n_events": len(ego_corpus.events),
                "n_turns": sum(len(s.turns) for s in ego_corpus.sessions),
            }
        report["ego_projections"] = {
            "n_agents": len(ego_data),
            "avg_sessions": sum(v["n_sessions"] for v in ego_data.values()) / max(len(ego_data), 1),
            "avg_turns": sum(v["n_turns"] for v in ego_data.values()) / max(len(ego_data), 1),
        }

        session_participants = {s.session_id: s.participants for s in corpus.sessions}
        session_ends = session_end_times(corpus.sessions)
        for dim, instances in eval_instances.items():
            if dim == Dimension.D4_PERMISSION:
                # D6 items carry their own ego (the fact owner), evidence
                # ([source session]), context tier and query_timestamp (after
                # the last session); see ground_truth/d4_permission.py.
                continue
            if dim in _FULL_EGO_DIMS:
                tier = "full_ego"
            elif dim in _CROSS_SESSION_DIMS:
                tier = "cross_session"
            else:
                tier = "single_session"
            for inst in instances:
                ego_agent = inst.metadata.get("query_agent")
                evidence = inst.metadata.get("evidence_sessions", [])
                if not ego_agent:
                    if inst.answerer_agent_id and not is_assistant(inst.answerer_agent_id):
                        ego_agent = inst.answerer_agent_id
                    elif evidence:
                        participants = session_participants.get(evidence[0], [])
                        ego_agent = participants[0] if participants else None
                inst.ego_agent_id = ego_agent or ""
                inst.context_tier = tier
                if tier == "full_ego" and ego_agent and ego_agent in ego_session_map:
                    inst.evidence_session_ids = list(ego_session_map[ego_agent])
                else:
                    inst.evidence_session_ids = list(evidence)
                qts, qts_rule = query_timestamp(evidence, ego_session_map.get(inst.ego_agent_id), session_ends)
                inst.metadata["query_timestamp"] = qts
                inst.metadata["query_timestamp_rule"] = qts_rule
                inst.metadata["query_modality"] = "online"

        # Write outputs
        timer.start("stage_11_write_outputs")
        log.info("Writing outputs...")
        self._write_outputs(
            corpus, eval_instances, ego_data, report,
            ego_session_map=ego_session_map,
        )

        timer.stop()
        duration = time.monotonic() - start_time
        report["duration_seconds"] = round(duration, 2)
        report["llm_calls"] = llm_client.call_count
        report["token_usage"] = llm_client.get_usage_stats()
        report["stage_timings"] = timer.to_dict()
        report["generated_at"] = datetime.now(timezone.utc).isoformat()

        llm_client.write_timing_log(self.output_dir / "llm_timing.jsonl")

        write_json_report(self.output_dir / "pipeline_report.json", report)
        log.info("Resume complete in %.1fs. Output: %s", duration, self.output_dir)
        return report

    # ------------------------------------------------------------------
    # regen_eval: regenerate eval instances without touching the corpus
    # ------------------------------------------------------------------

    def _regen_eval_only(
        self,
        corpus: DialogueCorpus,
        agents: Dict[str, Any],
        agent_ids: List[str],
        persona_map: Dict[str, Any],
        llm_client: Any,
        timer: Any,
        report: Dict[str, Any],
        start_time: float,
    ) -> Dict[str, Any]:
        """Regenerate eval instances from an existing corpus.

        Does NOT re-inject conflicts or permissions into the corpus.
        Instead, reconstructs ground-truth records from already-injected turns
        and metadata, then re-runs stages 9-11.
        """
        from MASim.ground_truth.d1_conflict import detect_organic_conflicts

        log.info("regen_eval: Reconstructing GTs from existing corpus...")

        # ── Reconstruct ConflictGTs from injected conflict turns ──────────
        timer.start("regen_reconstruct_conflict_gts")
        conflict_gts: List[ConflictGT] = []
        turn_by_id: Dict[str, DialogueTurn] = {}
        for session in corpus.sessions:
            for turn in session.turns:
                turn_by_id[turn.turn_id] = turn

        for session in corpus.sessions:
            for turn in session.turns:
                if not turn.metadata.get("injected_conflict"):
                    continue
                orig_turn_id = turn.metadata.get("original_turn_id", "")
                orig_turn = turn_by_id.get(orig_turn_id)
                if not orig_turn:
                    continue
                # Find original session
                orig_session_id = orig_turn.session_id
                conflict_gts.append(ConflictGT(
                    fact=orig_turn.text,
                    contradictory_fact=turn.text,
                    source_agent=orig_turn.speaker_id,
                    conflicting_agent=turn.speaker_id,
                    original_session=orig_session_id,
                    conflicting_session=session.session_id,
                    original_timestamp=orig_turn.timestamp,
                    conflicting_timestamp=turn.timestamp,
                    # detail fields unavailable for old runs — left empty
                    original_detail=turn.metadata.get("original_detail", ""),
                    changed_detail=turn.metadata.get("changed_detail", ""),
                ))
        log.info("Reconstructed %d conflict GTs from corpus", len(conflict_gts))

        # Organic conflicts
        organic_gts = detect_organic_conflicts(corpus, agents)
        conflict_gts = conflict_gts + organic_gts
        report["n_conflicts"] = len(conflict_gts)

        # ── Permission / autonomous-privacy facts ─────────────────────────
        # D4 (paper D6) reads the injected turns and the Dunbar graph from the
        # corpus itself (ground_truth/d4_permission.py), so no PermissionGT /
        # AutonomousPrivacyGT records are reconstructed here.
        injected = [t.metadata for s in corpus.sessions for t in s.turns]
        report["n_permission_facts"] = sum(1 for m in injected if m.get("injected_permission"))
        report["n_autonomous_privacy_facts"] = sum(1 for m in injected if m.get("injected_autonomous_privacy"))
        log.info("Corpus holds %d permission and %d autonomous privacy facts",
                 report["n_permission_facts"], report["n_autonomous_privacy_facts"])

        # ── Coreference tracking (runs fresh — deterministic) ─────────────
        timer.start("regen_coreferences")
        coref_tracker = CoreferenceTracker()
        cross_refs, anaphora_gts = coref_tracker.track(corpus)
        corpus = coref_tracker.annotate_corpus(corpus, cross_refs)

        # ── Stage 9: Ground truth extraction ──────────────────────────────
        timer.start("stage_9_ground_truth")
        log.info("regen_eval: Extracting ground truth for evaluated dimensions...")
        gt_extractor = GTExtractor(
            llm_client=llm_client if not self.cfg.dry_run else None,
            hardening_cfg=self.cfg.hardening,
            graph_layer_weights=[layer.weight for layer in self.cfg.graph.layers],
        )
        eval_instances = gt_extractor.extract_all(
            corpus,
            conflict_gts=conflict_gts,
            anaphora_gts=anaphora_gts,
            max_per_dimension=self.cfg.max_instances_per_dim,
        )
        eval_instances = _apply_gt_cutoff(
            eval_instances, self.cfg.max_instances_per_dim, self.cfg.gt_cutoff_ratio,
        )
        report["eval_instances"] = {d.value: len(v) for d, v in eval_instances.items()}

        # ── Stage 10: Ego projections ─────────────────────────────────────
        timer.start("stage_10_ego_projections")
        log.info("regen_eval: Computing ego projections...")
        _FULL_EGO_DIMS = {Dimension.D3_CONFABULATION}
        _CROSS_SESSION_DIMS = {Dimension.D1_CONFLICT, Dimension.D2_ANAPHORA, Dimension.D8_TEMPORAL}

        ego_session_map: Dict[str, List[str]] = {}
        ego_data = {}
        for agent_id in agent_ids:
            ego_corpus = ego_project(agent_id, corpus)
            ego_sids = [s.session_id for s in ego_corpus.sessions]
            ego_session_map[agent_id] = ego_sids
            ego_data[agent_id] = {
                "n_sessions": len(ego_corpus.sessions),
                "n_events": len(ego_corpus.events),
                "n_turns": sum(len(s.turns) for s in ego_corpus.sessions),
            }

        session_participants = {s.session_id: s.participants for s in corpus.sessions}
        session_ends = session_end_times(corpus.sessions)
        for dim, instances in eval_instances.items():
            if dim == Dimension.D4_PERMISSION:
                # D6 items carry their own ego (the fact owner), evidence
                # ([source session]), context tier and query_timestamp (after
                # the last session); see ground_truth/d4_permission.py.
                continue
            if dim in _FULL_EGO_DIMS:
                tier = "full_ego"
            elif dim in _CROSS_SESSION_DIMS:
                tier = "cross_session"
            else:
                tier = "single_session"
            for inst in instances:
                ego_agent = inst.metadata.get("query_agent")
                evidence = inst.metadata.get("evidence_sessions", [])
                if not ego_agent:
                    if inst.answerer_agent_id and not is_assistant(inst.answerer_agent_id):
                        ego_agent = inst.answerer_agent_id
                    elif evidence:
                        participants = session_participants.get(evidence[0], [])
                        ego_agent = participants[0] if participants else None
                inst.ego_agent_id = ego_agent or ""
                inst.context_tier = tier
                if tier == "full_ego" and ego_agent and ego_agent in ego_session_map:
                    inst.evidence_session_ids = list(ego_session_map[ego_agent])
                else:
                    inst.evidence_session_ids = list(evidence)
                qts, qts_rule = query_timestamp(evidence, ego_session_map.get(inst.ego_agent_id), session_ends)
                inst.metadata["query_timestamp"] = qts
                inst.metadata["query_timestamp_rule"] = qts_rule
                inst.metadata["query_modality"] = "online"

        # ── Stage 11: Write outputs ───────────────────────────────────────
        timer.start("stage_11_write_outputs")
        log.info("regen_eval: Writing outputs...")
        self._write_outputs(
            corpus, eval_instances, ego_data, report,
            ego_session_map=ego_session_map,
        )

        timer.stop()
        duration = time.monotonic() - start_time
        report["duration_seconds"] = round(duration, 2)
        report["llm_calls"] = llm_client.call_count
        report["token_usage"] = llm_client.get_usage_stats()
        report["stage_timings"] = timer.to_dict()
        report["generated_at"] = datetime.now(timezone.utc).isoformat()

        llm_client.write_timing_log(self.output_dir / "llm_timing.jsonl")
        write_json_report(self.output_dir / "pipeline_report.json", report)
        log.info("regen_eval complete in %.1fs. Output: %s", duration, self.output_dir)
        return report

    # Minimum number of pairwise configs to accumulate before flushing a batch.
    _BATCH_FLUSH_THRESHOLD = 16

    # ------------------------------------------------------------------
    # PA day-plan helpers (used by _run_scheduled_simulation)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pa_day_plan(
        specs: list,
        agents: Dict[str, EntityAgent],
        day_keys: List[int],
        pa_cfg: "PersonAgentConfig",
        topic_selector: Any,
        rng: np.random.Generator,
    ) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """Pre-compute per-agent-per-day PA allocation.

        *day_keys* are the actual day indices from the PP day buckets.

        Returns ``{day_idx: {agent_id: {"activity_topics": [...],
        "n_reflection": int, "n_probe": int}}}``
        """
        n_days = max(len(day_keys), 1)

        # Count PP specs per agent
        agent_pp_count: Dict[str, int] = {}
        for spec in specs:
            for pid in spec.participants:
                if pid in agents:
                    agent_pp_count[pid] = agent_pp_count.get(pid, 0) + 1
        if not agent_pp_count:
            agent_pp_count = {aid: 1 for aid in agents}

        # Count only non-facet PP specs for PA allocation (facet routines
        # have session_type="routine" and should not inflate PA count).
        n_pp_real = sum(
            1 for s in specs
            if getattr(s, "session_type", None) != "routine"
        )
        if n_pp_real == 0:
            n_pp_real = len(specs)          # fallback: no facet specs at all
        capped_ratio = min(pa_cfg.ratio, 1.0)   # PA ≤ PP (1:1 max)
        n_pa = max(1, int(round(n_pp_real * capped_ratio)))

        # 3-way split
        if pa_cfg.activity_reflection_probe_split:
            split = pa_cfg.activity_reflection_probe_split
            n_activity = max(1, int(round(n_pa * split[0])))
            n_reflection = max(0, int(round(n_pa * split[1])))
            n_probe = n_pa - n_activity - n_reflection
        else:
            n_activity = max(1, int(round(n_pa * pa_cfg.activity_reflection_split)))
            n_reflection = n_pa - n_activity
            n_probe = 0

        log.info(
            "PA day-plan: %d total (%d activity, %d reflection, %d probe) "
            "from %d real-PP specs (%d total specs incl. facet) across %d days",
            n_pa, n_activity, n_reflection, n_probe, n_pp_real, len(specs), n_days,
        )

        total_pp = sum(agent_pp_count.values())
        agent_ids_sorted = sorted(agent_pp_count.keys())

        # Per-agent allocation
        agent_n_act: Dict[str, int] = {}
        agent_n_ref: Dict[str, int] = {}
        agent_n_prb: Dict[str, int] = {}
        act_rem, ref_rem, prb_rem = n_activity, n_reflection, n_probe
        for i, aid in enumerate(agent_ids_sorted):
            frac = agent_pp_count[aid] / total_pp
            if i == len(agent_ids_sorted) - 1:
                agent_n_act[aid] = act_rem
                agent_n_ref[aid] = ref_rem
                agent_n_prb[aid] = prb_rem
            else:
                a = max(0, int(round(n_activity * frac)))
                r = max(0, int(round(n_reflection * frac)))
                p = max(0, int(round(n_probe * frac)))
                agent_n_act[aid] = min(a, act_rem)
                agent_n_ref[aid] = min(r, ref_rem)
                agent_n_prb[aid] = min(p, prb_rem)
                act_rem -= agent_n_act[aid]
                ref_rem -= agent_n_ref[aid]
                prb_rem -= agent_n_prb[aid]

        # Distribute per day: spread each agent's allocation evenly across day_keys
        day_plan: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for aid in agent_ids_sorted:
            total_act = agent_n_act.get(aid, 0)
            total_ref = agent_n_ref.get(aid, 0)
            total_prb = agent_n_prb.get(aid, 0)
            if total_act + total_ref + total_prb == 0:
                continue

            # Pre-select all activity topics for this agent
            all_topics = topic_selector.select_topics(agents[aid].persona, n=total_act) if total_act > 0 else []

            # Distribute counts across days using Bresenham-style spread
            def _spread(total: int, n: int) -> List[int]:
                """Distribute *total* items across *n* buckets as evenly as possible."""
                base, extra = divmod(total, n)
                return [base + (1 if i < extra else 0) for i in range(n)]

            act_per_day = _spread(total_act, n_days)
            ref_per_day = _spread(total_ref, n_days)
            prb_per_day = _spread(total_prb, n_days)

            act_assigned = 0
            for di, day_key in enumerate(day_keys):
                d_act = act_per_day[di]
                d_ref = ref_per_day[di]
                d_prb = prb_per_day[di]

                if d_act + d_ref + d_prb == 0:
                    continue

                entry: Dict[str, Any] = {
                    "activity_topics": all_topics[act_assigned:act_assigned + d_act],
                    "n_reflection": d_ref,
                    "n_probe": d_prb,
                }
                day_plan.setdefault(day_key, {})[aid] = entry
                act_assigned += d_act

        return day_plan

    @staticmethod
    def _build_day_pa_configs(
        day_idx: int,
        day_agent_plan: Dict[str, Dict[str, Any]],
        agents: Dict[str, EntityAgent],
        session_pool: List[Session],
        probe_selector: Any,
        rng: np.random.Generator,
        day_ts: float,
        day_length: float = 1.0,
        topic_selector: Any = None,
    ) -> List[Dict]:
        """Build activity + reflection + probe configs for one day."""
        configs: List[Dict] = []

        # Index sessions by agent for reflection targets
        agent_session_pool: Dict[str, List[Session]] = {}
        for sess in session_pool:
            for pid in sess.participants:
                agent_session_pool.setdefault(pid, []).append(sess)

        for aid, plan in sorted(day_agent_plan.items()):
            agent = agents.get(aid)
            if agent is None:
                continue

            # Activity configs
            for topic in plan.get("activity_topics", []):
                configs.append({
                    "agent": agent,
                    "session_type": "activity",
                    "topic": topic,
                    "session_seed": int(rng.integers(0, 2**31)),
                })

            # Reflection configs — pick targets from prior-day session pool
            n_ref = plan.get("n_reflection", 0)
            pool = agent_session_pool.get(aid, [])
            if n_ref > 0 and pool:
                n_ref = min(n_ref, len(pool))
                target_indices = rng.choice(len(pool), size=n_ref, replace=False)
                for idx in target_indices:
                    configs.append({
                        "agent": agent,
                        "session_type": "reflection",
                        "target_session": pool[idx],
                        "session_seed": int(rng.integers(0, 2**31)),
                    })

            # Probe configs — fall back to activity if knowledge is empty
            n_prb = plan.get("n_probe", 0)
            if n_prb > 0:
                if agent.knowledge.facts:
                    probes = probe_selector.select_probes(aid, agent.knowledge, n=n_prb)
                    for probe_spec in probes:
                        configs.append({
                            "agent": agent,
                            "session_type": "probe",
                            "probe_spec": probe_spec,
                            "session_seed": int(rng.integers(0, 2**31)),
                        })
                else:
                    # No facts yet — redistribute probe slots to activity
                    fallback_topics = topic_selector.select_topics(agent.persona, n=n_prb) if topic_selector else []
                    for topic in fallback_topics:
                        configs.append({
                            "agent": agent,
                            "session_type": "activity",
                            "topic": topic,
                            "session_seed": int(rng.integers(0, 2**31)),
                        })

        # Assign timestamps spread evenly within the day
        n = len(configs)
        for i, cfg in enumerate(configs):
            cfg["timestamp"] = day_ts + day_length * (0.1 + 0.8 * i / max(n, 1))

        return configs

    def _run_faster_batch_simulation(
        self,
        daily_scheduler: Any,  # DailyScheduler (import at call site)
        engine: DialogueEngine,
        scheduler: BatchScheduler,
        agents: Dict[str, EntityAgent],
        encounter_windows: List[EncounterWindow],
        events: List[WorldEvent],
        *,
        skeleton_logs: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> List[Session]:
        """Faster-batch simulation: races pairwise conversations.

        Over-provisions pairwise configs by overprovision_ratio, races them
        concurrently, keeps the first n_target to complete, and harvests the
        best truncated sessions for next-turn prediction cloze ground truth.
        """
        log.info("faster-batch: racing pairwise conversations")
        return self._run_scheduled_simulation(
            daily_scheduler, engine, scheduler, agents,
            encounter_windows, events,
            skeleton_logs=skeleton_logs,
            llm_client=llm_client,
            racing=True,
        )

    def _run_scheduled_simulation(
        self,
        daily_scheduler: Any,  # DailyScheduler (import at call site)
        engine: DialogueEngine,
        scheduler: BatchScheduler,
        agents: Dict[str, EntityAgent],
        encounter_windows: List[EncounterWindow],
        events: List[WorldEvent],
        *,
        skeleton_logs: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
        racing: bool = False,
    ) -> List[Session]:
        """Run the scheduler-driven simulation loop with day-level batching.

        Each day generates ``n_agents * sessions_per_day_per_agent`` sessions
        (PP + PA combined) with fixed proportions.  The loop runs until
        ``time_range`` is exhausted or ``target_tokens`` is reached.

        Sessions are streamed to corpus_sessions.jsonl as each day completes.
        """
        from MASim.core.scheduler import ConversationSpec, build_role_context_prefix

        t_start, t_end = self.cfg.time_range
        t_span = max(t_end - t_start, 1.0)
        day_length = self.cfg.day_length
        if day_length <= 0:
            day_length = t_span / 30.0

        n_agents = len(agents)
        use_per_day = self.cfg.sessions_per_day_per_agent > 0
        total_per_day = n_agents * self.cfg.sessions_per_day_per_agent if use_per_day else 0

        # Facet scheduler (created once, called per day)
        facet_scheduler = None
        if self.cfg.facets_enabled:
            from MASim.generation.facet_scheduler import FacetScheduler
            facet_scheduler = FacetScheduler(
                agents=agents,
                graph=daily_scheduler.graph,
                rng=np.random.default_rng(self.cfg.seed + 333),
            )

        # Lookup tables for role context injection
        group_map = {g.group_id: g for g in daily_scheduler.groups}
        location_map = daily_scheduler._location_map
        agent_role_in_group: Dict[str, Dict[str, str]] = {}
        for g in daily_scheduler.groups:
            for member_id, role_name in g.members.items():
                agent_role_in_group.setdefault(member_id, {})[g.group_id] = role_name

        sessions: List[Session] = []
        rng = np.random.default_rng(self.cfg.seed)

        # Open corpus_sessions.jsonl for streaming
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stream_path = self.output_dir / "corpus_sessions.jsonl"
        _stream_fh = stream_path.open("w", encoding="utf-8")

        # Counters for the report
        n_group_meeting = 0
        n_event = 0
        n_encounter = 0
        n_remote = 0
        n_facet_routine = 0
        n_pa_activity = 0
        n_pa_reflection = 0
        n_pa_probe = 0
        cumulative_tokens = 0

        # PA setup
        pa_engine = None
        session_pool: List[Session] = []
        topic_selector = None
        probe_selector = None
        pa_rng = None

        if self.cfg.person_agent.enabled and llm_client is not None:
            from MASim.core.person_agent_engine import PersonAgentEngine
            from MASim.generation.activity_topic_selector import ActivityTopicSelector
            from MASim.generation.probe_selector import ProbeSelector

            pa_cfg = self.cfg.person_agent
            pa_engine = PersonAgentEngine(
                llm_client,
                max_turns=pa_cfg.max_turns,
                min_turns=pa_cfg.min_turns,
                probe_max_turns=pa_cfg.probe_max_turns,
                probe_min_turns=pa_cfg.probe_min_turns,
                no_token_limit=self.cfg.no_token_limit,
            )
            topic_selector = ActivityTopicSelector(seed=self.cfg.seed + 888)
            probe_selector = ProbeSelector(seed=self.cfg.seed + 999)
            pa_rng = np.random.default_rng(self.cfg.seed + 777)

        def _build_pairwise_config(spec) -> Optional[Dict[str, Any]]:
            """Build a pairwise conversation config from a spec."""
            if len(spec.participants) < 2:
                return None
            agent_a_id, agent_b_id = spec.participants[0], spec.participants[1]
            if agent_a_id not in agents or agent_b_id not in agents:
                return None
            _group = group_map.get(spec.group_id)
            _location = location_map.get(spec.location_id)
            role_a = agent_role_in_group.get(agent_a_id, {}).get(spec.group_id, "")
            role_b = agent_role_in_group.get(agent_b_id, {}).get(spec.group_id, "")

            prefix_a = build_role_context_prefix(
                group=_group, role_name=role_a, location=_location,
            )
            prefix_b = build_role_context_prefix(
                group=_group, role_name=role_b, location=_location,
            )
            if spec.routine_prompt:
                routine_line = f"Scenario: {spec.routine_prompt}\n"
                prefix_a = routine_line + prefix_a
                prefix_b = routine_line + prefix_b

            config = {
                "agent_a": agents[agent_a_id],
                "agent_b": agents[agent_b_id],
                "event": spec.event,
                "timestamp": spec.start_time,
                "modality": spec.modality,
                "session_seed": int(rng.integers(0, 2**31)),
                "context_prefix_a": prefix_a,
                "context_prefix_b": prefix_b,
            }
            if spec.interest_domain:
                config["interest_domain"] = spec.interest_domain
            return config

        def _build_group_event(spec):
            """Build or retrieve the event for a group conversation."""
            event = spec.event
            _group = group_map.get(spec.group_id)
            if event is None:
                from MASim.core.schema import WorldEvent as _WE, EventCategory
                grp_desc = (
                    f"{_group.name} meeting — {_group.description}"
                    if _group and _group.description
                    else f"Regular meeting of {_group.name if _group else spec.group_id}"
                )
                event = _WE(
                    event_id=f"grp_ctx_{spec.spec_id}",
                    timestamp=spec.start_time,
                    event_type="group_meeting",
                    content=grp_desc,
                    visibility_mask=set(spec.participants),
                    category=EventCategory.COMMUNITY,
                    location_id=spec.location_id,
                )
            return event

        # ================================================================
        # Per-day generation + execution loop
        # ================================================================
        day_idx = 0
        try:
            while True:
                # --- Termination check ---
                day_start = t_start + day_idx * day_length
                if self.cfg.target_tokens > 0:
                    if cumulative_tokens >= self.cfg.target_tokens:
                        log.info("Target tokens reached (%d >= %d), stopping.",
                                 cumulative_tokens, self.cfg.target_tokens)
                        break
                else:
                    if day_start >= t_end:
                        break
                day_end = day_start + day_length
                day_time_range = (day_start, day_end)

                # --- Compute PP and PA targets for this day ---
                if use_per_day:
                    r = float(rng.uniform(*self.cfg.pa_day_ratio_range))
                    n_pp_target = max(1, int(total_per_day / (1.0 + r)))
                    n_pa_target_day = total_per_day - n_pp_target
                else:
                    # Legacy mode: use max_sessions / estimated_days
                    est_days = max(1, int(np.ceil(t_span / day_length)))
                    n_pp_target = max(1, self.cfg.max_sessions // est_days)
                    n_pa_target_day = n_pp_target  # 1:1

                # --- PP proportion targets (excluding facet_routine — generated separately) ---
                props = self.cfg.pp_day_proportions
                # facet_routine share handled by FacetScheduler; remaining proportions
                # are rescaled for DailyScheduler
                facet_frac = props.get("facet_routine", 0.40)
                n_facet_target = int(n_pp_target * facet_frac)
                n_non_facet_target = n_pp_target - n_facet_target
                non_facet_total = sum(v for k, v in props.items() if k != "facet_routine")
                scheduler_targets = {}
                if non_facet_total > 0:
                    for k, v in props.items():
                        if k != "facet_routine":
                            scheduler_targets[k] = max(1, int(n_non_facet_target * v / non_facet_total))

                # --- Generate PP specs for this day ---
                # 1. DailyScheduler specs (event, encounter, remote)
                day_specs = daily_scheduler.build_day_schedule(
                    encounter_windows, events,
                    day_time_range=day_time_range,
                    day_length=day_length,
                    targets=scheduler_targets,
                    skeleton_logs=skeleton_logs,
                )

                # 2. Facet routine specs
                facet_specs: List[ConversationSpec] = []
                if facet_scheduler is not None:
                    facet_specs = facet_scheduler.build_routine_specs_for_day(
                        day_start=day_start, day_length=day_length,
                    )
                    # Cap or sample to target
                    if len(facet_specs) > n_facet_target:
                        _frng = np.random.default_rng(rng.integers(2**63))
                        indices = _frng.choice(len(facet_specs), n_facet_target, replace=False)
                        facet_specs = [facet_specs[i] for i in sorted(indices)]

                # 3. Combine and fill shortfall with encounter specs
                all_pp_specs = day_specs + facet_specs
                shortfall = n_pp_target - len(all_pp_specs)
                if shortfall > 0 and daily_scheduler.graph is not None:
                    extra = daily_scheduler._phase_c2_graph_encounters(
                        existing_specs=all_pp_specs,
                        target_count=shortfall,
                        time_range=day_time_range,
                        max_sessions_per_day=0,
                        day_length=day_length,
                    )
                    all_pp_specs.extend(extra)

                all_pp_specs.sort(key=lambda s: s.start_time)

                # --- Minimum-session guarantee (day 0 only) ---
                if day_idx == 0:
                    participating_ids: set = set()
                    for spec in all_pp_specs:
                        participating_ids.update(spec.participants)
                    missing_ids = set(agents.keys()) - participating_ids
                    if missing_ids:
                        _grng = np.random.default_rng(self.cfg.seed + 7)
                        graph = daily_scheduler.graph
                        for mid in sorted(missing_ids):
                            partner = None
                            if graph is not None and mid in graph:
                                neighbors = sorted(
                                    graph.neighbors(mid),
                                    key=lambda n: graph[mid][n].get("weight", 0),
                                    reverse=True,
                                )
                                for n in neighbors:
                                    if n in agents:
                                        partner = n
                                        break
                            if partner is None:
                                candidates = [a for a in agents if a != mid]
                                partner = candidates[int(_grng.integers(len(candidates)))]
                            t_mid = day_start + day_length * float(_grng.uniform(0.1, 0.5))
                            all_pp_specs.append(ConversationSpec(
                                spec_id=f"guarantee_{mid}",
                                participants=[mid, partner],
                                start_time=t_mid,
                                end_time=t_mid + 1.0,
                                trigger="encounter",
                                modality="face_to_face",
                            ))
                        log.info("Minimum-session guarantee: injected %d specs", len(missing_ids))

                # --- Classify PP specs into group vs pairwise ---
                group_specs_day = []
                pairwise_configs: List[Dict] = []
                pairwise_spec_meta: List[Dict[str, str]] = []

                for spec in all_pp_specs:
                    if spec.trigger == "group_meeting":
                        n_group_meeting += 1
                    elif spec.trigger == "event":
                        n_event += 1
                    elif spec.trigger == "remote":
                        n_remote += 1
                    elif spec.trigger == "routine_facet":
                        n_facet_routine += 1
                    else:
                        n_encounter += 1

                    if spec.is_group and len(spec.participants) >= 3:
                        group_specs_day.append(spec)
                    else:
                        config = _build_pairwise_config(spec)
                        if config is not None:
                            pairwise_configs.append(config)
                            pairwise_spec_meta.append({
                                "trigger": spec.trigger,
                                "location_id": spec.location_id,
                                "group_id": spec.group_id,
                                "interest_domain": getattr(spec, "interest_domain", ""),
                                "session_type": getattr(spec, "session_type", ""),
                            })

                day_sessions: List[Session] = []

                # --- Build PA configs for this day ---
                pa_configs: List[Dict] = []
                pa_day_sessions: List[Session] = []
                if pa_engine and n_pa_target_day > 0:
                    # 1:1:1 split for activity:reflection:probe
                    n_each = n_pa_target_day // 3
                    n_act = n_each
                    n_ref = n_each
                    n_prb = n_pa_target_day - 2 * n_each

                    # Distribute across agents proportionally
                    day_agent_plan: Dict[str, Dict[str, Any]] = {}
                    agent_ids = sorted(agents.keys())
                    n_ag = len(agent_ids)
                    for ai, aid in enumerate(agent_ids):
                        a_act = n_act // n_ag + (1 if ai < n_act % n_ag else 0)
                        a_ref = n_ref // n_ag + (1 if ai < n_ref % n_ag else 0)
                        a_prb = n_prb // n_ag + (1 if ai < n_prb % n_ag else 0)
                        if a_act + a_ref + a_prb == 0:
                            continue
                        topics = topic_selector.select_topics(agents[aid].persona, n=a_act) if a_act > 0 else []
                        day_agent_plan[aid] = {
                            "activity_topics": topics,
                            "n_reflection": a_ref,
                            "n_probe": a_prb,
                        }

                    day_ts = day_start
                    pa_configs = self._build_day_pa_configs(
                        day_idx, day_agent_plan, agents, session_pool,
                        probe_selector, pa_rng, day_ts,
                        day_length=day_length,
                        topic_selector=topic_selector,
                    )

                log.info(
                    "Day %d: %d PP specs (%d facet, %d event, %d encounter, %d remote, %d group) "
                    "+ %d PA configs",
                    day_idx, len(all_pp_specs),
                    len(facet_specs), len([s for s in day_specs if s.trigger == "event"]),
                    len([s for s in day_specs if s.trigger in ("encounter",)]) +
                    len([s for s in all_pp_specs if s.spec_id.startswith("spec_gf2f_") or s.spec_id.startswith("guarantee_")]),
                    len([s for s in day_specs if s.trigger == "remote"]),
                    len(group_specs_day), len(pa_configs),
                )

                # --- 2. Prepare group conversation callables ---
                group_call_args: List[Dict[str, Any]] = []
                for spec in group_specs_day:
                    group_agents = [agents[aid] for aid in spec.participants if aid in agents]
                    if len(group_agents) < 2:
                        continue

                    event = _build_group_event(spec)
                    _group = group_map.get(spec.group_id)
                    _location = location_map.get(spec.location_id)

                    group_context_prefixes: Dict[str, str] = {}
                    for pid in spec.participants:
                        role = agent_role_in_group.get(pid, {}).get(spec.group_id, "")
                        group_context_prefixes[pid] = build_role_context_prefix(
                            group=_group, role_name=role, location=_location,
                        )

                    group_call_args.append({
                        "agents": group_agents,
                        "event": event,
                        "modality": spec.modality,
                        "session_seed": int(rng.integers(0, 2**31)),
                        "context_prefixes": group_context_prefixes,
                        "spec": spec,
                    })

                # --- 3. Log and launch all workstreams concurrently ---
                n_pp = len(pairwise_configs)
                n_grp = len(group_call_args)
                n_pa = len(pa_configs)
                log.info(
                    "Day %d: launching %d pairwise + %d group + %d PA concurrently",
                    day_idx, n_pp, n_grp, n_pa,
                )

                from concurrent.futures import ThreadPoolExecutor, Future
                from tqdm import tqdm
                import sys as _sys

                truncated_sess: List[Session] = []

                # --- Pooled PP+PA racing setup ---
                n_pp_pa = n_pp + n_pa
                pooled_racing = self.cfg.day_completion_ratio < 1.0 and n_pp_pa > 0

                # --- Progress bar for this day ---
                n_group = len(group_call_args)
                pbar_total = n_group + n_pp_pa
                pbar = tqdm(
                    total=pbar_total,
                    desc=f"Day {day_idx}",
                    unit="sess",
                    bar_format=(
                        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                    ),
                    dynamic_ncols=True,
                    file=_sys.stdout,
                    mininterval=1.0,
                )
                _pp_done = [0]
                _pa_done = [0]
                _pbar_lock = __import__("threading").Lock()

                def _update_pbar(kind: str):
                    with _pbar_lock:
                        if kind == "pp":
                            _pp_done[0] += 1
                        elif kind == "pa":
                            _pa_done[0] += 1
                        pbar.set_postfix_str(
                            f"G={n_group} PP={_pp_done[0]}/{n_pp} PA={_pa_done[0]}/{n_pa}",
                            refresh=False,
                        )
                        pbar.update(1)
                        pbar.refresh()

                if pooled_racing:
                    import threading as _thr
                    n_pp_target = max(1, int(n_pp * self.cfg.day_completion_ratio))
                    n_pa_target = max(1, int(n_pa * self.cfg.day_completion_ratio)) if n_pa > 0 else 0
                    pp_stop = _thr.Event()
                    pa_stop = _thr.Event()
                    _pp_counter_lock = _thr.Lock()
                    _pa_counter_lock = _thr.Lock()
                    _pp_n_done = [0]
                    _pa_n_done = [0]

                    def _on_pp_done():
                        _update_pbar("pp")
                        with _pp_counter_lock:
                            _pp_n_done[0] += 1
                            if _pp_n_done[0] >= n_pp_target:
                                pp_stop.set()

                    def _on_pa_done():
                        _update_pbar("pa")
                        with _pa_counter_lock:
                            _pa_n_done[0] += 1
                            if _pa_n_done[0] >= n_pa_target:
                                pa_stop.set()

                    log.info(
                        "Day %d: independent racing PP %d (target=%d) + PA %d (target=%d), ratio=%.0f%%",
                        day_idx, n_pp, n_pp_target, n_pa, n_pa_target,
                        self.cfg.day_completion_ratio * 100,
                    )
                else:
                    pp_stop = None
                    pa_stop = None
                    _on_pp_done = lambda: _update_pbar("pp")
                    _on_pa_done = lambda: _update_pbar("pa")

                with ThreadPoolExecutor() as day_exec:
                    # Submit group conversations — each thread runs the conversation
                    # AND its finalize (NER/mood/impressions) back-to-back, so
                    # finalize starts as soon as each group conv finishes.
                    group_futures: List[tuple] = []  # (Future, spec)

                    def _run_group_conv_and_finalize(ga, _day_idx, _dry_run):
                        sess = engine.simulate_group_conversation(
                            ga["agents"], ga["event"], ga["event"].timestamp,
                            dry_run=_dry_run,
                            modality=ga["modality"],
                            session_seed=ga["session_seed"],
                            context_prefixes=ga["context_prefixes"],
                            call_tags={"day_idx": _day_idx},
                            defer_finalize=True,
                        )
                        if sess is not None and not _dry_run:
                            engine.unified_batch_finalize(
                                pp_sessions=[sess],
                                pa_sessions=[],
                                agent_map=agents,
                                call_tags={"day_idx": _day_idx},
                            )
                        return sess

                    for ga in group_call_args:
                        fut = day_exec.submit(
                            _run_group_conv_and_finalize,
                            ga, day_idx, self.cfg.dry_run,
                        )
                        group_futures.append((fut, ga["spec"]))

                    # Submit PP wavefront (one thread — internally uses its own executor)
                    pp_future: Optional[Future] = None
                    if pairwise_configs:
                        if pooled_racing:
                            # Independent racing: PP has its own stop event
                            pp_future = day_exec.submit(
                                engine.simulate_conversations_wavefront,
                                pairwise_configs, dry_run=self.cfg.dry_run,
                                call_tags={"day_idx": day_idx},
                                stop_event=pp_stop,
                                on_complete_callback=_on_pp_done,
                            )
                        elif racing:
                            n_target = len(pairwise_configs)
                            n_extra = int(n_target * (self.cfg.overprovision_ratio - 1.0))
                            n_truncated_keep = max(1, int(n_target * self.cfg.cutoff_ratio))

                            extra_configs = self._sample_extra_pairwise_configs(
                                daily_scheduler, agents, pairwise_configs, n_extra,
                                day_idx, rng, group_map, location_map, agent_role_in_group,
                            )
                            all_configs = pairwise_configs + extra_configs
                            log.info(
                                "Day %d: racing %d PP configs (%d target + %d extra), "
                                "keep %d truncated",
                                day_idx, len(all_configs), n_target, len(extra_configs),
                                n_truncated_keep,
                            )
                            pp_future = day_exec.submit(
                                engine.simulate_conversations_wavefront,
                                all_configs,
                                dry_run=self.cfg.dry_run,
                                call_tags={"day_idx": day_idx},
                                n_target=n_target,
                                n_truncated_keep=n_truncated_keep,
                                on_complete_callback=_on_pp_done,
                            )
                        else:
                            pp_future = day_exec.submit(
                                engine.simulate_conversations_wavefront,
                                pairwise_configs, dry_run=self.cfg.dry_run,
                                call_tags={"day_idx": day_idx},
                                on_complete_callback=_on_pp_done,
                            )

                    # Submit PA wavefront (one thread — internally uses its own executor)
                    pa_future: Optional[Future] = None
                    if pa_configs and pa_engine:
                        pa_future = day_exec.submit(
                            pa_engine.simulate_pa_sessions_wavefront,
                            pa_configs, dry_run=self.cfg.dry_run,
                            call_tags={"day_idx": day_idx},
                            stop_event=pa_stop,
                            on_complete_callback=_on_pa_done,
                        )

                    # --- Collect group results (already finalized in their threads) ---
                    group_sessions: List[Session] = []
                    for fut, spec in group_futures:
                        sess = fut.result()
                        if sess is not None:
                            sess.metadata["trigger"] = spec.trigger
                            if spec.group_id:
                                sess.group_id = spec.group_id
                            sess.location_id = spec.location_id
                            group_sessions.append(sess)
                            _update_pbar("group")
                    day_sessions.extend(group_sessions)

                    # --- Collect PP results ---
                    def _apply_spec_meta(sess, meta):
                        """Apply spec metadata (trigger, location, domain) to a session."""
                        sess.metadata["trigger"] = meta.get("trigger", "encounter")
                        if not sess.location_id:
                            sess.location_id = meta.get("location_id", "")
                        if meta.get("group_id"):
                            sess.group_id = meta["group_id"]
                        if meta.get("interest_domain"):
                            sess.interest_domain = meta["interest_domain"]
                        if meta.get("session_type"):
                            sess.session_type = meta["session_type"]

                    if pp_future is not None:
                        completed_sess, truncated_sess = pp_future.result()
                        if racing:
                            for i, sess in enumerate(completed_sess):
                                if i < len(pairwise_spec_meta):
                                    _apply_spec_meta(sess, pairwise_spec_meta[i])
                                day_sessions.append(sess)
                            for sess in truncated_sess:
                                sess.metadata.setdefault("trigger", "racing_extra")
                                day_sessions.append(sess)
                        else:
                            for i, sess in enumerate(completed_sess):
                                if i < len(pairwise_spec_meta):
                                    _apply_spec_meta(sess, pairwise_spec_meta[i])
                                day_sessions.append(sess)

                    # --- Collect PA results ---
                    if pa_future is not None:
                        all_pa_sessions = pa_future.result()
                        if racing and 'n_pa_target' in locals():
                            # Racing: keep top n_pa_target by quality
                            all_pa_sessions.sort(
                                key=lambda s: (
                                    len(s.turns),
                                    sum(len(t.text.split()) for t in s.turns),
                                ),
                                reverse=True,
                            )
                            pa_day_sessions = all_pa_sessions[:n_pa_target]
                            n_discarded = len(all_pa_sessions) - len(pa_day_sessions)
                            log.info(
                                "Day %d: PA racing done, kept %d/%d (discarded %d stragglers)",
                                day_idx, len(pa_day_sessions), len(all_pa_sessions), n_discarded,
                            )
                        else:
                            pa_day_sessions = all_pa_sessions

                # Close progress bar for this day
                pbar.close()

                # --- 4. Post-wavefront: generate next-turns for truncated (racing only) ---
                if racing and truncated_sess:
                    engine.generate_next_turns_for_truncated(
                        truncated_sess, agents,
                        dry_run=self.cfg.dry_run,
                        call_tags={"day_idx": day_idx},
                    )

                # --- 5. Finalize PP + PA (group sessions already finalized in their threads) ---
                pp_only = [s for s in day_sessions if s not in group_sessions]
                if (pp_only or pa_day_sessions) and not self.cfg.dry_run:
                    engine.unified_batch_finalize(
                        pp_sessions=pp_only,
                        pa_sessions=pa_day_sessions,
                        agent_map=agents,
                        call_tags={"day_idx": day_idx},
                    )

                # Store completed sessions on agents for memory context injection
                if self.cfg.memory_context_sessions:
                    for sess in day_sessions + pa_day_sessions:
                        for pid in sess.participants:
                            if pid in agents:
                                agents[pid].store_completed_session(sess)

                # Route sessions to interest-facet memory + build daily bulletin
                if self.cfg.facets_enabled:
                    for sess in day_sessions + pa_day_sessions:
                        for pid in sess.participants:
                            if pid in agents:
                                agents[pid].route_session_to_facet(sess)
                    for agent in agents.values():
                        agent.build_daily_bulletin()

                # Count PA types
                for s in pa_day_sessions:
                    st = s.metadata.get("session_type", "")
                    if st == "person_agent_activity":
                        n_pa_activity += 1
                    elif st == "person_agent_reflection":
                        n_pa_reflection += 1
                    elif st == "person_agent_probe":
                        n_pa_probe += 1

                # Accumulate into session_pool (for next day's reflections)
                session_pool.extend(day_sessions)
                session_pool.extend(pa_day_sessions)

                # Stream completed day to disk (PP + PA)
                for sess in day_sessions:
                    sessions.append(sess)
                    _stream_fh.write(json.dumps(sess.to_dict(), ensure_ascii=False) + "\n")
                for sess in pa_day_sessions:
                    sessions.append(sess)
                    _stream_fh.write(json.dumps(sess.to_dict(), ensure_ascii=False) + "\n")
                _stream_fh.flush()

                # Track token count for target_tokens mode
                all_day = day_sessions + pa_day_sessions
                if self.cfg.target_tokens > 0 and all_day:
                    cumulative_tokens += _count_corpus_tokens(all_day)

                total_day = len(all_day)
                if total_day:
                    log.info(
                        "Day %d complete: %d PP + %d PA sessions written to corpus"
                        " (cumulative tokens: %d)",
                        day_idx, len(day_sessions), len(pa_day_sessions),
                        cumulative_tokens,
                    )

                # Per-day checkpoint: save agent states + progress marker
                _ckpt_path = self.output_dir / "day_checkpoint.json"
                _ckpt = {
                    "completed_day": day_idx,
                    "n_sessions": len(sessions),
                    "cumulative_tokens": cumulative_tokens,
                    "agent_moods": {aid: a.current_mood for aid, a in agents.items()},
                    "agent_knowledge_sizes": {
                        aid: len(a.knowledge.facts) if hasattr(a.knowledge, 'facts') else 0
                        for aid, a in agents.items()
                    },
                }
                _ckpt_path.write_text(json.dumps(_ckpt, indent=2, default=str))
                log.info("Day %d checkpoint saved (%d sessions, %d tokens)",
                         day_idx, len(sessions), cumulative_tokens)

                day_idx += 1

        finally:
            _stream_fh.close()

        n_pa_total = n_pa_activity + n_pa_reflection + n_pa_probe
        log.info(
            "Scheduled simulation produced %d sessions across %d days "
            "(group_meeting=%d, event=%d, encounter=%d, facet_routine=%d, "
            "remote=%d, pa_activity=%d, pa_reflection=%d, pa_probe=%d)",
            len(sessions), day_idx,
            n_group_meeting, n_event, n_encounter, n_facet_routine,
            n_remote, n_pa_activity, n_pa_reflection, n_pa_probe,
        )
        return sessions

    def _sample_extra_pairwise_configs(
        self,
        daily_scheduler: Any,
        agents: Dict[str, EntityAgent],
        existing_configs: List[Dict],
        n_extra: int,
        day_idx: int,
        rng: np.random.Generator,
        group_map: Dict[str, Any],
        location_map: Dict[str, Any],
        agent_role_in_group: Dict[str, Dict[str, str]],
    ) -> List[Dict]:
        """Sample extra pairwise configs from social graph edges not already used.

        Edges are weighted by graph weight (DESC) with small random jitter.
        Each extra config gets a random timestamp within the day, random
        modality, and random seed.
        """
        from MASim.core.scheduler import build_role_context_prefix

        if n_extra <= 0:
            return []

        # Build set of existing pairs
        existing_pairs: set = set()
        for cfg in existing_configs:
            pair = frozenset([cfg["agent_a"].agent_id, cfg["agent_b"].agent_id])
            existing_pairs.add(pair)

        # Collect social graph edges not already used
        graph = daily_scheduler.graph
        if graph is None:
            return []

        candidate_edges = []
        for u, v, data in graph.edges(data=True):
            pair = frozenset([u, v])
            if pair not in existing_pairs and u in agents and v in agents:
                weight = data.get("weight", 1.0)
                candidate_edges.append((u, v, weight))

        if not candidate_edges:
            return []

        # Sort by weight DESC with small random jitter for variety
        candidate_edges.sort(
            key=lambda e: e[2] + rng.uniform(-0.1, 0.1),
            reverse=True,
        )

        # Compute day time range
        t_start, t_end = self.cfg.time_range
        day_length = self.cfg.day_length
        if day_length <= 0:
            t_span = max(t_end - t_start, 1.0)
            day_length = t_span / 30.0
        day_start = t_start + day_idx * day_length
        day_end = day_start + day_length

        configs: List[Dict] = []
        for u, v, weight in candidate_edges[:n_extra]:
            timestamp = float(rng.uniform(day_start, day_end))
            modality = _roll_modality(rng, self.cfg.modality_weights)

            # Find shared group (if any) for context prefix
            u_groups = agent_role_in_group.get(u, {})
            v_groups = agent_role_in_group.get(v, {})
            shared_groups = set(u_groups.keys()) & set(v_groups.keys())
            group_id = next(iter(shared_groups), "")
            _group = group_map.get(group_id)
            _location = location_map.get(
                _group.meeting_location if _group and hasattr(_group, "meeting_location") else "", None
            )

            role_u = u_groups.get(group_id, "")
            role_v = v_groups.get(group_id, "")

            configs.append({
                "agent_a": agents[u],
                "agent_b": agents[v],
                "event": None,
                "timestamp": timestamp,
                "modality": modality,
                "session_seed": int(rng.integers(0, 2**31)),
                "context_prefix_a": build_role_context_prefix(
                    group=_group, role_name=role_u, location=_location,
                ),
                "context_prefix_b": build_role_context_prefix(
                    group=_group, role_name=role_v, location=_location,
                ),
            })

        log.info(
            "Sampled %d extra pairwise configs from social graph (day %d)",
            len(configs), day_idx,
        )
        return configs

    @staticmethod
    def _sample_extra_pa_configs(
        existing_configs: List[Dict],
        n_extra: int,
        agents: Dict[str, "EntityAgent"],
        session_pool: List[Session],
        topic_selector: Any,
        rng: np.random.Generator,
        day_ts: float,
    ) -> List[Dict]:
        """Sample extra PA configs for racing (activity + reflection only, not probes).

        Strategy:
          - Pick agents proportionally to their existing allocation
          - Generate fresh activity topics for each picked agent
          - If an agent has prior sessions, sample reflection targets
          - Split extras ~60% activity / 40% reflection (matching typical PA split)
        """
        if n_extra <= 0:
            return []

        # Index past sessions per agent for reflection targets
        agent_sessions: Dict[str, List[Session]] = {}
        for sess in session_pool:
            for pid in sess.participants:
                agent_sessions.setdefault(pid, []).append(sess)

        # Collect candidate agents (those with existing PA configs)
        agent_ids_in_configs = list({c["agent"].agent_id for c in existing_configs})
        if not agent_ids_in_configs:
            return []

        # Split: 60% activity, 40% reflection
        n_act_extra = max(1, int(n_extra * 0.6))
        n_ref_extra = n_extra - n_act_extra

        extras: List[Dict] = []
        ts_offset = 0

        # Extra activity configs — pick random agents, generate new topics
        for _ in range(n_act_extra):
            aid = rng.choice(agent_ids_in_configs)
            agent = agents.get(aid)
            if agent is None:
                continue
            topics = topic_selector.select_topics(agent.persona, n=1)
            if not topics:
                continue
            extras.append({
                "agent": agent,
                "session_type": "activity",
                "topic": topics[0],
                "timestamp": day_ts + 0.5 + ts_offset * 0.01,
                "session_seed": int(rng.integers(0, 2**31)),
            })
            ts_offset += 1

        # Extra reflection configs — pick agents that have past sessions
        agents_with_history = [aid for aid in agent_ids_in_configs if agent_sessions.get(aid)]
        for _ in range(n_ref_extra):
            if not agents_with_history:
                break
            aid = rng.choice(agents_with_history)
            agent = agents.get(aid)
            if agent is None:
                continue
            pool = agent_sessions[aid]
            target = pool[int(rng.integers(0, len(pool)))]
            extras.append({
                "agent": agent,
                "session_type": "reflection",
                "target_session": target,
                "timestamp": day_ts + 0.6 + ts_offset * 0.01,
                "session_seed": int(rng.integers(0, 2**31)),
            })
            ts_offset += 1

        log.info("Sampled %d extra PA configs (day_ts=%.1f)", len(extras), day_ts)
        return extras

    def _run_simulation_legacy(
        self,
        broadcaster: WorldBroadcaster,
        engine: DialogueEngine,
        scheduler: BatchScheduler,
        agents: Dict[str, EntityAgent],
        events: List[WorldEvent],
    ) -> List[Session]:
        """Legacy event-driven simulation loop with day-level batching.

        Events are grouped by simulation "day" (controlled by ``day_length``).
        All conversations triggered within the same day are batched together
        and run in parallel.  Knowledge updates apply once at the end of each
        day, so intra-day sessions do not see each other's outcomes.

        Sessions are streamed to corpus_sessions.jsonl as each day completes.
        """
        sessions: List[Session] = []
        rng = np.random.default_rng(self.cfg.seed)

        # Compute day_length
        t_start, t_end = self.cfg.time_range
        t_span = max(t_end - t_start, 1.0)
        day_length = self.cfg.day_length
        if day_length <= 0:
            day_length = t_span / 30.0

        # Group events by day
        events_sorted = sorted(events, key=lambda e: e.timestamp)
        day_events: Dict[int, List[WorldEvent]] = {}
        for evt in events_sorted:
            day_idx = int((evt.timestamp - t_start) / day_length)
            day_events.setdefault(day_idx, []).append(evt)

        log.info(
            "Legacy day-batching: day_length=%.1f, %d events across %d days",
            day_length, len(events), len(day_events),
        )

        # Open corpus_sessions.jsonl for streaming
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stream_path = self.output_dir / "corpus_sessions.jsonl"
        _stream_fh = stream_path.open("w", encoding="utf-8")

        try:
            for day_idx in sorted(day_events.keys()):
                if len(sessions) >= self.cfg.max_sessions:
                    break

                day_evts = day_events[day_idx]
                day_sessions: List[Session] = []
                pending_configs: List[Dict] = []

                for event in day_evts:
                    if len(sessions) + len(pending_configs) >= self.cfg.max_sessions:
                        break

                    recipients = broadcaster.broadcast_event(event)
                    is_group_event = event.category.value in ("global", "community")

                    if is_group_event and len(recipients) >= 3 and rng.random() < 0.5:
                        group_size = min(int(rng.integers(3, 6)), len(recipients))
                        group_ids = list(rng.choice(
                            list(recipients), size=group_size, replace=False,
                        ))
                        group_agents = [agents[aid] for aid in group_ids]
                        group_modality = _roll_modality(rng, self.cfg.modality_weights)
                        group_seed = int(rng.integers(0, 2**31))
                        _day_tags_legacy = {"day_idx": day_idx}
                        tasks = [
                            lambda ga=group_agents, mod=group_modality, ss=group_seed, ev=event, dt=_day_tags_legacy: engine.simulate_group_conversation(
                                ga, ev, ev.timestamp,
                                dry_run=self.cfg.dry_run,
                                modality=mod,
                                session_seed=ss,
                                call_tags=dt,
                            )
                        ]
                        result = scheduler.schedule_wave(tasks, label="group conversation")
                        for sess in result.results:
                            if sess is not None:
                                day_sessions.append(sess)
                        continue

                    # Pairwise conversations
                    conversation_pairs = []
                    for aid in recipients:
                        partner = agents[aid].decide_to_converse(event)
                        if partner and partner in agents:
                            conversation_pairs.append((aid, partner))

                    seen = set()
                    for a, b in conversation_pairs:
                        key = tuple(sorted([a, b]))
                        if key not in seen:
                            seen.add(key)
                            pending_configs.append({
                                "agent_a": agents[a],
                                "agent_b": agents[b],
                                "event": event,
                                "timestamp": event.timestamp,
                                "modality": _roll_modality(rng, self.cfg.modality_weights),
                                "session_seed": int(rng.integers(0, 2**31)),
                            })

                # Run ALL pairwise conversations for this day in one batch
                if pending_configs:
                    log.info("Day %d: batching %d pairwise conversations", day_idx, len(pending_configs))
                    batch_sessions = engine.simulate_conversations_stepped(
                        pending_configs, dry_run=self.cfg.dry_run,
                        call_tags={"day_idx": day_idx},
                    )
                    day_sessions.extend(batch_sessions)

                # Apply knowledge updates ONCE at end of day
                if day_sessions and not self.cfg.dry_run:
                    engine.batch_finalize_sessions(day_sessions, agents, call_tags={"day_idx": day_idx})

                # Store completed sessions on agents for memory context injection
                if self.cfg.memory_context_sessions:
                    for sess in day_sessions:
                        for pid in sess.participants:
                            if pid in agents:
                                agents[pid].store_completed_session(sess)

                # Stream completed day to disk
                for sess in day_sessions:
                    sessions.append(sess)
                    _stream_fh.write(json.dumps(sess.to_dict(), ensure_ascii=False) + "\n")
                _stream_fh.flush()

        finally:
            _stream_fh.close()

        log.info(
            "Legacy simulation produced %d sessions across %d days",
            len(sessions), len(day_events),
        )
        return sessions

    def _process_inline_images(
        self, sessions: List[Session],
    ) -> List[Dict[str, Any]]:
        """Parse [IMAGE:] tags from text_message turns and record a generation queue.

        Tags are PRESERVED in turn.text — the image description remains readable
        by text-only memory systems.  Phase 2 generates the actual image files
        and multimodal evaluators can swap the tag for a real image at eval time.

        Actual image generation is deferred to Phase 2:
            python -m MASim.pipeline.cli generate-images --run <dir>

        Always returns [] — multimodal_index.jsonl is populated in Phase 2.
        """
        from MASim.generation.multimodal.inline_image import parse_image_tags

        queue: List[Dict[str, Any]] = []

        for sess in sessions:
            if sess.modality != "text_message":
                continue
            sess_ts = getattr(sess, "start_time", 0.0)
            for turn in sess.turns:
                tags = parse_image_tags(turn.text)
                if not tags:
                    continue
                # Text is kept unchanged — [IMAGE: description] stays in the corpus
                # so the full description is available without the image file.
                for tag in tags:
                    queue.append({
                        "turn_id": turn.turn_id,
                        "session_id": sess.session_id,
                        "speaker_id": turn.speaker_id,
                        "description": tag["description"],
                        "timestamp": sess_ts,
                    })
                turn.metadata["inline_images"] = [
                    {"description": t["description"], "path": None, "queued": True}
                    for t in tags
                ]

        if queue:
            write_jsonl(self.output_dir / "image_queue.jsonl", queue)
            log.info(
                "Queued %d image(s) → image_queue.jsonl  "
                "(run `generate-images` to generate Phase 2)",
                len(queue),
            )
        return []

    def _process_voice_messages(
        self, sessions: List[Session],
    ) -> List[Dict[str, Any]]:
        """Record TTS generation queue for voice_message turns.

        Writes voice_queue.jsonl so the text corpus is complete and usable
        by text-only memory systems immediately after Phase 1.

        Actual audio generation is deferred to Phase 3:
            python -m MASim.pipeline.cli generate-audio --run <dir>

        Always returns [] — multimodal_index.jsonl is populated in Phase 3.
        """
        queue: List[Dict[str, Any]] = []

        for sess in sessions:
            if sess.modality != "voice_message":
                continue
            sess_ts = getattr(sess, "start_time", 0.0)
            for turn in sess.turns:
                queue.append({
                    "turn_id": turn.turn_id,
                    "session_id": sess.session_id,
                    "speaker_id": turn.speaker_id,
                    "text": turn.text,
                    "timestamp": sess_ts,
                })

        if queue:
            write_jsonl(self.output_dir / "voice_queue.jsonl", queue)
            log.info(
                "Queued %d voice turn(s) → voice_queue.jsonl  "
                "(run `generate-audio` to generate Phase 3)",
                len(queue),
            )
        return []

    def _run_person_agent_sessions(
        self,
        agents: Dict[str, EntityAgent],
        pp_sessions: List[Session],
        llm_client: Any,
        scheduler: Any,
        rng: np.random.Generator,
    ) -> List[Session]:
        """Generate person-agent sessions using step-wise batched generation.

        Three phases:
          A. Activity narration
          B. Memory reflection
          C. Memory probe (uses BM25 over agent facts/feelings)

        Uses PersonAgentEngine.simulate_pa_sessions_stepped() so all PA
        sessions advance one turn at a time with generate_batch(), letting
        the sglang backend batch prefills efficiently (#new-seq >> 1).
        """
        from MASim.core.person_agent_engine import PersonAgentEngine
        from MASim.generation.activity_topic_selector import ActivityTopicSelector
        from MASim.generation.probe_selector import ProbeSelector

        pa_cfg = self.cfg.person_agent
        engine = PersonAgentEngine(
            llm_client,
            max_turns=pa_cfg.max_turns,
            min_turns=pa_cfg.min_turns,
            probe_max_turns=pa_cfg.probe_max_turns,
            probe_min_turns=pa_cfg.probe_min_turns,
            no_token_limit=self.cfg.no_token_limit,
        )
        topic_selector = ActivityTopicSelector(seed=self.cfg.seed + 888)
        probe_selector = ProbeSelector(seed=self.cfg.seed + 999)

        n_pp = len(pp_sessions)
        n_pa = max(1, int(round(n_pp * pa_cfg.ratio)))

        # Compute 3-way split (activity / reflection / probe)
        if pa_cfg.activity_reflection_probe_split:
            split = pa_cfg.activity_reflection_probe_split
            n_activity = max(1, int(round(n_pa * split[0])))
            n_reflection = max(0, int(round(n_pa * split[1])))
            n_probe = n_pa - n_activity - n_reflection
        else:
            # Legacy 2-way split (no probes)
            n_activity = max(1, int(round(n_pa * pa_cfg.activity_reflection_split)))
            n_reflection = n_pa - n_activity
            n_probe = 0

        log.info(
            "Person-agent targets: %d total (%d activity, %d reflection, %d probe) from %d person-person",
            n_pa, n_activity, n_reflection, n_probe, n_pp,
        )

        # Distribute sessions across agents proportional to their person-person count
        agent_pp_count: Dict[str, int] = {}
        for sess in pp_sessions:
            for pid in sess.participants:
                if pid in agents:
                    agent_pp_count[pid] = agent_pp_count.get(pid, 0) + 1
        if not agent_pp_count:
            agent_pp_count = {aid: 1 for aid in agents}

        total_pp = sum(agent_pp_count.values())
        agent_ids_sorted = sorted(agent_pp_count.keys())

        # Compute per-agent allocation
        agent_n_activity: Dict[str, int] = {}
        agent_n_reflection: Dict[str, int] = {}
        agent_n_probe: Dict[str, int] = {}
        act_remaining = n_activity
        ref_remaining = n_reflection
        probe_remaining = n_probe
        for i, aid in enumerate(agent_ids_sorted):
            frac = agent_pp_count[aid] / total_pp
            if i == len(agent_ids_sorted) - 1:
                agent_n_activity[aid] = act_remaining
                agent_n_reflection[aid] = ref_remaining
                agent_n_probe[aid] = probe_remaining
            else:
                a = max(0, int(round(n_activity * frac)))
                r = max(0, int(round(n_reflection * frac)))
                p = max(0, int(round(n_probe * frac)))
                agent_n_activity[aid] = min(a, act_remaining)
                agent_n_reflection[aid] = min(r, ref_remaining)
                agent_n_probe[aid] = min(p, probe_remaining)
                act_remaining -= agent_n_activity[aid]
                ref_remaining -= agent_n_reflection[aid]
                probe_remaining -= agent_n_probe[aid]

        max_ts = max((s.end_time for s in pp_sessions), default=0.0)

        # --- Phase A: Build activity configs ---
        activity_configs: List[Dict] = []
        for aid in agent_ids_sorted:
            n_act = agent_n_activity.get(aid, 0)
            if n_act <= 0:
                continue
            topics = topic_selector.select_topics(agents[aid].persona, n=n_act)
            for topic in topics:
                ts = max_ts + 0.1 + len(activity_configs) * 0.05
                activity_configs.append({
                    "agent": agents[aid],
                    "session_type": "activity",
                    "topic": topic,
                    "timestamp": ts,
                    "session_seed": int(rng.integers(0, 2**31)),
                })

        # Run all activity configs in a single stepped batch
        activity_sessions: List[Session] = []
        if activity_configs:
            log.info("=== Phase A: Activity generation — %d sessions ===", len(activity_configs))
            t0 = time.monotonic()
            activity_sessions = engine.simulate_pa_sessions_stepped(
                activity_configs, dry_run=self.cfg.dry_run,
            )
            gen_dur = time.monotonic() - t0
            log.info("Activity generation done in %.1fs (%.2fs/session)", gen_dur, gen_dur / len(activity_configs))
            if not self.cfg.dry_run:
                log.info("Activity NER/mood finalization for %d sessions...", len(activity_sessions))
                t1 = time.monotonic()
                engine.batch_finalize_pa_sessions(activity_sessions, agents)
                log.info("Activity finalization done in %.1fs", time.monotonic() - t1)

        # --- Phase B: Build reflection configs ---
        agent_session_pool: Dict[str, List[Session]] = {aid: [] for aid in agents}
        for sess in pp_sessions + activity_sessions:
            for pid in sess.participants:
                if pid in agent_session_pool:
                    agent_session_pool[pid].append(sess)

        ref_ts = max_ts + 0.1 + len(activity_configs) * 0.05 + 1.0

        reflection_configs: List[Dict] = []
        for aid in agent_ids_sorted:
            n_ref = agent_n_reflection.get(aid, 0)
            pool = agent_session_pool.get(aid, [])
            if n_ref <= 0 or not pool:
                continue
            n_ref = min(n_ref, len(pool))
            target_indices = rng.choice(len(pool), size=n_ref, replace=False)
            for idx in target_indices:
                reflection_configs.append({
                    "agent": agents[aid],
                    "session_type": "reflection",
                    "target_session": pool[idx],
                    "timestamp": ref_ts,
                    "session_seed": int(rng.integers(0, 2**31)),
                })
                ref_ts += 0.05

        # Run all reflection configs in a single stepped batch
        reflection_sessions: List[Session] = []
        if reflection_configs:
            log.info("=== Phase B: Reflection generation — %d sessions ===", len(reflection_configs))
            t0 = time.monotonic()
            reflection_sessions = engine.simulate_pa_sessions_stepped(
                reflection_configs, dry_run=self.cfg.dry_run,
            )
            gen_dur = time.monotonic() - t0
            log.info("Reflection generation done in %.1fs (%.2fs/session)", gen_dur, gen_dur / len(reflection_configs))
            if not self.cfg.dry_run:
                log.info("Reflection NER/mood finalization for %d sessions...", len(reflection_sessions))
                t1 = time.monotonic()
                engine.batch_finalize_pa_sessions(reflection_sessions, agents)
                log.info("Reflection finalization done in %.1fs", time.monotonic() - t1)

        # --- Phase C: Build memory probe configs ---
        probe_sessions: List[Session] = []
        if n_probe > 0:
            probe_ts = ref_ts + 1.0

            probe_configs: List[Dict] = []
            for aid in agent_ids_sorted:
                n_prb = agent_n_probe.get(aid, 0)
                if n_prb <= 0 or aid not in agents:
                    continue
                agent = agents[aid]
                # Need facts to probe — skip agents with empty knowledge
                if not agent.knowledge.facts:
                    continue
                probes = probe_selector.select_probes(aid, agent.knowledge, n=n_prb)
                for probe_spec in probes:
                    probe_configs.append({
                        "agent": agent,
                        "session_type": "probe",
                        "probe_spec": probe_spec,
                        "timestamp": probe_ts,
                        "session_seed": int(rng.integers(0, 2**31)),
                    })
                    probe_ts += 0.05

            if probe_configs:
                log.info("=== Phase C: Memory probe generation — %d sessions ===", len(probe_configs))
                t0 = time.monotonic()
                probe_sessions = engine.simulate_pa_sessions_stepped(
                    probe_configs, dry_run=self.cfg.dry_run,
                )
                gen_dur = time.monotonic() - t0
                log.info("Probe generation done in %.1fs (%.2fs/session)", gen_dur, gen_dur / len(probe_configs))
                if not self.cfg.dry_run:
                    log.info("Probe NER/mood finalization for %d sessions...", len(probe_sessions))
                    t1 = time.monotonic()
                    engine.batch_finalize_pa_sessions(probe_sessions, agents)
                    log.info("Probe finalization done in %.1fs", time.monotonic() - t1)

        all_pa = activity_sessions + reflection_sessions + probe_sessions
        log.info(
            "Person-agent generated: %d activity + %d reflection + %d probe = %d total",
            len(activity_sessions), len(reflection_sessions), len(probe_sessions), len(all_pa),
        )
        return all_pa

    def _run_multimodal_pipeline(
        self,
        corpus: DialogueCorpus,
        llm_client: Any,
    ) -> None:
        """Run multimodal augmentation: scan, generate, filter, write index."""
        from MASim.generation.multimodal.scanner import MultimodalScanner
        from MASim.generation.multimodal.image_generator import (
            DiffusionImageGenerator, DummyImageGenerator, ImageGenRequest,
        )
        from MASim.generation.multimodal.audio_generator import (
            AudioGenRequest, DummyAudioGenerator, VoiceProfile,
        )
        from MASim.generation.multimodal.quality_filter import (
            DummyQualityFilter, QualityFilterConfig,
        )

        mm_cfg = self.cfg.multimodal
        media_dir = self.output_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Scan for augmentable turns
        scanner = MultimodalScanner(llm_client)
        candidates = scanner.scan(
            corpus,
            max_candidates=mm_cfg.max_image_candidates,
            dry_run=self.cfg.dry_run,
        )
        log.info("Multimodal scanner found %d candidates", len(candidates))

        index_entries: List[Dict[str, Any]] = []

        # Step 2: Generate images
        if mm_cfg.enable_image:
            image_dir = media_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            image_candidates = [c for c in candidates if c.modality == "image"]
            if image_candidates:
                if mm_cfg.image_endpoint or mm_cfg.image_model:
                    try:
                        generator = DiffusionImageGenerator(
                            endpoint=mm_cfg.image_endpoint or None,
                            model=mm_cfg.image_model,
                            output_dir=image_dir,
                        )
                    except Exception:
                        log.warning("Diffusion generator unavailable, falling back to dummy")
                        generator = DummyImageGenerator(output_dir=image_dir)
                else:
                    generator = DummyImageGenerator(output_dir=image_dir)

                requests = [
                    ImageGenRequest(turn_id=c.turn_id, prompt=c.image_prompt)
                    for c in image_candidates
                ]
                results = generator.generate_batch(requests)

                # Quality filter
                qf = DummyQualityFilter()
                qf_cfg = QualityFilterConfig(clip_threshold=mm_cfg.clip_threshold)
                pairs = [
                    (r.turn_id, r.image_path, c.text)
                    for r, c in zip(results, image_candidates)
                    if r.success and r.image_path
                ]
                if pairs:
                    passed, rejected = qf.filter_batch(pairs, qf_cfg)
                    log.info("Image quality filter: %d passed, %d rejected", len(passed), len(rejected))
                else:
                    passed = []

                for r in results:
                    if r.success and r.turn_id in set(passed):
                        index_entries.append({
                            "turn_id": r.turn_id,
                            "modality": "image",
                            "path": str(r.image_path),
                        })

        # Step 3: Generate audio
        if mm_cfg.enable_audio:
            audio_dir = media_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

            audio_candidates = [c for c in candidates if c.modality == "audio"]
            if audio_candidates:
                generator = DummyAudioGenerator(output_dir=audio_dir)
                requests = [
                    AudioGenRequest(
                        turn_id=c.turn_id,
                        text=c.text,
                        voice_profile=VoiceProfile(agent_id=c.turn_id.split("_")[0]),
                    )
                    for c in audio_candidates
                ]
                results = generator.generate_batch(requests)
                for r in results:
                    if r.success:
                        index_entries.append({
                            "turn_id": r.turn_id,
                            "modality": "audio",
                            "path": str(r.audio_path),
                            "duration_seconds": r.duration_seconds,
                        })

        # Write multimodal index
        if index_entries:
            write_jsonl(self.output_dir / "multimodal_index.jsonl", index_entries)
            log.info("Wrote %d multimodal index entries", len(index_entries))

    def _write_static_checkpoint(
        self,
        agents: Dict[str, EntityAgent],
        locations: List,
        groups: List,
        person_roles: List,
        transit_matrix: Dict,
    ) -> None:
        """Write pre-simulation static data for crash recovery.

        Called immediately after Stage 2c so that agents/locations/groups are
        on disk even if the simulation crashes hours later.  _write_outputs()
        will overwrite these files with the authoritative post-simulation data.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            self.output_dir / "agents_personas.jsonl",
            [
                {
                    "agent_id": aid,
                    "persona": a.persona.to_dict(),
                    "social_neighbors": dict(a.social_neighbors),
                    "person_impressions": {},
                }
                for aid, a in agents.items()
            ],
        )
        if locations:
            write_json_report(
                self.output_dir / "locations.json",
                [loc.to_dict() for loc in locations],
            )
        if groups:
            write_json_report(
                self.output_dir / "groups.json",
                [g.to_dict() for g in groups],
            )
        if person_roles:
            write_jsonl(
                self.output_dir / "person_roles.jsonl",
                [r.to_dict() for r in person_roles],
            )
        if transit_matrix:
            write_json_report(self.output_dir / "transit_matrix.json", transit_matrix)

    def _write_outputs(
        self,
        corpus: DialogueCorpus,
        eval_instances: Dict[Dimension, List[EvalInstance]],
        ego_data: Dict[str, Any],
        report: Dict[str, Any],
        locations: Optional[List] = None,
        groups: Optional[List] = None,
        person_roles: Optional[List] = None,
        transit_matrix: Optional[Dict[str, Any]] = None,
        activity_logs: Optional[Dict[str, Any]] = None,
        daily_schedules: Optional[Dict[str, Any]] = None,
        inline_image_entries: Optional[List[Dict[str, Any]]] = None,
        voice_audio_entries: Optional[List[Dict[str, Any]]] = None,
        ego_session_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Write all pipeline outputs to disk."""
        # Corpus sessions
        write_jsonl(
            self.output_dir / "corpus_sessions.jsonl",
            [s.to_dict() for s in corpus.sessions],
        )

        # Events
        write_jsonl(
            self.output_dir / "events.jsonl",
            [e.to_dict() for e in corpus.events],
        )

        # Eval instances per dimension
        eval_dir = self.output_dir / "eval_instances"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for dim, instances in eval_instances.items():
            write_jsonl(
                eval_dir / f"{dim.value}.jsonl",
                [inst.to_dict() for inst in instances],
            )

        # Ego projection summaries
        write_json_report(
            self.output_dir / "ego_projections.json",
            ego_data,
        )

        # Ego session map: agent_id -> [session_id, ...]
        if ego_session_map:
            write_json_report(
                self.output_dir / "ego_session_map.json",
                ego_session_map,
            )

        # Graph edges
        write_jsonl(
            self.output_dir / "graph_edges.jsonl",
            corpus.social_graph_edges,
        )

        # Agent personas + social graph + impressions (for agent_memory generation)
        write_jsonl(
            self.output_dir / "agents_personas.jsonl",
            [
                {
                    "agent_id": aid,
                    "persona": state.persona.to_dict(),
                    "social_neighbors": state.social_neighbors,
                    "person_impressions": state.knowledge_state.get("person_impressions", {}),
                }
                for aid, state in corpus.agents.items()
            ],
        )

        # Locations, groups, roles, transit matrix
        if locations:
            write_json_report(
                self.output_dir / "locations.json",
                [loc.to_dict() for loc in locations],
            )
        if groups:
            write_json_report(
                self.output_dir / "groups.json",
                [g.to_dict() for g in groups],
            )
        if person_roles:
            write_jsonl(
                self.output_dir / "person_roles.jsonl",
                [r.to_dict() for r in person_roles],
            )
        if transit_matrix:
            write_json_report(
                self.output_dir / "transit_matrix.json",
                transit_matrix,
            )

        # Activity logs (all entries chronological, one per line)
        if activity_logs:
            all_entries = sorted(
                [e for entries in activity_logs.values() for e in entries],
                key=lambda e: (e.agent_id, e.start_time),
            )
            write_jsonl(
                self.output_dir / "activity_log.jsonl",
                [e.to_dict() for e in all_entries],
            )

        # Per-agent daily schedules
        if daily_schedules:
            write_jsonl(
                self.output_dir / "agent_schedules.jsonl",
                [sched.to_dict() for sched in daily_schedules.values()],
            )

        # Merge inline image + voice audio entries into multimodal_index.jsonl
        mm_entries = []
        if inline_image_entries:
            mm_entries.extend(inline_image_entries)
        if voice_audio_entries:
            mm_entries.extend(voice_audio_entries)
        if mm_entries:
            write_jsonl(self.output_dir / "multimodal_index.jsonl", mm_entries)
            log.info("Wrote %d multimodal index entries (inline images + voice)", len(mm_entries))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _partition_conversations(
    pairs: List[tuple],
) -> List[List[tuple]]:
    """Partition conversation pairs into independent sub-batches.

    Two pairs conflict if they share a participant. Uses greedy graph coloring
    (largest-degree-first) to group non-conflicting pairs together.

    Returns a list of sub-batches. When all pairs are independent, returns a
    single sub-batch (equivalent to the previous behavior).
    """
    n = len(pairs)
    if n <= 1:
        return [pairs]

    # Build adjacency: pairs that share a participant
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        pi = set(pairs[i])
        for j in range(i + 1, n):
            if pi & set(pairs[j]):
                adj[i].append(j)
                adj[j].append(i)

    # Greedy coloring in largest-degree-first order
    order = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
    colors = [-1] * n
    for node in order:
        neighbor_colors = {colors[nb] for nb in adj[node] if colors[nb] >= 0}
        c = 0
        while c in neighbor_colors:
            c += 1
        colors[node] = c

    # Group pairs by color
    max_color = max(colors) + 1
    batches: List[List[tuple]] = [[] for _ in range(max_color)]
    for i, pair in enumerate(pairs):
        batches[colors[i]].append(pair)

    return [b for b in batches if b]


def _assign_event_locations(
    events: List[WorldEvent],
    locations: List[Location],
) -> None:
    """Assign a plausible location_id to each event in-place.

    Rules:
      global/community events → public third place or outdoor location
      dyadic events           → any semi-public or private location
      private events          → the target agent's home (if available)
    """
    from MASim.core.schema import EventCategory

    public_locs = [
        l for l in locations
        if l.location_type in ("third_place_regular", "third_place_event", "outdoor")
    ]
    semi_locs = [
        l for l in locations
        if l.location_type in ("workplace", "service")
    ]
    home_locs = {l.owner_agent_id: l for l in locations if l.location_type == "home_private"}

    fallback = locations[0].location_id if locations else ""

    for event in events:
        if event.location_id:
            continue  # already assigned

        if event.category in (EventCategory.GLOBAL, EventCategory.COMMUNITY):
            pool = public_locs or semi_locs or locations
        elif event.category == EventCategory.DYADIC:
            pool = semi_locs or public_locs or locations
        else:  # PRIVATE
            target = event.metadata.get("target_agent", "")
            if target and target in home_locs:
                event.location_id = home_locs[target].location_id
                continue
            pool = [l for l in locations if l.location_type == "home_private"] or locations

        if pool:
            # Deterministic selection based on event_id hash
            idx = int(event.event_id.split("_")[-1]) % len(pool) if event.event_id else 0
            event.location_id = pool[idx % len(pool)].location_id
        else:
            event.location_id = fallback
