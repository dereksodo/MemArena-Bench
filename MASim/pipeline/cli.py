"""CLI entry point for MASim pipeline.

Usage:
    masim run --config MASim/configs/memarena_l.yaml --output runs/run_001/
    masim evaluate --run runs/run_001/ --system zep --output results/
    masim validate --run runs/run_001/
    masim stats --run runs/run_001/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memarena.runtime import configure_live_output
from MASim.pipeline.orchestrator import Orchestrator, PipelineConfig
from MASim.utils.io import read_jsonl, read_yaml, write_json_report
from MASim.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full simulation pipeline."""
    config_path = Path(args.config)
    output_dir = Path(args.output)

    if config_path.exists():
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        log.warning("Config not found: %s, using defaults", config_path)
        cfg = PipelineConfig()

    if args.dry_run:
        cfg.dry_run = True
    if getattr(args, "faster_batch", False):
        cfg.faster_batch = True

    stop_after = getattr(args, "stop_after", None)

    orchestrator = Orchestrator(cfg, output_dir)
    report = orchestrator.run(stop_after=stop_after)

    print(json.dumps(report, indent=2, default=str))


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a memory system against generated instances."""
    run_dir = Path(args.run)
    output_dir = Path(args.output) if args.output else run_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load eval instances
    eval_dir = run_dir / "eval_instances"
    if not eval_dir.exists():
        log.error("No eval instances found in %s", eval_dir)
        sys.exit(1)

    all_instances = {}
    for f in eval_dir.glob("*.jsonl"):
        dim_name = f.stem
        instances = read_jsonl(f)
        all_instances[dim_name] = instances
        log.info("Loaded %d instances for %s", len(instances), dim_name)

    # TODO: Connect to target memory system and run evaluation
    log.info("Evaluation framework ready. %d dimensions loaded.", len(all_instances))
    log.info("Target system: %s", args.system)
    log.info("Output directory: %s", output_dir)

    report = {
        "run_dir": str(run_dir),
        "system": args.system,
        "dimensions": {k: len(v) for k, v in all_instances.items()},
        "status": "ready",
    }
    write_json_report(output_dir / "eval_report.json", report)
    print(json.dumps(report, indent=2))


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume a pipeline run from a specific stage."""
    run_dir = Path(args.run)
    config_path = Path(args.config) if args.config else None

    if config_path and config_path.exists():
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        # Try to infer config from run dir name
        run_name = run_dir.name.lower()
        config_map = {
            "memarena_l": "MASim/configs/memarena_l.yaml",
            "l": "MASim/configs/memarena_l.yaml",
        }
        inferred = None
        for key, path in config_map.items():
            if run_name.startswith(key):
                inferred = path
                break
        if inferred and Path(inferred).exists():
            log.info("Inferred config: %s", inferred)
            cfg = PipelineConfig.from_yaml(Path(inferred))
        else:
            log.warning("No config specified and cannot infer from run dir name. Using defaults.")
            cfg = PipelineConfig()

    stage = args.stage
    valid_stages = ("5d", "post_sim", "regen_eval")
    if stage not in valid_stages:
        log.error("Unsupported resume stage: %s. Valid stages: %s", stage, ", ".join(valid_stages))
        sys.exit(1)

    orchestrator = Orchestrator(cfg, run_dir)
    report = orchestrator.resume_from(stage)
    print(json.dumps(report, indent=2, default=str))


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate consistency of a pipeline run."""
    run_dir = Path(args.run)
    issues = []

    # Check required files
    required = ["corpus_sessions.jsonl", "events.jsonl", "pipeline_report.json"]
    for fname in required:
        if not (run_dir / fname).exists():
            issues.append(f"Missing required file: {fname}")

    # Check eval instances
    eval_dir = run_dir / "eval_instances"
    if eval_dir.exists():
        for f in eval_dir.glob("*.jsonl"):
            rows = read_jsonl(f)
            for i, row in enumerate(rows):
                if not row.get("instance_id"):
                    issues.append(f"{f.name}[{i}]: missing instance_id")
                if not row.get("query"):
                    issues.append(f"{f.name}[{i}]: missing query")
    else:
        issues.append("Missing eval_instances/ directory")

    # Check corpus sessions
    corpus_path = run_dir / "corpus_sessions.jsonl"
    if corpus_path.exists():
        sessions = read_jsonl(corpus_path)
        for i, s in enumerate(sessions):
            if not s.get("turns"):
                issues.append(f"Session {i}: no turns")

    if issues:
        print(f"Validation found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("Validation passed. All checks OK.")


def cmd_stats(args: argparse.Namespace) -> None:
    """Print dataset statistics for a pipeline run."""
    run_dir = Path(args.run)

    report_path = run_dir / "pipeline_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        print("=== Pipeline Report ===")
        print(json.dumps(report, indent=2, default=str))
    else:
        print("No pipeline report found.")

    # Session stats
    corpus_path = run_dir / "corpus_sessions.jsonl"
    if corpus_path.exists():
        from MASim.utils.tokens import count_tokens

        sessions = read_jsonl(corpus_path)
        n_turns = 0
        total_tokens = 0
        total_chars = 0
        for s in sessions:
            for t in s.get("turns", []):
                n_turns += 1
                text = t.get("text", "")
                total_tokens += count_tokens(text)
                total_chars += len(text)
        print(f"\n=== Corpus Stats ===")
        print(f"Sessions: {len(sessions)}")
        print(f"Total turns: {n_turns}")
        if sessions:
            avg_turns = n_turns / len(sessions)
            print(f"Avg turns/session: {avg_turns:.1f}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total characters: {total_chars:,}")
        if n_turns:
            print(f"Avg tokens/turn: {total_tokens / n_turns:.1f}")

    # Eval instance stats
    eval_dir = run_dir / "eval_instances"
    if eval_dir.exists():
        print(f"\n=== Eval Instances ===")
        total = 0
        for f in sorted(eval_dir.glob("*.jsonl")):
            rows = read_jsonl(f)
            print(f"  {f.stem}: {len(rows)}")
            total += len(rows)
        print(f"  TOTAL: {total}")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Generate a readable chat transcript for inspection."""
    run_dir = Path(args.run)
    out_path = run_dir / "combined_inspect.txt"

    sessions = read_jsonl(run_dir / "corpus_sessions.jsonl")
    events_list = read_jsonl(run_dir / "events.jsonl") if (run_dir / "events.jsonl").exists() else []
    events = {e["event_id"]: e for e in events_list}

    # Build session→questions map from eval instances
    # For cross-session questions, attach to each associated session
    from collections import defaultdict
    session_questions: dict[str, list[dict]] = defaultdict(list)
    eval_dir = run_dir / "eval_instances"
    all_instances: list[dict] = []
    if eval_dir.exists():
        for f in sorted(eval_dir.glob("*.jsonl")):
            all_instances.extend(read_jsonl(f))

    for inst in all_instances:
        # Get associated sessions: prefer metadata.evidence_sessions, fall back to ego_context
        meta = inst.get("metadata", {})
        assoc_sessions = set(meta.get("evidence_sessions", []))
        if not assoc_sessions:
            assoc_sessions = {t["session_id"] for t in inst.get("ego_context", []) if "session_id" in t}
        for sid in assoc_sessions:
            session_questions[sid].append(inst)

    # Track which cross-session questions we've already shown (to avoid full duplication)
    shown_questions: set[str] = set()

    lines: list[str] = []
    for sess in sessions:
        sid = sess["session_id"]
        evt_content = ""
        for eid in sess.get("triggering_events", []):
            if eid in events:
                evt_content = f"[{events[eid].get('category', '?')}] {events[eid].get('content', '')}"

        lines.append(f"=== {sid} ({', '.join(sess['participants'])}) ===")
        if evt_content:
            lines.append(f"  Event: {evt_content}")
        lines.append("")

        for t in sess["turns"]:
            lines.append(f"{t['speaker_id']}: {t['text']}")

        # Append associated questions
        questions = session_questions.get(sid, [])
        if questions:
            lines.append("")
            lines.append(f"  --- Questions ({len(questions)}) ---")
            for q in questions:
                qid = q["instance_id"]
                dim = q["dimension"]
                # For cross-session questions, mark if already shown after a prior session
                cross_note = ""
                if qid in shown_questions:
                    cross_note = " [cross-ref, also shown above]"
                shown_questions.add(qid)

                meta = q.get("metadata", {})
                asked_to = meta.get("query_agent", "?")
                feature = meta.get("question_feature", "")
                feature_tag = f" ({feature})" if feature else ""
                qt = meta.get("query_timestamp")
                ts_tag = f" t={qt:.3f}" if qt is not None else ""
                lines.append(f"  [{dim}]{feature_tag}{ts_tag}{cross_note} {qid}")
                lines.append(f"    Asked to: {asked_to}")
                lines.append(f"    Q: {q['query']}")
                gt = q.get("ground_truth", {})
                # Format ground truth compactly
                if isinstance(gt, dict):
                    gt_str = "; ".join(f"{k}: {v}" for k, v in gt.items())
                else:
                    gt_str = str(gt)
                lines.append(f"    A: {gt_str}")

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to {out_path}")
    print(f"Dialogues: {len(sessions)} sessions")
    print(f"Questions: {len(all_instances)} total, {len(shown_questions)} unique attached to sessions")


_DUNBAR_LABELS = {
    0: "ego",
    1: "intimate (5)",
    2: "close friend (15)",
    3: "active contact (50)",
    4: "acquaintance (150)",
}


def _slug_to_name(slug: str) -> str:
    """Convert agent ID slug to readable display name: maya_chen -> Maya Chen."""
    return " ".join(w.capitalize() for w in slug.split("_"))


def cmd_agent_memory(args: argparse.Namespace) -> None:
    """Write a human-readable memory diary for every agent in a run."""
    run_dir = Path(args.run)
    out_path = run_dir / "agent_memory.txt"

    agents_path = run_dir / "agents_personas.jsonl"
    if not agents_path.exists():
        log.error(
            "agents_personas.jsonl not found in %s. "
            "Re-run the pipeline to generate it (requires updated orchestrator).",
            run_dir,
        )
        sys.exit(1)

    agents = {a["agent_id"]: a for a in read_jsonl(agents_path)}
    sessions = read_jsonl(run_dir / "corpus_sessions.jsonl")
    events_list = (
        read_jsonl(run_dir / "events.jsonl")
        if (run_dir / "events.jsonl").exists()
        else []
    )
    events = {e["event_id"]: e for e in events_list}

    # Group sessions by participant
    from collections import defaultdict
    agent_sessions: dict[str, list] = defaultdict(list)
    for sess in sessions:
        for pid in sess.get("participants", []):
            agent_sessions[pid].append(sess)

    WIDE = 66
    SEP = "=" * WIDE
    THIN = "─" * WIDE

    lines: list[str] = []

    for agent_id in sorted(agents.keys()):
        info = agents[agent_id]
        persona = info["persona"]
        neighbors: dict[str, float] = info.get("social_neighbors", {})
        my_sessions = agent_sessions.get(agent_id, [])

        # ── Agent header ──────────────────────────────────────────────────
        lines.append(SEP)
        lines.append(f"  AGENT: {persona['name']}")
        lines.append(SEP)

        demo = persona.get("demographics", {})
        layer = persona.get("dunbar_layer", 0)
        layer_label = _DUNBAR_LABELS.get(layer, f"layer {layer}")

        lines.append(f"  Occupation : {persona.get('occupation', '?')}")
        lines.append(
            f"  Age        : {persona.get('age', '?')}  |  "
            f"Gender: {demo.get('gender', '?')}  |  "
            f"Dunbar: {layer_label}"
        )

        # Demographics line 1: nationality, race, language
        demo_parts = []
        if demo.get("nationality"):
            demo_parts.append(f"Nationality: {demo['nationality']}")
        if demo.get("race"):
            demo_parts.append(f"Race: {demo['race']}")
        if demo.get("mother_language"):
            demo_parts.append(f"Language: {demo['mother_language']}")
        if demo_parts:
            lines.append(f"  Demo       : {', '.join(demo_parts)}")

        # Demographics line 2: marital status, income, education
        life_parts = []
        if demo.get("marital_status"):
            life_parts.append(f"Marital: {demo['marital_status']}")
        if demo.get("income"):
            life_parts.append(f"Income: {demo['income']}")
        if persona.get("education_level"):
            life_parts.append(f"Education: {persona['education_level']}")
        if life_parts:
            lines.append(f"  Life       : {', '.join(life_parts)}")

        # Personality traits (16PF)
        traits = persona.get("personality_traits") or []
        if traits:
            lines.append(f"  Personality: {', '.join(traits)}")

        # Fun details: zodiac, idol, motto, social platform
        fun_parts = []
        if demo.get("zodiac"):
            fun_parts.append(f"Zodiac: {demo['zodiac']}")
        if demo.get("idol"):
            fun_parts.append(f"Idol: {demo['idol']}")
        if demo.get("social_platform"):
            fun_parts.append(f"Platform: {demo['social_platform']}")
        if fun_parts:
            lines.append(f"  Fun        : {', '.join(fun_parts)}")
        if demo.get("motto"):
            lines.append(f"  Motto      : \"{demo['motto']}\"")

        # Background (one-sentence)
        if persona.get("backstory"):
            lines.append("")
            lines.append("  Background :")
            for chunk in _wrap(persona["backstory"], 60):
                lines.append(f"    {chunk}")

        # Speaking style (one-sentence)
        if persona.get("speaking_style"):
            lines.append("")
            lines.append("  Speaking style :")
            for chunk in _wrap(persona["speaking_style"], 60):
                lines.append(f"    {chunk}")

        # Hobbies, values, concerns
        if persona.get("hobbies"):
            lines.append(f"\n  Hobbies    : {', '.join(persona['hobbies'])}")
        if persona.get("values"):
            lines.append(f"  Values     : {', '.join(persona['values'])}")
        if persona.get("current_concerns"):
            lines.append(f"  Concerns   : {'; '.join(persona['current_concerns'])}")

        # Favourites
        fav_parts = []
        if demo.get("favourite_movie"):
            fav_parts.append(f"Movie: {demo['favourite_movie']}")
        if demo.get("favourite_sport"):
            fav_parts.append(f"Sport: {demo['favourite_sport']}")
        if fav_parts:
            lines.append(f"  Favourites : {', '.join(fav_parts)}")

        # Relationships / key people
        if persona.get("relationships"):
            rels = "; ".join(
                f"{k} ({v})" for k, v in list(persona["relationships"].items())[:5]
            )
            lines.append(f"  People     : {rels}")

        # Schedule
        sleep_start = persona.get("sleep_start_hour")
        sleep_end = persona.get("sleep_end_hour")
        work_sched = persona.get("work_schedule", "")
        if sleep_start is not None and sleep_end is not None:
            lines.append(
                f"  Schedule   : Sleep {sleep_start:.0f}:00–{sleep_end:.0f}:00"
                f"  |  Work: {work_sched}"
            )

        # Social neighbors (top 8 by weight)
        if neighbors:
            top = sorted(neighbors.items(), key=lambda x: -x[1])[:8]
            neighbor_str = ", ".join(f"{_slug_to_name(nid)} ({w:.2f})" for nid, w in top)
            lines.append(f"\n  Social     : {neighbor_str}")

        # Impressions of other people (built up across sessions)
        person_impressions: dict = info.get("person_impressions", {})
        if person_impressions:
            lines.append("")
            lines.append("  Impressions of others :")
            for other_id, observations in person_impressions.items():
                other_name = _slug_to_name(other_id)
                lines.append(f"    {other_name}:")
                for obs in observations:
                    lines.append(f"      - {obs}")

        # ── Memory section ────────────────────────────────────────────────
        n_sessions = len(my_sessions)
        n_turns_heard = sum(
            1 for s in my_sessions
            for t in s.get("turns", [])
            if t["speaker_id"] != agent_id
        )
        lines.append("")
        lines.append(f"  {THIN}")
        lines.append(
            f"  MEMORY  —  {n_sessions} session(s) participated, "
            f"{n_turns_heard} turn(s) heard"
        )
        lines.append(f"  {THIN}")

        if not my_sessions:
            lines.append("  (no sessions recorded)")
        else:
            for i, sess in enumerate(my_sessions, 1):
                sid = sess["session_id"]
                participants = sess.get("participants", [])
                other_names = [_slug_to_name(p) for p in participants if p != agent_id]
                is_group = (
                    sess.get("metadata", {}).get("group_conversation", False)
                    or len(participants) > 2
                )

                if is_group:
                    partner_str = (
                        f"{persona['name']}, " + ", ".join(other_names) + "  [group]"
                    )
                else:
                    partner_str = (
                        f"{persona['name']} ↔ {other_names[0] if other_names else '?'}"
                    )

                lines.append("")
                lines.append(f"  [{i}/{n_sessions}]  {sid}")
                lines.append(f"  Participants: {partner_str}")

                # Location if available
                loc_id = sess.get("location_id", "")
                if loc_id:
                    loc_name = loc_id.replace("loc_", "").replace("_", " ").title()
                    lines.append(f"  Location: {loc_name}")

                for eid in sess.get("triggering_events", []):
                    if eid in events:
                        ev = events[eid]
                        cat = ev.get("category", "?").upper()
                        lines.append(f"  Event: [{cat}] {ev.get('content', '')}")

                lines.append("")

                name_map = {p: _slug_to_name(p) for p in participants}
                max_name_len = max((len(n) for n in name_map.values()), default=10)

                for turn in sess.get("turns", []):
                    speaker = turn["speaker_id"]
                    speaker_name = name_map.get(speaker, _slug_to_name(speaker))
                    marker = "→" if speaker == agent_id else " "
                    padded = speaker_name.ljust(max_name_len)
                    lines.append(f"  {marker} {padded}  {turn['text']}")

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to {out_path}")
    print(f"Agents: {len(agents)}  |  Sessions: {len(sessions)}")


def cmd_generate_images(args: argparse.Namespace) -> None:
    """Phase 2: Generate images from image_queue.jsonl using FLUX.2."""
    run_dir = Path(args.run)
    queue_path = run_dir / "image_queue.jsonl"
    if not queue_path.exists():
        log.error("image_queue.jsonl not found in %s. Run Phase 1 first.", run_dir)
        sys.exit(1)

    from MASim.generation.multimodal.inline_image import FluxImageClient

    queue = read_jsonl(queue_path)
    if not queue:
        print("No images queued.")
        return

    # Skip already-generated entries (idempotent re-runs)
    index_path = run_dir / "multimodal_index.jsonl"
    done_ids: set[str] = set()
    if index_path.exists():
        for entry in read_jsonl(index_path):
            if entry.get("modality") == "inline_image" and entry.get("path"):
                done_ids.add(entry["turn_id"])

    remaining = [e for e in queue if e["turn_id"] not in done_ids]
    print(
        f"Image queue: {len(queue)} total, {len(done_ids)} already done, "
        f"{len(remaining)} to generate"
    )
    if not remaining:
        print("All images already generated.")
        return

    endpoint = args.endpoint or "http://localhost:8100"
    if not FluxImageClient.check_health(endpoint):
        log.error("FLUX.2 endpoint %s is not reachable.", endpoint)
        sys.exit(1)

    media_dir = run_dir / "media" / "inline_images"
    client = FluxImageClient(endpoint=endpoint, output_dir=media_dir)

    new_entries: list[dict] = []
    for entry in remaining:
        result = client.generate(
            turn_id=entry["turn_id"],
            prompt=entry["description"],
            seed=args.seed,
        )
        if result.success:
            new_entries.append({
                "turn_id": entry["turn_id"],
                "modality": "inline_image",
                "path": result.image_path,
                "description": entry["description"],
            })
            print(f"  OK  {entry['turn_id']} → {result.image_path}")
        else:
            print(f"  FAIL {entry['turn_id']}: {result.error}")

    if new_entries:
        with index_path.open("a", encoding="utf-8") as f:
            for e in new_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Generated {len(new_entries)} image(s). Index: {index_path}")


def cmd_generate_audio(args: argparse.Namespace) -> None:
    """Phase 3: Generate audio from voice_queue.jsonl using Qwen3-TTS."""
    run_dir = Path(args.run)
    queue_path = run_dir / "voice_queue.jsonl"
    if not queue_path.exists():
        log.error("voice_queue.jsonl not found in %s. Run Phase 1 first.", run_dir)
        sys.exit(1)

    from MASim.generation.multimodal.tts_client import Qwen3TTSClient

    queue = read_jsonl(queue_path)
    if not queue:
        print("No voice turns queued.")
        return

    # Skip already-generated entries (idempotent re-runs)
    index_path = run_dir / "multimodal_index.jsonl"
    done_ids: set[str] = set()
    if index_path.exists():
        for entry in read_jsonl(index_path):
            if entry.get("modality") == "voice_audio" and entry.get("path"):
                done_ids.add(entry["turn_id"])

    remaining = [e for e in queue if e["turn_id"] not in done_ids]
    print(
        f"Voice queue: {len(queue)} total, {len(done_ids)} already done, "
        f"{len(remaining)} to generate"
    )
    if not remaining:
        print("All audio already generated.")
        return

    endpoint = args.endpoint or "http://localhost:8200"
    if not Qwen3TTSClient.check_health(endpoint):
        log.error("TTS endpoint %s is not reachable.", endpoint)
        sys.exit(1)

    media_dir = run_dir / "media" / "voice_audio"
    client = Qwen3TTSClient(endpoint=endpoint, output_dir=media_dir)

    new_entries: list[dict] = []
    for entry in remaining:
        result = client.generate(
            turn_id=entry["turn_id"],
            text=entry["text"],
            speaker=entry["speaker_id"],
        )
        if result.success:
            new_entries.append({
                "turn_id": entry["turn_id"],
                "modality": "voice_audio",
                "path": result.audio_path,
            })
            print(f"  OK  {entry['turn_id']} → {result.audio_path}")
        else:
            print(f"  FAIL {entry['turn_id']}")

    if new_entries:
        with index_path.open("a", encoding="utf-8") as f:
            for e in new_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Generated {len(new_entries)} audio file(s). Index: {index_path}")


def _wrap(text: str, width: int) -> list[str]:
    """Simple word-wrap that respects existing newlines."""
    import textwrap
    result = []
    for paragraph in text.splitlines():
        result.extend(textwrap.wrap(paragraph, width) or [""])
    return result


def main() -> None:
    """Main CLI entry point."""
    configure_live_output()
    parser = argparse.ArgumentParser(
        prog="masim",
        description="MASim: Multi-Agent Simulation for Ego-Centric Memory Evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    p_run = subparsers.add_parser("run", help="Run the full simulation pipeline")
    p_run.add_argument("--config", "-c", default="MASim/configs/memarena_l.yaml", help="Path to config YAML")
    p_run.add_argument("--output", "-o", required=True, help="Output directory")
    p_run.add_argument("--dry-run", action="store_true", help="Skip LLM calls, use placeholders")
    p_run.add_argument("--faster-batch", action="store_true", help="Use faster batched generation pipeline")
    p_run.add_argument("--stop-after", type=int, default=None, metavar="STAGE",
                       help="Stop after this stage number (e.g. 2 = after personas/locations/groups, 4 = after events)")

    # resume
    p_resume = subparsers.add_parser("resume", help="Resume a pipeline run from a specific stage")
    p_resume.add_argument("--run", "-r", required=True, help="Path to existing pipeline run directory")
    p_resume.add_argument("--stage", "-s", required=True, help="Stage to resume from (5d, post_sim)")
    p_resume.add_argument("--config", "-c", default=None, help="Config YAML (auto-inferred from run dir name if omitted)")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a memory system")
    p_eval.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")
    p_eval.add_argument("--system", "-s", default="baseline", help="Name of the memory system to evaluate")
    p_eval.add_argument("--output", "-o", help="Output directory for results")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate a pipeline run")
    p_val.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")

    # stats
    p_stats = subparsers.add_parser("stats", help="Print dataset statistics")
    p_stats.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Generate combined JSON for visual inspection")
    p_inspect.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")

    # agent-memory
    p_mem = subparsers.add_parser("agent-memory", help="Write human-readable memory diary for all agents")
    p_mem.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")

    # generate-images  (Phase 2)
    p_img = subparsers.add_parser(
        "generate-images",
        help="Phase 2: Generate images from image_queue.jsonl via FLUX.2",
    )
    p_img.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")
    p_img.add_argument("--endpoint", default=None, help="FLUX.2 endpoint (default: http://localhost:8100)")
    p_img.add_argument("--seed", type=int, default=42, help="Generation seed (default: 42)")

    # generate-audio  (Phase 3)
    p_aud = subparsers.add_parser(
        "generate-audio",
        help="Phase 3: Generate audio from voice_queue.jsonl via Qwen3-TTS",
    )
    p_aud.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")
    p_aud.add_argument("--endpoint", default=None, help="TTS endpoint (default: http://localhost:8200)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    setup_logging()

    commands = {
        "run": cmd_run,
        "resume": cmd_resume,
        "evaluate": cmd_evaluate,
        "validate": cmd_validate,
        "stats": cmd_stats,
        "inspect": cmd_inspect,
        "agent-memory": cmd_agent_memory,
        "generate-images": cmd_generate_images,
        "generate-audio": cmd_generate_audio,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
