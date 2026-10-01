"""Fixes from the fresh-clone MASim smoke test on H200 (2026-10-01)."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from MASim.core.schema import EventCategory, PersonaCard, WorldEvent
from MASim.core.scheduler import DailyScheduler
from MASim.generation.location_factory import _persona_fingerprint
from MASim.ground_truth import d4_permission as d4
from MASim.pipeline import cli as masim_cli
from scripts import run_masim

REPO = Path(__file__).resolve().parents[1]


def _tiny_world():
    def turn(speaker, text, **meta):
        return {"speaker_id": speaker, "text": text, "metadata": meta}

    sessions = [
        {"session_id": "s1", "participants": ["ann", "bob"], "start_time": 0.2, "end_time": 0.21,
         "turns": [turn("ann", "I have been getting into birdwatching lately.",
                        injected_permission=True, permission_level="public"),
                   turn("bob", "Nice!")]},
        {"session_id": "s2", "participants": ["cat", "dan"], "start_time": 0.4, "end_time": 0.41,
         "turns": [turn("cat", "Keep this between us: my door code is 4471.",
                        injected_permission=True, permission_level="private"),
                   turn("dan", "Of course.")]},
        {"session_id": "s3", "participants": ["eve", "fay"], "start_time": 0.6, "end_time": 0.61,
         "turns": [turn("eve", "Lovely weather today."), turn("fay", "It is.")]},
    ]
    edges = [{"source": "ann", "target": "bob", "weight": 1.0},
             {"source": "cat", "target": "dan", "weight": 0.33},
             {"source": "eve", "target": "fay", "weight": 0.109}]
    return sessions, edges


def test_d6_probe_modes() -> None:
    sessions, edges = _tiny_world()
    out = {mode: d4.build_instances(sessions, edges, target_total=40, probes=mode) for mode in d4.PROBE_MODES}
    is_probe = lambda inst: d4.family(inst) in d4.PROBE_FAMILIES  # noqa: E731
    included, _ = out[d4.PROBES_INCLUDE]
    dropped, drop_stats = out[d4.PROBES_DROP]
    excluded, exc_stats = out[d4.PROBES_EXCLUDE]
    assert any(map(is_probe, included))
    assert not any(map(is_probe, dropped)) and not any(map(is_probe, excluded))
    assert drop_stats["probes_dropped"] > 0 and exc_stats["probes_dropped"] == 0
    # drop_after_sampling keeps exactly the fact items of the full draw (same RNG stream, same ids)
    assert [i.instance_id for i in dropped] == [i.instance_id for i in included if not is_probe(i)]
    # without a mode the legacy behaviour is unchanged
    legacy, _ = d4.build_instances(sessions, edges, target_total=40)
    assert [i.to_dict() for i in legacy] == [i.to_dict() for i in dropped]
    with pytest.raises(ValueError):
        d4.build_instances(sessions, edges, probes="sometimes")


def test_location_cache_key_follows_the_personas() -> None:
    nurse = [PersonaCard(name="Ann", occupation="nurse")]
    assert _persona_fingerprint(nurse, ["a0"]) == _persona_fingerprint([PersonaCard(name="Ann", occupation="nurse")], ["a0"])
    assert _persona_fingerprint(nurse, ["a0"]) != _persona_fingerprint([PersonaCard(name="Ann", occupation="teacher")], ["a0"])


def test_event_conversations_end_inside_the_events_day() -> None:
    agents = {a: object() for a in "abcde"}
    sched = DailyScheduler([], agents, [], np.random.default_rng(0), {"face_to_face": 1.0})
    late = WorldEvent(event_id="late", timestamp=2.97, visibility_mask=set("ab"), category=EventCategory.DYADIC)
    crowd = WorldEvent(event_id="crowd", timestamp=2.90, visibility_mask=set("abcde"), category=EventCategory.COMMUNITY)
    specs, _ = sched._phase_b_event_encounters([], [late, crowd], day_length=1.0)
    assert specs and all(s.end_time <= 3.0 + 1e-9 for s in specs)
    assert not any(s.event.event_id == "late" for s in specs)


def _run_dir(tmp_path: Path, *, end_time: float, occupation: str) -> Path:
    run = tmp_path / "run"
    (run / "eval_instances").mkdir(parents=True)
    (run / "corpus_sessions.jsonl").write_text(json.dumps(
        {"session_id": "s1", "start_time": 2.9, "end_time": end_time, "turns": [{"text": "hi"}]}) + "\n")
    (run / "events.jsonl").write_text("")
    (run / "pipeline_report.json").write_text("{}")
    (run / "eval_instances" / "d7_qa.jsonl").write_text(json.dumps({"instance_id": "q1", "query": "?"}) + "\n")
    (run / "effective_config.yaml").write_text(yaml.safe_dump({"time_range": [0.0, 3.0]}))
    (run / "agents_personas.jsonl").write_text(json.dumps(
        {"agent_id": "a0", "persona": {"name": "Ann", "occupation": occupation, "backstory": "x" if occupation else ""}}) + "\n")
    return run


def test_validate_flags_out_of_range_sessions_and_placeholder_personas(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        masim_cli.cmd_validate(argparse.Namespace(run=str(_run_dir(tmp_path, end_time=3.02, occupation=""))))
    out = capsys.readouterr().out
    assert "outside time_range" in out and "missing occupation or backstory" in out
    ok = _run_dir(tmp_path / "ok", end_time=2.95, occupation="nurse")
    masim_cli.cmd_validate(argparse.Namespace(run=str(ok)))
    assert "Validation passed" in capsys.readouterr().out


def test_run_masim_days_copied_config_and_agents_guard(tmp_path: Path, monkeypatch) -> None:
    copied = tmp_path / "small.yaml"
    shutil.copy(REPO / "MASim" / "configs" / "memarena_5a10d_5k.yaml", copied)  # its base stays in MASim/configs/
    monkeypatch.setattr(run_masim, "_run_and_tee", lambda *a, **k: 7)  # stop right after the config is written
    out = tmp_path / "run"
    assert run_masim.main(["--config", str(copied), "--days", "3", "--dry-run", "--output", str(out)]) == 7
    effective = yaml.safe_load((out / "effective_config.yaml").read_text())
    assert effective["time_range"] == [0.0, 3.0] and effective["d6_probes"] == "exclude"
    assert effective["max_instances_per_dim"] == 40
    with pytest.raises(SystemExit, match="--agents applies to --smoke only"):
        run_masim.main(["--config", str(copied), "--agents", "3", "--dry-run", "--output", str(tmp_path / "x")])
    # a bad config fails before --overwrite clears the existing run
    bad = tmp_path / "bad.yaml"
    bad.write_text("base: nope.yaml\n")
    with pytest.raises(SystemExit, match="Config base file not found"):
        run_masim.main(["--config", str(bad), "--dry-run", "--output", str(out), "--overwrite"])
    assert (out / "effective_config.yaml").exists()


def test_sglang_launcher_gpu_and_port_overrides() -> None:
    script = (REPO / "start_sglang_servers.sh").read_text()
    snippet = script[script.index("MODEL_SPECS=("):script.index("\ndone\n", script.index("MODEL_SPECS=(")) + 6]
    env = {"SGLANG_GPUS_8b": "6", "SGLANG_PORT_8b": "31606", "PATH": "/usr/bin:/bin"}
    specs = subprocess.run(["bash", "-c", f'set -euo pipefail\n{snippet}\nprintf "%s\\n" "${{MODEL_SPECS[@]}}"'],
                           env=env, capture_output=True, text=True, check=True).stdout.split()
    assert "8b|Qwen/Qwen3-8B|/models/8b|31606|6" in specs
    assert "llama3b|meta-llama/Llama-3.2-3B-Instruct|/models/llama3b|16001|1" in specs
    assert 'REPLACE_EXISTING="${REPLACE_EXISTING:-0}"' in script
