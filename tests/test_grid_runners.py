"""Local coverage for the grid runners (scripts/test_cell.sh, test.sh, judge_cell.sh,
judge.sh): syntax, argument checks, dry runs, skip logic and the helper. No GPU,
Docker or API key is needed."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
RUNNERS = ["test_cell.sh", "test.sh", "judge_cell.sh", "judge.sh", "grid_common.sh"]
HELPER = SCRIPTS / "grid_helper.py"


def _run(args, **kw):
    env = dict(os.environ)
    env.pop("OPENROUTER_API_KEY", None)
    env.update(kw.pop("env", {}))
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, env=env, **kw)


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    (d / "eval_instances").mkdir(parents=True)
    for name, n in (("d1_conflict.jsonl", 3), ("d4_permission.jsonl", 5)):
        rows = [json.dumps({"instance_id": f"{name[:2]}_{i}", "query": "q"}) for i in range(n)]
        (d / "eval_instances" / name).write_text("\n".join(rows) + "\n")
    (d / "corpus_sessions.jsonl.gz").write_bytes(b"")
    return d


@pytest.mark.parametrize("script", RUNNERS)
def test_runner_syntax(script: str) -> None:
    subprocess.run(["bash", "-n", str(SCRIPTS / script)], check=True)


@pytest.mark.parametrize("script", ["test_cell.sh", "test.sh", "judge_cell.sh", "judge.sh"])
def test_runner_help(script: str) -> None:
    r = _run(["bash", str(SCRIPTS / script), "-help"])
    assert r.returncode == 0 and "Usage:" in r.stdout


@pytest.mark.parametrize("args, message", [
    (["-model", "9b", "-backend", "rag", "-gpu", "0", "-seed", "s2"], "unknown model"),
    (["-model", "8b", "-backend", "inmem", "-gpu", "0", "-seed", "s2"], "unknown backend"),
    (["-model", "8b", "-backend", "rag", "-gpu", "0,1", "-seed", "s2"], "single GPU"),
    (["-model", "8b", "-backend", "rag", "-gpu", "0", "-seed", "s5"], "unknown seed"),
    (["-model", "8b", "-backend", "rag", "-seed", "s2"], "required"),
])
def test_test_cell_rejects_bad_arguments(args, message) -> None:
    r = _run(["bash", str(SCRIPTS / "test_cell.sh"), *args])
    assert r.returncode == 2 and message in r.stderr


def test_allowed_gpus_guard() -> None:
    r = _run(["bash", str(SCRIPTS / "test_cell.sh"), "-model", "8b", "-backend", "rag", "-gpu", "0",
              "-seed", "s2", "-dry-run"], env={"MEMARENA_ALLOWED_GPUS": "3,4,6,7"})
    assert r.returncode == 2 and "MEMARENA_ALLOWED_GPUS" in r.stderr


@pytest.mark.parametrize("backend, system, expect", [
    ("vanilla", "vanilla", ["--config eval/config/pipeline_vanilla512.yaml", "--stages answer", "--answer-concurrency 64"]),
    ("oracle", "oracle", ["--config eval/config/pipeline.yaml", "--stages answer", "--answer-concurrency 64"]),
    ("rag", "inmem", ["--stages add search answer", "--answer-concurrency 64"]),
    ("memobase", "memory_cache", ["--expected-memory-system memobase", "--expected-extractor Qwen/Qwen3-8B",
                                  "--openclaw-soul-path __no_policy_playbook__", "setup_memory_backends.sh start",
                                  "build_memory_cache.py --system memobase", "--concurrency 32"]),
    ("memsearch", "memory_cache", ["--no-cache-strict", "--expected-memory-system memsearch",
                                   "build_memory_cache.py --system memsearch", "--extractor-model shared",
                                   "OLLAMA_NUM_PARALLEL=16"]),
])
def test_dry_run_prints_the_paper_commands(tmp_path, dataset, backend, system, expect) -> None:
    out = tmp_path / "runs"
    r = _run(["bash", str(SCRIPTS / "test_cell.sh"), "-model", "8b", "-backend", backend, "-gpu", "3",
              "-seed", "s3", "-dataset", str(dataset), "-out", str(out), "-dry-run"])
    assert r.returncode == 0, r.stderr
    text = r.stdout
    assert "--gpus \\\"device=3\\\"" in text
    assert "--tp 1 --dp 1 --context-length 16384 --mem-fraction-static 0.85" in text
    assert f"--system {system} " in text
    assert f"--namespace {system if backend in ('vanilla', 'oracle') else ('inmem' if backend == 'rag' else backend)}_8b_s3" in text
    assert "--temperature 0.3" in text and "--trial-seed 1003" in text
    assert "--d6-arm A --fail-on-thinking" in text
    assert "vllm" not in text
    for piece in expect:
        assert piece in text, piece
    assert not out.exists(), "a dry run must not write anything"


def test_dry_run_seed_s1_is_greedy(tmp_path, dataset) -> None:
    r = _run(["bash", str(SCRIPTS / "test_cell.sh"), "-model", "0_6b", "-backend", "oracle", "-gpu", "0",
              "-seed", "s1", "-dataset", str(dataset), "-out", str(tmp_path / "o"), "-dry-run"])
    assert "--temperature 0.0" in r.stdout and "--trial-seed 1001" in r.stdout


def test_skip_finished_cell(tmp_path, dataset) -> None:
    out = tmp_path / "runs"
    cell = out / "oracle" / "8b" / "s2"
    (cell / "oracle").mkdir(parents=True)
    (cell / "oracle" / "answer_results_oracle_8b_s2.json").write_text("[]")
    (cell / "progress.json").write_text(json.dumps({"status": "done"}))
    r = _run(["bash", str(SCRIPTS / "test_cell.sh"), "-model", "8b", "-backend", "oracle", "-gpu", "0",
              "-seed", "s2", "-dataset", str(dataset), "-out", str(out)])
    assert r.returncode == 0 and "[skip]" in r.stdout
    # a -limit run lives in its own directory and is not the finished cell
    r = _run(["bash", str(SCRIPTS / "test_cell.sh"), "-model", "8b", "-backend", "oracle", "-gpu", "0",
              "-seed", "s2", "-dataset", str(dataset), "-out", str(out), "-limit", "2", "-dry-run"])
    assert "[skip]" not in r.stdout and f"{cell}/limit2" in r.stdout


def test_judge_cell_needs_answers_and_key(tmp_path) -> None:
    out = tmp_path / "runs"
    r = _run(["bash", str(SCRIPTS / "judge_cell.sh"), "-model", "8b", "-backend", "rag", "-seed", "s2",
              "-out", str(out)])
    assert r.returncode == 2 and "no finished answers" in r.stderr
    r = _run(["bash", str(SCRIPTS / "judge_cell.sh"), "-model", "8b", "-backend", "rag", "-seed", "s2",
              "-out", str(out), "-dry-run", "-judge", "gpt4omini"])
    assert r.returncode == 0
    assert "scripts/llmjudge.py" in r.stdout and "--judge-model openai/gpt-4o-mini-2024-07-18" in r.stdout
    assert "--judge-preset remote" in r.stdout and "--concurrency 512" in r.stdout
    assert "scripts/rerun_d6_judge.py --eval-path" in r.stdout
    assert "evaluation_results_inmem_8b_s2_judge_remote.json" in r.stdout
    assert "api-key" not in r.stdout.replace("--api-key EMPTY", "")
    # the default judge is gpt-4o-mini
    r_default = _run(["bash", str(SCRIPTS / "judge_cell.sh"), "-model", "8b", "-backend", "rag", "-seed", "s2",
                      "-out", str(out), "-dry-run"])
    # the dry-run prints wall-clock times; compare with digits masked
    import re as _re
    def _mask(t: str) -> str:
        return _re.sub(r"\d", "0", t)
    assert r_default.returncode == 0 and _mask(r_default.stdout) == _mask(r.stdout)
    # DeepSeek (secondary) has its own file names
    r = _run(["bash", str(SCRIPTS / "judge_cell.sh"), "-model", "8b", "-backend", "rag", "-seed", "s2",
              "-out", str(out), "-dry-run", "-judge", "deepseek"])
    assert r.returncode == 0
    assert "--judge-model deepseek/deepseek-v4.1-flash --judge-tag deepseek" in r.stdout
    assert "evaluation_results_inmem_8b_s2_judge_deepseek.json --judge-model deepseek/deepseek-v4.1-flash --workers 128" in r.stdout
    assert "judge_remote" not in r.stdout


def test_helper_limit_dataset_and_counts(tmp_path, dataset) -> None:
    dst = tmp_path / "limit2"
    subprocess.run(["python3", str(HELPER), "make-limit-dataset", str(dataset), str(dst), "2"], check=True)
    count = subprocess.run(["python3", str(HELPER), "count-items", str(dst)], capture_output=True, text=True, check=True)
    assert count.stdout.strip() == "4"
    full = subprocess.run(["python3", str(HELPER), "count-items", str(dataset)], capture_output=True, text=True, check=True)
    assert full.stdout.strip() == "8"
    assert (dst / "corpus_sessions.jsonl.gz").is_symlink()


def test_helper_cache_check(tmp_path) -> None:
    ok = tmp_path / "ok.summary.json"
    ok.write_text(json.dumps({"status": "complete", "ingest": {"messages_failed": 0}}))
    lost = tmp_path / "lost.summary.json"
    lost.write_text(json.dumps({"status": "complete", "ingest": {"messages_failed": 4}}))
    assert subprocess.run(["python3", str(HELPER), "cache-check", str(ok)], capture_output=True).returncode == 0
    assert subprocess.run(["python3", str(HELPER), "cache-check", str(lost)], capture_output=True).returncode == 1


def test_helper_parses_progress(tmp_path) -> None:
    log = tmp_path / "a.log"
    log.write_text("x\r[eval_answer] |####----| 12/345  3.5%\r[eval_answer] |#####---| 40/345 11.6%")
    r = subprocess.run(["python3", str(HELPER), "parse-log", str(log), "answer"], capture_output=True, text=True)
    assert r.stdout.split() == ["40", "345", "eval_answer"]
    log.write_text("[day 1/15] day=0\n[day 7/15] day=6\n")
    r = subprocess.run(["python3", str(HELPER), "parse-log", str(log), "build"], capture_output=True, text=True)
    assert r.stdout.split() == ["7", "15", "days"]


def test_helper_judge_mark_touches_only_the_named_dimensions(tmp_path) -> None:
    ev = tmp_path / "eval.json"
    details = [
        {"question_id": "d6_a", "correct": True, "reason": "evidence_judge: ok"},          # D2
        {"question_id": "d10_b", "correct": False, "reason": "evidence_judge: no"},        # D3
        {"question_id": "d4_perm_L_001", "policy_category": "OTHER", "rationale_v2": "x",  # D6
         "leaked_fact_in_output": False},
    ]
    ev.write_text(json.dumps({"summary": {}, "details": details}))
    out = subprocess.run(["python3", str(HELPER), "judge-mark", str(ev), "D2,D6"],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "marked D2 1, D6 1"
    d = {r["question_id"]: r for r in json.loads(ev.read_text())["details"]}
    assert d["d6_a"]["needs_judge"] is True
    assert "needs_judge" not in d["d10_b"] and d["d10_b"]["reason"] == "evidence_judge: no"
    assert d["d4_perm_L_001"]["policy_category"] is None and d["d4_perm_L_001"]["rationale_v2"] is None
    assert d["d4_perm_L_001"]["needs_judge"] is True   # re-read from the (possibly re-answered) answer file
    pending = subprocess.run(["python3", str(HELPER), "judge-pending", str(ev)],
                             capture_output=True, text=True, check=True).stdout
    assert pending.strip() == "2"
    bad = subprocess.run(["python3", str(HELPER), "judge-mark", str(ev), "D7"], capture_output=True, text=True)
    assert bad.returncode != 0


def test_helper_splice_answers_replaces_rows_and_keeps_a_backup(tmp_path) -> None:
    ans = tmp_path / "answer_results_x.json"
    ans.write_text(json.dumps([{"question_id": "d1_a", "prediction": "old"},
                               {"question_id": "d4_perm_L_001", "prediction": "old"}]))
    new = tmp_path / "new.json"
    new.write_text(json.dumps([{"question_id": "d4_perm_L_001", "prediction": "new"}]))
    subprocess.run(["python3", str(HELPER), "splice-answers", str(ans), str(new)], check=True, capture_output=True)
    rows = json.loads(ans.read_text())
    assert [r["prediction"] for r in rows] == ["old", "new"]
    assert list(tmp_path.glob("answer_results_x.before_redo_*.json"))
    new.write_text(json.dumps([{"question_id": "d9_unknown", "prediction": "x"}]))
    assert subprocess.run(["python3", str(HELPER), "splice-answers", str(ans), str(new)], capture_output=True).returncode != 0


def test_redo_dims_resolves_task_files(tmp_path, dataset) -> None:
    out = subprocess.run(["python3", str(HELPER), "redo-dims", str(dataset), "D6"],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out[1:] == ["d4_permission"] and int(out[0]) == 5


def test_redo_dims_dry_run_answers_only_those_task_files(tmp_path, dataset) -> None:
    out = tmp_path / "runs"
    cell = out / "oracle" / "8b" / "s2"
    r = _run(["bash", str(SCRIPTS / "test_cell.sh"), "-model", "8b", "-backend", "oracle", "-gpu", "0",
              "-seed", "s2", "-dataset", str(dataset), "-out", str(out), "-redo-dims", "D6", "-dry-run"])
    assert r.returncode != 0 and "no finished answers" in r.stderr
    (cell / "oracle").mkdir(parents=True)
    (cell / "oracle" / "answer_results_oracle_8b_s2.json").write_text("[]")
    (cell / "progress.json").write_text(json.dumps({"status": "done"}))
    r = _run(["bash", str(SCRIPTS / "test_cell.sh"), "-model", "8b", "-backend", "oracle", "-gpu", "0",
              "-seed", "s2", "-dataset", str(dataset), "-out", str(out), "-redo-dims", "D6", "-dry-run"])
    assert r.returncode == 0, r.stderr
    assert f"--output-dir {cell}/redo_D6 " in r.stdout and "--dimensions d4_permission" in r.stdout
    assert "splice-answers" in r.stdout and "[skip]" not in r.stdout
    assert json.loads((cell / "progress.json").read_text()) == {"status": "done"}


def test_ablation_variant_dry_runs(tmp_path, dataset) -> None:
    out = tmp_path / "runs"
    base = ["bash", str(SCRIPTS / "test_cell.sh"), "-model", "8b", "-gpu", "0", "-seed", "s2",
            "-dataset", str(dataset), "-out", str(out), "-dry-run"]
    r = _run(base + ["-backend", "rag", "-top-k", "8"])
    assert r.returncode != 0 and "need -tag" in r.stderr
    r = _run(base + ["-backend", "rag", "-top-k", "8", "-tag", "k8"])
    assert r.returncode == 0, r.stderr
    assert f"--output-dir {out}/rag-k8/8b/s2 " in r.stdout and "--namespace inmem_k8_8b_s2" in r.stdout
    assert "--top-k 8" in r.stdout
    r = _run(base + ["-backend", "dense_bge_m3"])
    assert r.returncode == 0 and "--system dense_bge_m3 " in r.stdout and "--stages add search answer" in r.stdout
    r = _run(base + ["-backend", "omniscient"])
    assert r.returncode == 0 and "--system omniscient " in r.stdout and "--stages answer" in r.stdout
    r = _run(base + ["-backend", "oracle", "-d6-access", "-dims", "D6", "-tag", "access"])
    assert r.returncode == 0 and "--d6-inject-access" in r.stdout and "--dimensions d4_permission" in r.stdout
    r = _run(base + ["-backend", "oracle", "-memory-from", str(tmp_path), "-tag", "w"])
    assert r.returncode != 0 and "is for -backend memobase" in r.stderr
