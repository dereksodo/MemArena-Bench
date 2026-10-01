"""Selectable judge (gpt4omini primary, deepseek secondary): request kwargs, per-judge
file names and progress state, skip / force, parallel judging, the runner-layout
figure loader, judge agreement and the human-calibration re-judge. Stub clients
only; no API call, no key."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from memarena.judges import JUDGES, cohen_kappa, eval_filename, judge_request_kwargs, judge_state

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
HELPER = SCRIPTS / "grid_helper.py"
DS = "deepseek/deepseek-v4.1-flash"
MINI = "openai/gpt-4o-mini-2024-07-18"
NO_REASONING = {"reasoning": {"enabled": True},  # DeepSeek runs with reasoning on, on its own endpoint
                "provider": {"order": ["deepseek"], "allow_fallbacks": False}}


# ----------------------------------------------------------------- stub client
class _Completions:
    def __init__(self, content: str, delay: float = 0.0) -> None:
        self.content = content
        self.delay = delay
        self.calls: list[dict] = []
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            msg = SimpleNamespace(content=self.content)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        finally:
            with self.lock:
                self.in_flight -= 1


def _client(content: str, delay: float = 0.0):
    comp = _Completions(content, delay)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


JUDGE_OK = '{"correct": true, "score": 1.0, "reason": "ok"}'
D6_REFUSE = '{"label": "REFUSE", "epistemic_claim": "declines", "fact_in_output": false}'


# ----------------------------------------------------------------- registry
def test_registry_and_shell_agree() -> None:
    assert JUDGES["deepseek"].model == DS and JUDGES["gpt4omini"].model == MINI
    assert eval_filename("inmem_8b_s2", "gpt4omini") == "evaluation_results_inmem_8b_s2_judge_remote.json"
    assert eval_filename("inmem_8b_s2", "deepseek") == "evaluation_results_inmem_8b_s2_judge_deepseek.json"
    for tag, spec in JUDGES.items():
        out = subprocess.run(
            ["bash", "-c", f'. {SCRIPTS}/grid_common.sh; judge_model {tag}; judge_file_tag {tag}; echo "$GRID_JUDGES|$GRID_DEFAULT_JUDGE"'],
            capture_output=True, text=True, check=True).stdout.splitlines()
        assert out[0] == spec.model and out[1] == spec.file_tag
        assert out[2] == "deepseek gpt4omini|gpt4omini"


def test_request_kwargs_only_for_deepseek() -> None:
    assert judge_request_kwargs(DS) == {"max_tokens": 8192, "extra_body": NO_REASONING}
    assert judge_request_kwargs("DeepSeek/deepseek-chat") == {"max_tokens": 8192, "extra_body": NO_REASONING}
    assert judge_request_kwargs(MINI) == {"max_tokens": 200}


# ----------------------------------------------------------------- scoring
@pytest.mark.parametrize("model, extra", [(DS, NO_REASONING), (MINI, None)])
def test_scoring_judges_send_extra_body_for_deepseek(model, extra) -> None:
    from eval.src.scoring import _armB_judge_d4, _evidence_grounded_judge_score, _llm_judge_score

    client, comp = _client(JUDGE_OK)
    assert _llm_judge_score(client, model, "q", "gold", "pred", "d7_qa")[0] is True
    assert _evidence_grounded_judge_score(client, model, "q", "gold", "pred", "d7_qa", "ctx")[0] is True
    d6_client, d6_comp = _client(D6_REFUSE)
    assert _armB_judge_d4(d6_client, model, "q", "pred", {"fact": "x"})[0] == "REFUSE"
    for call in comp.calls + d6_comp.calls:
        assert call["model"] == model and call["temperature"] == 0.0 and call["max_tokens"] == (8192 if extra else 200)
        assert call.get("extra_body") == extra


def test_evaluate_answers_reaches_the_requested_concurrency() -> None:
    """asyncio.to_thread capped the judge at min(32, cpus + 4) calls in flight."""
    from eval.src.scoring import evaluate_answers
    from eval.src.types import AnswerRecord, QAItem

    n = 64
    qas = [QAItem(question_id=f"d4_perm_{i}", question="q", answer="",
                  metadata={"dimension": "d4_permission", "expected_answer_mode": "deny",
                            "ground_truth": {"fact": "x"}}) for i in range(n)]
    answers = [AnswerRecord(question_id=q.question_id, question="q", answer="", prediction="I can't share that.",
                            model="m", raw_response="I can't share that.") for q in qas]
    client, comp = _client(D6_REFUSE, delay=0.3)
    out = asyncio.run(evaluate_answers(qas=qas, answers=answers, judge_client=client, judge_model=DS,
                                       concurrency=n))
    assert len(comp.calls) == n and all(c["extra_body"] == NO_REASONING for c in comp.calls)
    assert comp.max_in_flight > 32
    assert out["summary"]["total"] == n


# ----------------------------------------------------------------- llmjudge
def test_llmjudge_judge_tag_names_the_output() -> None:
    from scripts.llmjudge import _output_namespace, _resolve_judge, build_arg_parser

    args = build_arg_parser().parse_args(["--judge-preset", "remote", "--judge-model", DS,
                                          "--judge-tag", "deepseek", "--judge-api-key", "stub"])
    tag, model, _endpoint, _key = _resolve_judge(args)
    assert (tag, model) == ("deepseek", DS)
    assert _output_namespace("inmem_8b_s2", tag, None) == "inmem_8b_s2_judge_deepseek"
    args = build_arg_parser().parse_args(["--judge-preset", "remote", "--judge-api-key", "stub"])
    assert _resolve_judge(args)[:2] == ("remote", MINI)  # unchanged default: *_judge_remote.json


def test_llmjudge_client_pool_fits_concurrency() -> None:
    from scripts.llmjudge import _make_client

    seen = {}

    def fake_openai(**kwargs):
        seen.update(kwargs)
        return kwargs

    _make_client(fake_openai, "k", "http://stub/v1", 512)
    pool = seen["http_client"]._transport._pool
    assert pool._max_keepalive_connections == 512 and pool._max_connections >= 512


# ----------------------------------------------------------------- D6 relabel
def _d6_eval_file(path: Path, n: int = 40) -> Path:
    details = [{"question_id": f"d4_perm_L_{i}", "prediction": "I can't say." if i else "",
                "expected_answer_mode": "deny", "gold_answer": "fact", "policy_category": "REFUSAL",
                "correct": False, "score": 0.0, "answer_scored": True} for i in range(n)]
    details.append({"question_id": "d7_1", "prediction": "x", "correct": True, "score": 1.0,
                    "answer_scored": True, "reason": "evidence_judge: ok"})
    path.write_text(json.dumps({"summary": {}, "details": details}))
    return path


def test_rerun_d6_parallel_with_deepseek(tmp_path) -> None:
    from scripts.rerun_d6_judge import process_cell

    path = _d6_eval_file(tmp_path / "evaluation_results_inmem_8b_s2_judge_deepseek.json")
    client, comp = _client(D6_REFUSE, delay=0.05)
    with ThreadPoolExecutor(max_workers=32) as pool:
        stats = process_cell(path, ("c",), client, DS, executor=pool)
    assert stats.n_done == 40 and stats.n_error == 0
    assert len(comp.calls) == 39  # the empty prediction is never sent
    assert comp.max_in_flight > 1
    assert all(c["extra_body"] == NO_REASONING and c["model"] == DS for c in comp.calls)
    data = json.loads(path.read_text())
    d6 = [d for d in data["details"] if d["question_id"].startswith("d4_perm")]
    assert {d["policy_category"] for d in d6} == {"REFUSE", "OTHER"}
    assert all(d["correct"] for d in d6)
    assert data["summary"]["d6_relabel_judge_model"] == DS
    assert (tmp_path / "evaluation_results_inmem_8b_s2_judge_deepseek_legacy.json").exists()
    # idempotent: a second pass sends nothing (the empty prediction is re-labelled OTHER locally)
    stats = process_cell(path, ("c",), client, DS)
    assert stats.n_skipped == 39 and len(comp.calls) == 39


def test_rerun_d6_gpt4omini_sends_no_extra_body(tmp_path) -> None:
    from scripts.rerun_d6_judge import process_cell

    path = _d6_eval_file(tmp_path / "e.json", n=3)
    client, comp = _client(D6_REFUSE)
    process_cell(path, ("c",), client, MINI)
    assert comp.calls and all("extra_body" not in c for c in comp.calls)


# ----------------------------------------------------------------- progress.json
def _helper(*args) -> str:
    return subprocess.run(["python3", str(HELPER), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_progress_per_judge_does_not_clobber(tmp_path) -> None:
    p = tmp_path / "progress.json"
    p.write_text(json.dumps({"status": "done"}))
    _helper("judge-set", str(p), "deepseek", "status=running", "judged=0")
    _helper("judge-set", str(p), "gpt4omini", "status=done", "accuracy=0.5", f"model={MINI}")
    _helper("judge-set", str(p), "deepseek", "status=done", "judged=10")
    data = json.loads(p.read_text())
    assert data["status"] == "done"
    assert data["judges"]["deepseek"]["status"] == "done" and data["judges"]["deepseek"]["judged"] == 10
    assert data["judges"]["gpt4omini"]["accuracy"] == "0.5"
    # gpt4omini mirrors the flat keys of the first runners; deepseek never writes them
    assert data["judge_status"] == "done" and data["judge_accuracy"] == "0.5" and "judged" not in data
    assert _helper("judge-get", str(p), "deepseek", "status") == "done"


def test_progress_legacy_flat_fields_count_for_gpt4omini(tmp_path) -> None:
    p = tmp_path / "progress.json"
    legacy = {"status": "done", "judge_status": "done", "judge_accuracy": "0.61", "judge_model": MINI,
              "judge_step1_done": True}
    p.write_text(json.dumps(legacy))
    assert _helper("judge-get", str(p), "gpt4omini", "status") == "done"
    assert _helper("judge-get", str(p), "gpt4omini", "step1_done") == "true"
    assert _helper("judge-get", str(p), "deepseek", "status") == ""
    assert judge_state(legacy, "deepseek") == {}


def test_progress_concurrent_writers_keep_both(tmp_path) -> None:
    p = tmp_path / "progress.json"
    p.write_text("{}")
    procs = [subprocess.Popen(["python3", str(HELPER), "judge-set", str(p), tag, f"n{i}={i}"])
             for i in range(12) for tag in ("deepseek", "gpt4omini")]
    for proc in procs:
        assert proc.wait() == 0
    data = json.loads(p.read_text())
    assert len(data["judges"]["deepseek"]) - 1 == 12 and len(data["judges"]["gpt4omini"]) - 1 == 12


# ----------------------------------------------------------------- judge_cell / judge.sh
FAKE_LLMJUDGE = r'''
import json, sys, time
from pathlib import Path
a = sys.argv[1:]
get = lambda k: a[a.index(k) + 1]
ans = Path(get("--answer-path")); tag = get("--judge-tag"); model = get("--judge-model")
ns = ans.stem[len("answer_results_"):]
rows = json.loads(ans.read_text())
det = [{"question_id": r["question_id"], "prediction": r["prediction"], "correct": True, "score": 1.0,
        "answer_scored": True, "reason": "evidence_judge: ok", "expected_answer_mode": "deny"} for r in rows]
out = ans.parent / f"evaluation_results_{ns}_judge_{tag}.json"
out.write_text(json.dumps({"summary": {"accuracy": 1.0, "judge_model": model}, "details": det}))
with open(Path(get("--manifest-out")).parent / "calls.txt", "a") as f:
    f.write(f"llmjudge {tag} {model} {get('--concurrency')}\n")
time.sleep(float(__import__("os").environ.get("FAKE_SLEEP", "0")))
'''
FAKE_RERUN = r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
get = lambda k: a[a.index(k) + 1]
p = Path(get("--eval-path"))
d = json.loads(p.read_text())
for r in d["details"]:
    if r["question_id"].startswith("d4_perm"):
        r.update(policy_category="REFUSE", rationale_v2="x", correct=True)
p.write_text(json.dumps(d))
with open(p.parents[1] / "logs" / "calls.txt", "a") as f:
    f.write(f"rerun {get('--judge-model')} {get('--workers')}\n")
print("records_error=0")
'''


@pytest.fixture()
def fake_grid(tmp_path):
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "llmjudge.py").write_text(FAKE_LLMJUDGE)
    (fake / "rerun.py").write_text(FAKE_RERUN)
    py = fake / "python"
    py.write_text('#!/usr/bin/env bash\ns="$1"; shift\n'
                  f'case "$s" in scripts/llmjudge.py) exec python3 {fake}/llmjudge.py "$@";; '
                  f'scripts/rerun_d6_judge.py) exec python3 {fake}/rerun.py "$@";; esac\nexit 9\n')
    py.chmod(0o755)
    data = tmp_path / "data"
    (data / "eval_instances").mkdir(parents=True)
    runs = tmp_path / "runs"

    def make_cell(backend: str, model: str, seed: str, progress: dict | None = None) -> Path:
        system = {"rag": "inmem", "memobase": "memory_cache", "memsearch": "memory_cache"}.get(backend, backend)
        ns = f"{'inmem' if backend == 'rag' else backend}_{model}_{seed}"
        cell = runs / backend / model / seed
        (cell / system).mkdir(parents=True)
        rows = [{"question_id": q, "prediction": "no"} for q in ("d7_1", "d4_perm_L_1")]
        (cell / system / f"answer_results_{ns}.json").write_text(json.dumps(rows))
        (cell / "progress.json").write_text(json.dumps(progress or {"status": "done"}))
        (cell / "args.json").write_text(json.dumps({"answered_dataset": str(data)}))
        return cell

    env = {"PYTHON": str(py), "OPENROUTER_API_KEY": "stub-not-a-key"}
    return SimpleNamespace(runs=runs, make_cell=make_cell, env=env)


def _run(args, env):
    full = dict(os.environ)
    full.update(env)
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, env=full)


def _judge_cell(fg, *extra):
    return _run(["bash", str(SCRIPTS / "judge_cell.sh"), "-model", "8b", "-backend", "rag", "-seed", "s2",
                 "-out", str(fg.runs), *extra], fg.env)


def test_judge_cell_per_judge_files_skip_and_force(fake_grid) -> None:
    fg = fake_grid
    cell = fg.make_cell("rag", "8b", "s2")
    r = _judge_cell(fg, "-judge", "deepseek")
    assert r.returncode == 0, r.stdout + r.stderr
    ds_file = cell / "inmem" / "evaluation_results_inmem_8b_s2_judge_deepseek.json"
    assert ds_file.exists() and (cell / "logs" / "judge_deepseek.log").exists()
    assert not (cell / "inmem" / "evaluation_results_inmem_8b_s2_judge_remote.json").exists()
    calls = (cell / "logs" / "calls.txt").read_text().splitlines()
    assert calls == [f"llmjudge deepseek {DS} 512", f"rerun {DS} 128"]
    prog = json.loads((cell / "progress.json").read_text())
    assert prog["judges"]["deepseek"]["status"] == "done" and prog["judges"]["deepseek"]["model"] == DS
    assert prog["judges"]["deepseek"]["judged"] == 2 and prog["judges"]["deepseek"]["eval_file"] == str(ds_file)
    assert "judge_status" not in prog  # the gpt4omini flat fields are left alone

    r = _judge_cell(fg, "-judge", "deepseek")
    assert r.returncode == 0 and "[skip]" in r.stdout
    assert len((cell / "logs" / "calls.txt").read_text().splitlines()) == 2

    r = _judge_cell(fg, "-judge", "gpt4omini", "-workers", "7")
    assert r.returncode == 0, r.stdout + r.stderr
    mini_file = cell / "inmem" / "evaluation_results_inmem_8b_s2_judge_remote.json"
    assert mini_file.exists() and (cell / "logs" / "judge.log").exists()
    assert (cell / "logs" / "calls.txt").read_text().splitlines()[2:] == [
        f"llmjudge remote {MINI} 512", f"rerun {MINI} 7"]
    prog = json.loads((cell / "progress.json").read_text())
    assert prog["judges"]["deepseek"]["status"] == "done"
    assert prog["judges"]["gpt4omini"]["status"] == "done" and prog["judge_status"] == "done"
    assert prog["eval_file"] == str(mini_file) and prog["judge_model"] == MINI

    r = _judge_cell(fg, "-force")
    assert r.returncode == 0 and "[skip]" not in r.stdout
    assert len((cell / "logs" / "calls.txt").read_text().splitlines()) == 6


def test_judge_cell_rejects_unknown_judge(fake_grid) -> None:
    fake_grid.make_cell("rag", "8b", "s2")
    r = _judge_cell(fake_grid, "-judge", "claude")
    assert r.returncode == 2 and "unknown judge" in r.stderr


def test_judge_cell_skips_a_running_legacy_gpt4omini_cell_only_for_that_judge(fake_grid) -> None:
    """A cell the old runner judged with gpt-4o-mini counts as done for gpt4omini, not for deepseek."""
    fg = fake_grid
    cell = fg.make_cell("rag", "8b", "s2", {"status": "done", "judge_status": "done", "judge_model": MINI})
    (cell / "inmem" / "evaluation_results_inmem_8b_s2_judge_remote.json").write_text('{"details": []}')
    r = _judge_cell(fg, "-judge", "gpt4omini")
    assert r.returncode == 0 and "[skip]" in r.stdout
    r = _judge_cell(fg, "-judge", "deepseek")
    assert r.returncode == 0 and "[skip]" not in r.stdout


def test_judge_sh_parallel(fake_grid) -> None:
    fg = fake_grid
    cells = [fg.make_cell(b, m, s) for s in ("s2", "s3") for m in ("8b", "32b") for b in ("rag", "oracle")]
    t0 = time.time()
    r = _run(["bash", str(SCRIPTS / "judge.sh"), "-seed", "s2,s3", "-models", "8b 32b", "-backends", "rag oracle",
              "-parallel", "4", "-out", str(fg.runs), "-judge", "deepseek"], {**fg.env, "FAKE_SLEEP": "1.5"})
    elapsed = time.time() - t0
    assert r.returncode == 0, r.stdout + r.stderr
    assert "judged 8, skipped 0, failed 0" in r.stdout
    # one after another each cell takes >= 1.5 s (fake llmjudge) + 2 x ~1 s (step polling)
    assert elapsed < 8 * 3.5 / 2  # cells overlapped
    for cell in cells:
        prog = json.loads((cell / "progress.json").read_text())
        assert prog["judges"]["deepseek"]["status"] == "done"
        assert (cell / "logs" / "judge_deepseek.log").exists()
    outs = list((fg.runs / "_grid_logs").glob("judge_deepseek_s2-s3_*/*.out"))
    assert len(outs) == 8
    summary = r.stdout.split("==== judge.sh summary")[1]
    assert summary.count(" done ") == 8
    # second run: every cell is skipped
    r = _run(["bash", str(SCRIPTS / "judge.sh"), "-seed", "s2,s3", "-models", "8b 32b", "-backends", "rag oracle",
              "-parallel", "3", "-out", str(fg.runs), "-judge", "deepseek"], fg.env)
    assert "judged 0, skipped 8, failed 0" in r.stdout


# ----------------------------------------------------------------- figures + agreement
def _write_eval(runs: Path, backend: str, model: str, seed: str, tag: str, details: list, status="done") -> Path:
    from memarena.figures.paper_data import runs_eval_path

    path = runs_eval_path(runs, backend, model, seed, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": {}, "details": details}))
    prog = path.parent.parent / "progress.json"
    data = json.loads(prog.read_text()) if prog.exists() else {"status": "done"}
    data.setdefault("judges", {})[tag] = {"status": status}
    prog.write_text(json.dumps(data))
    return path


def _rec(qid, correct, reason="evidence_judge: ok", **kw):
    return {"question_id": qid, "correct": correct, "reason": reason, "answer_scored": True, "prediction": "p", **kw}


def test_paper_data_reads_runner_layout_per_judge(tmp_path, monkeypatch) -> None:
    from memarena.figures import paper_data

    runs = tmp_path / "runs"
    _write_eval(runs, "rag", "8b", "s2", "deepseek", [_rec("d7_1", True), _rec("d7_2", True)])
    _write_eval(runs, "rag", "8b", "s2", "gpt4omini", [_rec("d7_1", True), _rec("d7_2", False)])
    _write_eval(runs, "oracle", "8b", "s3", "deepseek", [_rec("d7_1", True)], status="running")
    monkeypatch.setenv("MEMARENA_RUNS_DIR", str(runs))
    monkeypatch.delenv("MEMARENA_JUDGE", raising=False)
    grid = paper_data.load_all_cells()  # default judge: gpt4omini
    assert set(grid) == {("s2", "rag", "8b")}
    assert grid[("s2", "rag", "8b")].accuracy() == 0.5
    assert grid[("s2", "rag", "8b")].source_path.name.endswith("_judge_remote.json")
    monkeypatch.setenv("MEMARENA_JUDGE", "deepseek")
    grid = paper_data.load_all_cells(include_ablation=False)
    assert set(grid) == {("s2", "rag", "8b")}  # the unfinished deepseek cell is left out
    assert grid[("s2", "rag", "8b")].accuracy() == 1.0
    assert grid[("s2", "rag", "8b")].source_path.name == "evaluation_results_inmem_8b_s2_judge_deepseek.json"
    monkeypatch.delenv("MEMARENA_RUNS_DIR")
    csv = tmp_path / "index.csv"
    csv.write_text("table,trial,backend,model_tag,json_path\n")
    assert paper_data.load_all_cells(index_csv=csv) == {}  # old layout still used without the env var


def test_cohen_kappa_known_values() -> None:
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    # po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5 -> 0.4
    a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    b = [1, 1, 1, 1, 0, 1, 1, 0, 0, 0]
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(["A", "B", "C"], ["A", "B", "C"]) == 1.0


def test_judge_agreement_on_synthetic_cells(tmp_path) -> None:
    from scripts.judge_agreement import main

    runs = tmp_path / "runs"
    a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    b = [1, 1, 1, 1, 0, 1, 1, 0, 0, 0]
    da = [_rec(f"d7_{i}", bool(x)) for i, x in enumerate(a)]
    db = [_rec(f"d7_{i}", bool(x)) for i, x in enumerate(b)]
    # deterministic records: excluded and reported
    da += [_rec("d5_1", True, reason="cloze_3/5"), _rec("d3_1", False, reason="refusal_detection")]
    db += [_rec("d5_1", True, reason="cloze_3/5"), _rec("d3_1", False, reason="refusal_detection")]
    # D6: 4 agreeing labels, 1 disagreeing on the leak, 1 empty prediction
    la = ["REFUSE", "DONT_KNOW", "OTHER", "DISCLOSE_CORRECT", "DISCLOSE_CORRECT"]
    lb = ["REFUSE", "DONT_KNOW", "OTHER", "DISCLOSE_CORRECT", "DISCLOSE_WRONG"]
    da += [_rec(f"d4_perm_{i}", True, reason="x", policy_category=x) for i, x in enumerate(la)]
    db += [_rec(f"d4_perm_{i}", True, reason="x", policy_category=x) for i, x in enumerate(lb)]
    da.append(_rec("d4_perm_9", True, reason="x", policy_category="OTHER", prediction=""))
    db.append(_rec("d4_perm_9", True, reason="x", policy_category="OTHER", prediction=""))
    _write_eval(runs, "rag", "8b", "s2", "deepseek", da)
    _write_eval(runs, "rag", "8b", "s2", "gpt4omini", db)
    _write_eval(runs, "oracle", "8b", "s2", "deepseek", da)  # no gpt4omini: left out
    md, csv_path = tmp_path / "a.md", tmp_path / "a.csv"
    assert main(["--runs-dir", str(runs), "--seeds", "s2", "--models", "8b",
                 "--out-md", str(md), "--out-csv", str(csv_path)]) == 0
    import csv

    rows = {(r["scope"], r["key"]): r for r in csv.DictReader(csv_path.open())}
    pooled = rows[("pooled", "all")]
    assert int(pooled["binary_n"]) == 10
    assert float(pooled["binary_agreement"]) == pytest.approx(0.7)
    assert float(pooled["binary_kappa"]) == pytest.approx(0.4)
    assert int(pooled["d6_n"]) == 5 and float(pooled["d6_label_agreement"]) == pytest.approx(0.8)
    # leak: a = [0,0,0,1,1], b = [0,0,0,1,0] -> po 0.8, pe 0.6*0.8 + 0.4*0.2 = 0.56
    assert float(pooled["d6_leak_kappa"]) == pytest.approx((0.8 - 0.56) / 0.44)
    assert ("dimension", "D3") in rows and ("dimension", "D6") in rows
    text = md.read_text()
    assert "cloze_k/n" in text and "refusal_detection" in text and "D6 empty prediction" in text
    assert "oracle/8b/s2" in text and "no gpt4omini file" in text


# ----------------------------------------------------------------- human calibration
def test_rejudge_sample_with_stub_client(tmp_path) -> None:
    from eval.src.types import QAItem
    from memarena.human_calibration.analyze_judge_human import main as analyze
    from memarena.human_calibration.rejudge_sample import rejudge_sample

    qa = QAItem(question_id="d7_a", question="When?", answer="Monday", metadata={
        "dimension": "d7_qa", "gold_ground_truth": {"answer": "Monday"}})
    sample = {"metadata": {"n": 2}, "sample": [
        {"id": "d7_a", "dim": "D3", "dim_kind": "binary", "gold": "Monday", "prediction": "On Monday",
         "judge_correct": False, "judge_label": "INCORRECT", "judge_reason": "old"},
        {"id": "d4_perm_L_1", "dim": "D6", "dim_kind": "d6", "gold": "fact", "prediction": "I won't say.",
         "expected_answer_mode": "deny", "judge_correct": True, "judge_label": "OTHER", "judge_reason": "old"},
    ]}

    class Router:
        """binary judge JSON for the evidence/LLM judge, 5-label JSON for D6."""
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kw):
            self.calls.append(kw)
            system = kw["messages"][0]["content"]
            content = D6_REFUSE if "DISCLOSE_CORRECT" in system else JUDGE_OK
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = Router()
    out = rejudge_sample(sample, {"d7_a": qa}, client, "deepseek", corpus={}, workers=2)
    rows = {r["id"]: r for r in out["sample"]}
    assert rows["d7_a"]["judge_correct"] is True and rows["d7_a"]["judge_correct_original"] is False
    assert rows["d4_perm_L_1"]["judge_label"] == "REFUSE" and rows["d4_perm_L_1"]["judge_correct"] is True
    assert out["metadata"]["judge_model"] == DS and out["metadata"]["n_verdicts_changed_vs_original"] == 2
    assert client.calls and all(c["extra_body"] == NO_REASONING and c["temperature"] == 0.0 for c in client.calls)
    # the binary judge saw the gold
    binary_call = next(c for c in client.calls if "DISCLOSE_CORRECT" not in c["messages"][0]["content"])
    assert json.loads(binary_call["messages"][1]["content"])["gold_answer"] == "Monday"

    pool = tmp_path / "sample_judge_deepseek.json"
    pool.write_text(json.dumps(out))
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"reviewers": {"0": {"d7_a": 1, "d4_perm_L_1": "REFUSE"},
                                                "1": {"d7_a": 1, "d4_perm_L_1": "REFUSE"},
                                                "2": {"d7_a": 0, "d4_perm_L_1": "skip"}}}))
    js = tmp_path / "m.json"
    assert analyze(["--labels", str(labels), "--pool", str(pool), "--json", str(js),
                    "--tex", str(tmp_path / "m.tex")]) == 0
    m = json.loads(js.read_text())
    assert m["meta"]["judge"] == "deepseek" and m["overall"]["n"] == 1 and m["overall"]["agreement"] == 1.0
    assert m["d6"]["n"] == 1 and m["d6"]["label_agreement"] == 1.0
