"""Credentials passed on the eval CLI must never reach the archived args.json."""
from __future__ import annotations

import argparse
import json

from eval.cli import _archive_run_artifacts, _redact_args


def test_redact_args_blanks_keys_and_keeps_budgets() -> None:
    payload = {
        "api_key": "sk-or-v1-live",
        "judge_api_key": "sk-or-v1-judge",
        "secondary_judge_api_key": None,
        "hf_token": "hf_live",
        "max_tokens": 200,
        "system": "inmem",
    }
    out = _redact_args(payload)
    assert out["api_key"] == "<redacted>"
    assert out["judge_api_key"] == "<redacted>"
    assert out["secondary_judge_api_key"] is None
    assert out["hf_token"] == "<redacted>"
    assert out["max_tokens"] == 200
    assert out["system"] == "inmem"
    assert payload["api_key"] == "sk-or-v1-live"


def test_archived_args_json_contains_no_secret(tmp_path) -> None:
    args = argparse.Namespace(
        system="inmem",
        stages=["answer"],
        api_key="sk-or-v1-live",
        judge_api_key="sk-or-v1-judge",
        max_tokens=200,
    )
    run_dir = _archive_run_artifacts(tmp_path, "ns", args, {"evaluate": {}})
    text = (run_dir / "args.json").read_text(encoding="utf-8")
    assert "sk-or-v1" not in text
    assert json.loads(text)["max_tokens"] == 200
