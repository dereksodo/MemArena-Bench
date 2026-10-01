"""Coverage for the service health matrix helper."""
from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_service_health.sh"


def test_service_health_script_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], cwd=REPO, check=True)


def test_service_health_help_documents_matrix() -> None:
    result = subprocess.run(
        [str(SCRIPT), "--help"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "/v1/models" in result.stdout
    assert "/api/v1/healthcheck" in result.stdout
    assert "/health" in result.stdout
    assert "0_6b llama3b 7b 8b 32b" in result.stdout
    assert "3b -> llama3b" in result.stdout
    assert "HEALTH_PARALLEL=0" in result.stdout


def test_service_health_script_lists_paper_ports_and_auth() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "0_6b|Qwen/Qwen3-0.6B|16000|18100|18101" in text
    assert "llama3b|meta-llama/Llama-3.2-3B-Instruct|16001|18102|18103" in text
    assert "32b|Qwen/Qwen3-32B-AWQ|16004|18108|18109" in text
    assert "MEMOBASE_API_TOKEN" in text
    assert "MEMOS_API_KEY" in text
    assert '3b) wanted="llama3b"' in text
