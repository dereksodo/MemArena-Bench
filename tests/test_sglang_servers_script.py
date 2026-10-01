"""Coverage for the Docker SGLang server launcher script."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "start_sglang_servers.sh"


def test_sglang_server_script_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_sglang_server_help_documents_single_model_start() -> None:
    result = subprocess.run(
        [str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[--image IMAGE]" in result.stdout
    assert "[start|restart|stop|status] [MODEL_TAG ...]" in result.stdout
    assert "logs MODEL_TAG" in result.stdout
    assert "start 0_6b llama3b 7b 8b 32b" in result.stdout
    assert "--image ghcr.io/acme/sglang:cuda12 start 0_6b" in result.stdout
    assert "SGLANG_PIP_INSTALL" in result.stdout
    assert "SGLANG_32B_AWQ_PIP_INSTALL" in result.stdout


def test_sglang_server_defaults_include_serving_limits() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"' in text
    assert 'SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"' in text
    assert 'SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-128}"' in text
    assert "--context-length" in text
    assert "--mem-fraction-static" in text
    assert "--max-running-requests" in text


def test_sglang_server_installs_tokenizer_runtime_deps_before_launch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'SGLANG_PIP_INSTALL="protobuf sentencepiece"' in text
    assert '--env "SGLANG_PIP_INSTALL=${container_pip_install}"' in text
    assert "python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}" in text
    assert 'exec "$@"' in text


def test_sglang_server_installs_vllm_only_for_32b_awq() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'SGLANG_32B_AWQ_PIP_INSTALL="vllm==0.7.2"' in text
    assert 'if [[ "$tag" == "32b" && -n "${SGLANG_32B_AWQ_PIP_INSTALL:-}" ]]; then' in text
    assert 'container_pip_install="${container_pip_install} ${SGLANG_32B_AWQ_PIP_INSTALL}"' in text


def test_sglang_server_script_supports_selected_model_actions() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "start_selected" in text
    assert "status_selected" in text
    assert "stop_selected" in text
    assert 'start_selected "$@"' in text
    assert 'status_selected "$@"' in text
    assert 'stop_selected "$@"' in text


def test_sglang_server_script_starts_multiple_models_concurrently() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "start_many" in text
    assert 'start_one "$spec" &' in text
    assert "wait_ready_many" in text
    assert 'wait_ready_one "$tag" "$port" &' in text


def test_sglang_server_script_supports_cli_image_override() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "parse_args" in text
    assert "--image)" in text
    assert "--image=*)" in text
    assert 'SGLANG_IMAGE="$2"' in text
    assert "image=${SGLANG_IMAGE}" in text
