"""Smoke tests for the pinned memory backend setup helper."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "setup_memory_backends.sh"


def test_memory_backends_script_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], cwd=REPO, check=True)


def test_memory_backends_versions_are_pinned() -> None:
    result = subprocess.run(
        [str(SCRIPT), "versions"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Memobase ref:  v0.0.42" in result.stdout
    assert "MemOS ref:     v2.0.13" in result.stdout
    assert "Memobase port: 8019" in result.stdout
    assert "Memobase compose project: default" in result.stdout
    assert "MemOS port:    8020" in result.stdout
    assert "MemOS compose project: default" in result.stdout
    assert "Memory extractor reader tag: 0_6b" in result.stdout
    assert "Memory extractor model:      Qwen/Qwen3-0.6B" in result.stdout
    assert "Memory extractor endpoint:   http://host.docker.internal:16000/v1" in result.stdout
    assert "Reuse local repo cache:       yes" in result.stdout
    assert "Force docker compose build:   no" in result.stdout


def test_memory_backends_reader_tag_selects_matching_extractor() -> None:
    env = os.environ.copy()
    env["MEMORY_READER_TAG"] = "7b"
    result = subprocess.run(
        [str(SCRIPT), "versions"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Memory extractor reader tag: 7b" in result.stdout
    assert "Memory extractor model:      mistralai/Mistral-7B-Instruct-v0.3" in result.stdout
    assert "Memory extractor endpoint:   http://host.docker.internal:16002/v1" in result.stdout


def test_memory_backends_versions_accepts_reader_tag_argument() -> None:
    result = subprocess.run(
        [str(SCRIPT), "versions", "7b"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[7b] versions isolated stack: Memobase http://localhost:18104, MemOS http://localhost:18105" in result.stdout
    assert "Memobase port: 18104" in result.stdout
    assert "MemOS port:    18105" in result.stdout
    assert "Memobase compose project: memarena_7b_memobase" in result.stdout
    assert "Memory extractor reader tag: 7b" in result.stdout
    assert "Memory extractor model:      mistralai/Mistral-7B-Instruct-v0.3" in result.stdout


def test_memory_backends_compose_project_prefix_is_split_per_service() -> None:
    env = os.environ.copy()
    env["MEMORY_COMPOSE_PROJECT_PREFIX"] = "memarena_7b"
    result = subprocess.run(
        [str(SCRIPT), "versions"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Memobase compose project: memarena_7b_memobase" in result.stdout
    assert "MemOS compose project: memarena_7b_memos" in result.stdout


def test_memory_backends_help_documents_cache_controls() -> None:
    result = subprocess.run(
        [str(SCRIPT), "--help"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MEMORY_BACKENDS_UPDATE=1" in result.stdout
    assert "MEMORY_BACKENDS_BUILD=1" in result.stdout
    assert "MEMORY_BACKENDS_PARALLEL=0" in result.stdout


def test_memory_backends_tolerates_stale_partial_checkouts() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "moving non-git checkout out of the way" in script
    assert 'mv "$dir" "$stale_dir"' in script


def test_memory_backends_multi_reader_lifecycle_runs_in_parallel() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'PARALLEL_ISOLATED="${MEMORY_BACKENDS_PARALLEL:-1}"' in script
    assert "setup|start|restart|stop)" in script
    assert '"$script" "$action" "$tag"' in script
    assert ") &" in script
    assert "ERROR: [${labels[$i]}] ${action} failed" in script


def test_memory_backends_patches_fixed_compose_container_names() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "patch_memobase_compose" in script
    assert "patching fixed Memobase compose names" in script
    assert "patching fixed MemOS compose names" in script
    assert "container_name" in script
    assert "s/qdrant-docker/qdrant/g" in script
    assert "s/neo4j-docker/neo4j/g" in script
    assert "removing MemOS internal service host ports" in script
    assert '"7474:7474"' in script
    assert '"6333:6333"' in script


def test_memory_backends_multi_reader_versions_use_isolated_ports() -> None:
    result = subprocess.run(
        [str(SCRIPT), "versions", "0_6b,7b", "8b"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[0_6b] versions isolated stack: Memobase http://localhost:18100, MemOS http://localhost:18101" in result.stdout
    assert "[7b] versions isolated stack: Memobase http://localhost:18104, MemOS http://localhost:18105" in result.stdout
    assert "[8b] versions isolated stack: Memobase http://localhost:18106, MemOS http://localhost:18107" in result.stdout
    assert "Memobase port: 18100" in result.stdout
    assert "Memobase port: 18104" in result.stdout
    assert "Memobase port: 18106" in result.stdout
    assert "Memobase port: 18102" not in result.stdout


def test_memory_backends_env_accepts_reader_tag_argument() -> None:
    result = subprocess.run(
        [str(SCRIPT), "env", "llama3b"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "export MEMOBASE_BASE_URL=http://localhost:18102" in result.stdout
    assert "export MEMOS_BASE_URL=http://localhost:18103" in result.stdout
