"""Wave-parallel batch scheduling for simulation stages.

Organizes LLM calls into waves of independent tasks that can be
executed concurrently, maximizing GPU utilization.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from MASim.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


@dataclass
class WaveResult:
    """Result of a single wave execution."""
    wave_id: int
    n_tasks: int
    n_success: int
    n_failed: int
    duration_seconds: float
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class BatchScheduler:
    """Execute independent tasks in waves for maximum throughput.

    Wave pattern for simulation:
    1. Broadcast events (fast, no LLM)
    2. Batch conversation initiations (LLM: decide_to_converse)
    3. Batch dialogue turns (LLM: generate turns in parallel sessions)
    """

    def __init__(self, max_concurrent: int = 8):
        self.max_concurrent = max_concurrent
        self.wave_log: List[WaveResult] = []

    def schedule_wave(
        self,
        tasks: List[Callable[[], T]],
        wave_id: int = 0,
        label: str = "",
    ) -> WaveResult:
        """Execute a list of independent callables concurrently.

        Args:
            tasks: List of zero-argument callables.
            wave_id: Identifier for logging.
            label: Human-readable description of what this wave does.

        Returns:
            WaveResult with all results and error info.
        """
        start = time.monotonic()
        results = [None] * len(tasks)
        errors: List[str] = []
        n_success = 0

        with ThreadPoolExecutor(max_workers=min(self.max_concurrent, len(tasks))) as ex:
            fut_to_idx = {ex.submit(task): i for i, task in enumerate(tasks)}
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    results[idx] = fut.result()
                    n_success += 1
                except Exception as e:
                    errors.append(f"Task {idx}: {e}")
                    log.warning("Wave %d task %d failed: %s", wave_id, idx, e)

        duration = time.monotonic() - start
        wave_result = WaveResult(
            wave_id=wave_id,
            n_tasks=len(tasks),
            n_success=n_success,
            n_failed=len(errors),
            duration_seconds=duration,
            results=results,
            errors=errors,
        )
        self.wave_log.append(wave_result)

        tag = f" [{label}]" if label else ""
        log.info(
            "Wave %d%s: %d/%d tasks succeeded in %.2fs",
            wave_id, tag, n_success, len(tasks), duration,
        )
        return wave_result

    def schedule_pipeline(
        self,
        waves: List[List[Callable]],
    ) -> List[WaveResult]:
        """Execute multiple sequential waves, each containing parallel tasks."""
        results = []
        for i, wave_tasks in enumerate(waves):
            if not wave_tasks:
                continue
            result = self.schedule_wave(wave_tasks, wave_id=i)
            results.append(result)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all waves."""
        total_tasks = sum(w.n_tasks for w in self.wave_log)
        total_success = sum(w.n_success for w in self.wave_log)
        total_duration = sum(w.duration_seconds for w in self.wave_log)
        return {
            "n_waves": len(self.wave_log),
            "total_tasks": total_tasks,
            "total_success": total_success,
            "total_failed": total_tasks - total_success,
            "total_duration_seconds": round(total_duration, 2),
            "avg_wave_duration": round(total_duration / len(self.wave_log), 2) if self.wave_log else 0,
        }
