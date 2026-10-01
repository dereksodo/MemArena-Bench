"""Print and save the cache-off latency model (memarena.figures.latency_model).

    python scripts/fit_ttft_curve.py      # reads $MEMARENA_RESULTS_DIR/out/...

Writes $MEMARENA_RESULTS_DIR/out/latency_selection/ttft_curve_models.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memarena.figures.latency_model import BACKENDS, READERS, latency_model  # noqa: E402
from memarena.figures.paper_data import RESULTS_ROOT  # noqa: E402

OUT = RESULTS_ROOT / "out" / "latency_selection" / "ttft_curve_models.json"


def main() -> int:
    m = latency_model()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(m, indent=2))
    print(f"wrote {OUT}\n")
    for r, cv in m["curves"].items():
        dv, chk = m["decode"][r], m["cross_check_old_runs"].get(r, {})
        print(f"{r:8s} TTFT = {cv['a_ms']:.1f} + {cv['b_ms_per_1k']:.2f}·N + {cv['c_ms_per_1k2']:.3f}·N²  "
              f"(N in 1k tok)  R²={cv['R2']:.4f}  |rel resid| p50={cv['rel_resid_p50']:.3f} "
              f"p95={cv['rel_resid_p95']:.3f}  n={cv['n_obs']}")
        print(f"{'':8s} decode = {dv['d0_ms_per_tok']:.2f} + {dv['d1_ms_per_tok_per_1k']:.3f}·N ms/tok")
        print(f"{'':8s} old cold requests, measured/predicted p50: " + ", ".join(
            f"{b} {v['measured_over_predicted_p50']:.2f} (n={v['n']})" for b, v in chk.items()))
    print(f"\n{'cell':20s} {'n':>4s} {'N_p p50':>8s} {'search':>7s} {'TTFT p50':>9s} {'total p50':>10s}")
    for r in READERS:
        for b in BACKENDS:
            c = m["cells"].get(f"{b}_{r}")
            if c:
                print(f"{b + '_' + r:20s} {c['n_questions']:4d} {c['n_p_p50']:8.0f} {c['t_search_p50_ms']:7.1f} "
                      f"{c['ttft_p50_ms']:9.0f} {c['total_p50_ms']:10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
