#!/usr/bin/env python3
"""거친 입력 → Cooper 역산 → 게이트 판정 실행기.

사용 예:
    python scripts/run_gate.py --config configs/default.yaml
    python scripts/run_gate.py --config configs/default.yaml --data-root data/synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooperval.config import load_config          # noqa: E402
from cooperval.pipeline import run_pipeline       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="BubbleML 거친 입력 Cooper 검증 게이트")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root", default=None, help="설정의 data.root 을 덮어씀")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--order", type=int, default=None, help="기준 차분 차수 (기본 2)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--exclude-mixed-cells", action="store_true",
        help="섞인 셀을 결측 처리하는 옛 방식 재현 (편향 크기 확인용)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s | %(message)s",
    )

    overrides: dict = {}
    if args.data_root:
        overrides.setdefault("data", {})["root"] = args.data_root
    if args.out_dir:
        overrides.setdefault("run", {})["out_dir"] = args.out_dir
    if args.run_id:
        overrides.setdefault("run", {})["run_id"] = args.run_id
    if args.order:
        overrides.setdefault("wall_flux", {})["order"] = args.order
    if args.no_plots:
        overrides.setdefault("report", {})["make_plots"] = False
    if args.exclude_mixed_cells:
        overrides.setdefault("wall_flux", {})["include_mixed_cells"] = False

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path if cfg_path.exists() else None, overrides)
    result = run_pipeline(cfg)
    gate = result["gate"]

    print()
    print("=" * 72)
    print(f"  판정   : {gate.verdict}")
    print(f"  경로   : {gate.route}")
    print(f"  MAPE   : {gate.mape:.1%}  (기준 ≤ {gate.mape_max:.0%})")
    print(f"  기울기 : {gate.fit.slope:.3f} ± {gate.fit.slope_stderr:.3f}  "
          f"(목표 {gate.slope_target} ± {gate.slope_tol})")
    print(f"  산출물 : {result['out_dir']}")
    print("=" * 72)
    for reason in gate.reasons:
        print(f"  · {reason}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
