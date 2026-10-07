#!/usr/bin/env python3
"""거친 입력 → Cooper 역산 → 게이트 판정 실행기.

사용 예:
    python scripts/run_gate.py --config configs/default.yaml
    python scripts/run_gate.py --config configs/default.yaml --data-root data/synthetic
    python scripts/run_gate.py --config configs/ground_truth.yaml          # 정답 기포장 표
    python scripts/run_gate.py --config configs/ground_truth.yaml \
        --ground-truth path/to/ground_truth_final.csv                      # 원본 인계 표
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
    ap.add_argument(
        "--ground-truth", default=None,
        help="정답 기포장 인계 표(ground_truth_final.csv). 주면 HDF5 대신 이 표의 q'' 를 쓴다",
    )
    ap.add_argument(
        "--q-definition", default=None, choices=["arith", "harm", "kl", "legacy"],
        help="판정에 쓸 정답 q'' 정의 (기본 arith). 나머지는 민감도 표에 같이 남는다",
    )
    ap.add_argument(
        "--slope-role", default=None, choices=["auto", "criterion", "reference"],
        help="기울기 기준의 역할 (기본 auto: 처방 사이트 수가 다르면 참고로 내림)",
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
    if args.ground_truth:
        overrides.setdefault("ground_truth", {})["csv"] = args.ground_truth
    if args.q_definition:
        overrides.setdefault("ground_truth", {})["q_definition"] = args.q_definition
    if args.slope_role:
        overrides.setdefault("gate", {})["slope_role"] = args.slope_role

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path if cfg_path.exists() else None, overrides)
    result = run_pipeline(cfg)
    gate = result["gate"]

    print()
    print("=" * 72)
    print(f"  판정   : {gate.verdict}")
    print(f"  경로   : {gate.route}")
    print(f"  MAPE   : {gate.mape:.1%}  (기준 ≤ {gate.mape_max:.0%}; "
          f"조건별 이내 {gate.n_within}/{len(gate.points)})")
    role = (f"목표 {gate.slope_target} ± {gate.slope_tol}" if gate.slope_role == "criterion"
            else "참고만 — 사이트 수 처방 자료라 판정 제외")
    print(f"  기울기 : {gate.fit.slope:.3f} ± {gate.fit.slope_stderr:.3f}  ({role})")
    print(f"  산출물 : {result['out_dir']}")
    print("=" * 72)
    for reason in gate.reasons:
        print(f"  · {reason}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
