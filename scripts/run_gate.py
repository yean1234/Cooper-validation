#!/usr/bin/env python3
"""거친 입력 → Cooper 역산 → 게이트 판정 실행기.

사용 예:
    python scripts/run_gate.py --config configs/default.yaml
    python scripts/run_gate.py --config configs/default.yaml --data-root data/synthetic
    python scripts/run_gate.py --config configs/ground_truth.yaml          # 정답 기포장 표
    python scripts/run_gate.py --config configs/ground_truth.yaml \
        --ground-truth path/to/ground_truth_final.csv                      # 원본 인계 표
    python scripts/run_gate.py --config configs/experiments.yaml           # 실측 비등곡선
    python scripts/run_gate.py --config configs/experiments.yaml --experiments a.csv b.csv
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
    ap.add_argument(
        "--experiments", nargs="+", default=None,
        help="실측 비등곡선 CSV (여러 개 가능). 주면 실측 게이트를 돈다",
    )
    ap.add_argument("--q-min", type=float, default=None, help="실측 평가 구간 하한 [W/m²]")
    ap.add_argument("--q-max-frac", type=float, default=None, help="실측 평가 구간 상한 (곡선 최대 q 대비)")
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
    if args.experiments:
        overrides.setdefault("experiments", {})["csv"] = args.experiments
    if args.q_min is not None:
        overrides.setdefault("experiments", {})["q_min_W_m2"] = args.q_min
    if args.q_max_frac is not None:
        overrides.setdefault("experiments", {})["q_max_frac"] = args.q_max_frac

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path if cfg_path.exists() else None, overrides)
    result = run_pipeline(cfg)
    if "experiments" in result:
        return _print_experiments(result)
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


def _print_experiments(result) -> int:
    run = result["experiments"]
    c = run.verdict_counts
    print()
    print("=" * 72)
    print(f"  실측 곡선 {len(run.results)}개 — 통과 {c.get('통과', 0)} · 조건부 {c.get('조건부', 0)} · "
          f"불일치 {c.get('불일치', 0)} · 판정 불가 {c.get('판정 불가', 0)}")
    print(f"  구간 안 전체 점 MAPE {run.pooled_mape:.1%}, 평균 부호 오차 {run.pooled_bias:+.1%} "
          "(+면 Cooper 과열도가 높음)")
    for r in run.results:
        g = r.gate
        slope = f"{g.fit.slope:.3f}" if g else "—"
        mape = f"{g.mape:.0%}" if g else "—"
        print(f"  · {r.curve.curve_id:<22} {r.curve.p_Pa / 1e5:5.2f} bar  기울기 {slope:>6}  MAPE {mape:>5}  "
              f"{r.verdict}")
    for t in run.trends:
        print(f"  압력 항 [{t.group}]: 실측 ΔT ∝ p_r^{t.fit_meas.slope:.2f}, Cooper p_r^{t.fit_cooper.slope:.2f}")
    print(f"  산출물 : {result['out_dir']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
