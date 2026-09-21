#!/usr/bin/env python3
"""BubbleML 모양의 합성 HDF5 생성기 — 실데이터 없이 파이프라인을 돌려보기 위한 것.

**이건 비등 물리의 재현이 아니다.** 수치 경로(차분·혼합 물성·뭉개기·역산·판정)가
끝까지 도는지, 그리고 게이트가 의도한 대로 분기하는지를 확인하는 용도다.

만드는 방식
  1. 케이스마다 벽 과열도 ΔT_sup 를 정한다 (Twall 목록).
  2. 목표 열유속을 q = q_cooper(ΔT_sup) * scale * (형태 왜곡) 으로 정한다.
     ``--exponent`` 를 1/0.33 = 3.03 에서 벗어나게 주면 log-log 기울기가 깨지므로
     게이트의 '불일치' 분기를 테스트할 수 있다.
  3. 벽에 붙은/떠 있는 기포를 level-set 으로 새겨 넣어 섞인 셀을 만든다.
  4. 열 경계층 두께를 **열별로** δ(x) = k_mix(x) ΔT / q 로 잡고
     T(x, y) = T_sat + ΔT exp(-y/δ(x)) 를 깐다. 이렇게 하면 푸리에 법칙의 해석해가
     모든 x 에서 정확히 q 가 되므로, 추출된 q'' 가 q 에서 벗어난 만큼이 곧 **수치
     오차**다. (실제 비등에서는 기포 뿌리에서 q''(x) 가 크게 달라지지만, 이 픽스처의
     목적은 물리 재현이 아니라 수치 경로 검증이다.)
  5. θ = (T - T_bulk)/(T_wall - T_bulk) 로 무차원화해서 저장한다 (BubbleML 규약).

격자 간격은 **가장 얇은 경계층이 ``--min-layer-cells`` 칸을 차지하도록** 자동으로
정한다. 실제 BubbleML(22.7 µm)에서는 이 층이 격자보다 얇을 수 있는데, 그건 이
합성기가 아니라 게이트가 찾아내야 할 진짜 문제다.

사용 예:
    python scripts/make_synthetic.py --out data/synthetic
    python scripts/make_synthetic.py --out data/synthetic_bad --exponent 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooperval.cooper import CooperModel, DELTA_T_EXPONENT  # noqa: E402
from cooperval.fluids import get_fluid                      # noqa: E402
from cooperval.heaviside import (                         # noqa: E402
    liquid_fraction,
    mix_conductivity,
)

KELVIN_OFFSET = 273.15


def build_case(
    T_wall_K, T_sat_K, T_bulk_K, q_target, fluid, dy, nx, ny, n_t, rng,
    bubble_coverage=0.15, contact_line_boost=1.0,
):
    delta_t = T_wall_K - T_sat_K
    delta_liquid = fluid.k_liquid * delta_t / q_target     # 순수 액체일 때의 층 두께

    y = (np.arange(ny) + 0.5) * dy                          # 셀 중심 (벽은 y=0)
    x = (np.arange(nx) + 0.5) * dy
    X, Y = np.meshgrid(x, y)                                # (ny, nx)

    temperature = np.empty((n_t, ny, nx))
    dfun = np.empty((n_t, ny, nx))

    base_radius = 6.0 * dy
    n_bubbles = max(1, int(round(bubble_coverage * nx * dy / (2 * base_radius))))

    for t in range(n_t):
        # --- 기포 배치: 일부는 벽에 붙어 있고 일부는 떠오른 상태 ---
        phi = np.full((ny, nx), 1e3 * dy)
        for _ in range(n_bubbles):
            cx = rng.uniform(x[0], x[-1])
            radius = base_radius * rng.uniform(0.7, 1.4)
            attached = rng.random() < 0.6
            cy = 0.0 if attached else rng.uniform(radius, 12 * radius)
            dist = np.hypot(X - cx, Y - cy) - radius
            phi = np.minimum(phi, dist)                     # +: 액체, -: 증기

        # --- 접촉선 증강: 기포 뿌리(= 섞인 셀)에서 국소 열유속이 가장 크다 ---
        # 4a(1-a) 는 a=0.5 인 접촉선에서 1, 순수 액체/순수 증기에서 0 이다.
        # 히터 평균이 정확히 q_target 으로 유지되도록 정규화한다.
        alpha_wall = liquid_fraction(phi[0, :], 1.5 * dy)
        shape = 1.0 + contact_line_boost * 4.0 * alpha_wall * (1.0 - alpha_wall)
        q_col = q_target * shape / shape.mean()             # (nx,)

        # --- 열별 경계층 두께: 벽 인접 셀의 혼합 열전도도에 맞춘다 ---
        # 이렇게 해야 -k dT/dy|_wall 의 해석해가 모든 x 에서 정확히 q_col(x) 가 된다.
        k_wall = mix_conductivity(alpha_wall, fluid.k_liquid, fluid.k_vapor)
        delta_col = k_wall * delta_t / q_col                # (nx,)

        temperature[t] = T_sat_K + delta_t * np.exp(-Y / delta_col[None, :])
        dfun[t] = phi

    theta = (temperature - T_bulk_K) / (T_wall_K - T_bulk_K)
    return theta, dfun, x, y, delta_liquid


def main() -> int:
    ap = argparse.ArgumentParser(description="BubbleML 모양 합성 데이터 생성")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--fluid", default="FC-72")
    ap.add_argument("--twall-c", type=float, nargs="+",
                    default=[78.0, 84.0, 90.0, 96.0, 102.0, 108.0])
    ap.add_argument("--tsat-c", type=float, default=None, help="기본값은 유체의 포화온도")
    ap.add_argument("--tbulk-c", type=float, default=None, help="기본값 = T_sat (포화 조건)")
    ap.add_argument("--exponent", type=float, default=1.0 / DELTA_T_EXPONENT,
                    help="q ∝ ΔT^exponent. 기본 3.03 (Cooper 형태). 바꾸면 기울기 게이트가 깨짐")
    ap.add_argument("--prefactor-scale", type=float, default=1.0,
                    help="q 전체를 곱으로 이동 → MAPE 는 깨지고 기울기는 유지 (조건부 통과 분기)")
    ap.add_argument("--min-layer-cells", type=float, default=3.0)
    ap.add_argument("--nx", type=int, default=192)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--contact-line-boost", type=float, default=1.0,
                    help="기포 뿌리(섞인 셀)의 국소 열유속 증강. 히터 평균은 보존된다. "
                         "0 으로 두면 q''(x) 가 균일해져 '섞인 셀 제외' 편향이 안 보인다")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fluid = get_fluid(args.fluid)
    model = CooperModel(fluid=fluid)
    T_sat = (args.tsat_c + KELVIN_OFFSET) if args.tsat_c is not None else fluid.T_sat_K
    T_bulk = (args.tbulk_c + KELVIN_OFFSET) if args.tbulk_c is not None else T_sat

    walls = [tw + KELVIN_OFFSET for tw in args.twall_c]
    delta_ts = np.array([tw - T_sat for tw in walls])
    if np.any(delta_ts <= 0):
        raise SystemExit("모든 Twall 은 T_sat 보다 커야 합니다")

    # 기준점(중앙 케이스)의 Cooper 정합 열유속에서 출발해 지정한 지수로 늘린다.
    dt_ref = float(np.median(delta_ts))
    q_ref = float(model.q_from_delta_t(np.array([dt_ref]))[0]) * args.prefactor_scale
    q_targets = q_ref * (delta_ts / dt_ref) ** args.exponent

    layers = fluid.k_liquid * delta_ts / q_targets
    dy = float(layers.min() / args.min_layer_cells)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    case_table = {}
    print(f"유체 {fluid.name} · T_sat {T_sat - KELVIN_OFFSET:.1f} °C · "
          f"dy = {dy * 1e6:.2f} µm (참고: BubbleML 은 22.7 µm)")
    for T_wall, dt, q_t in zip(walls, delta_ts, q_targets):
        theta, phi, x, y, layer = build_case(
            T_wall, T_sat, T_bulk, q_t, fluid, dy, args.nx, args.ny, args.frames, rng,
            contact_line_boost=args.contact_line_boost,
        )
        name = f"Twall-{T_wall - KELVIN_OFFSET:.0f}"
        path = out_dir / f"{name}.hdf5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("temperature", data=theta.astype("float32"))
            h5.create_dataset("dfun", data=phi.astype("float32"))
            h5.create_dataset("velx", data=np.zeros_like(theta, dtype="float32"))
            h5.create_dataset("vely", data=np.zeros_like(theta, dtype="float32"))
            h5.create_dataset("pressure", data=np.zeros_like(theta, dtype="float32"))
            h5.create_dataset("x", data=x.astype("float32"))
            h5.create_dataset("y", data=y.astype("float32"))
            h5.attrs["synthetic"] = True
            h5.attrs["q_target_W_m2"] = q_t
            h5.attrs["thermal_layer_m"] = layer
        case_table[name] = {
            "T_wall_C": T_wall - KELVIN_OFFSET,
            "T_sat_C": T_sat - KELVIN_OFFSET,
            "T_bulk_C": T_bulk - KELVIN_OFFSET,
        }
        print(f"  {path.name}: ΔT_sup {dt:5.1f} K · q_target {q_t / 1e3:8.1f} kW/m² · "
              f"층두께 {layer * 1e6:6.2f} µm ({layer / dy:.1f} 칸)")

    truth_path = out_dir / "ground_truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "fluid": fluid.name,
                "exponent": args.exponent,
                "prefactor_scale": args.prefactor_scale,
                "contact_line_boost": args.contact_line_boost,
                "dy_m": dy,
                "case_table": case_table,
                "q_target_W_m2": {
                    f"Twall-{w - KELVIN_OFFSET:.0f}": float(q)
                    for w, q in zip(walls, q_targets)
                },
                "note": "수치 경로 점검용 합성 데이터. 비등 물리의 재현이 아님.",
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n정답표: {truth_path}")
    print(f"설정에 넣을 case_table 은 위 파일의 'case_table' 키를 그대로 쓰면 됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
