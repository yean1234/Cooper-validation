#!/usr/bin/env python3
"""BubbleML HDF5 를 열어서 '우리가 가정한 것들이 맞는지' 확인해주는 도구.

코드를 돌리기 전에 **가장 먼저** 실행하세요. 어려운 용어 없이, 확인해야 할 것마다
[확인됨] / [확인 필요] / [문제] 로 답을 찍어 줍니다.

확인하는 것
  1. 파일에 뭐가 들어 있나 (키, 모양, 값 범위)
  2. 온도가 '실제 온도(섭씨/켈빈)'인가 '비율(무차원)'인가
  3. 벽 온도가 고정값인가 (= Dirichlet 경계인가)
  4. 좌표가 미터인가 비율인가 → 비율이면 기준 길이를 역산
  5. dfun 의 어느 쪽이 액체인가
  6. 시뮬레이션 설정값(runtime params)에 물성이 들어 있나

사용:
    python scripts/inspect_bubbleml.py data/bubbleml/Twall-90.hdf5
    python scripts/inspect_bubbleml.py data/bubbleml/*.hdf5 --expected-grid-um 22.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooperval.fluids import get_fluid  # noqa: E402

OK, CHECK, BAD = "[확인됨]", "[확인 필요]", "[문제]"


def _line(title: str) -> None:
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def _axis(arr, which: str) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 2:
        a = a[0]
    if a.ndim == 2:
        a = a[0, :] if which == "x" else a[:, 0]
    return np.asarray(a, dtype=float).ravel()


def inspect(path: Path, expected_grid_um: float, fluid_name: str) -> None:
    print(f"\n\n{'=' * 72}\n  {path.name}\n{'=' * 72}")
    fluid = get_fluid(fluid_name)

    with h5py.File(path, "r") as h5:
        keys = sorted(h5.keys())

        # ---------- 1. 파일에 뭐가 있나 ----------
        _line("1. 파일에 들어 있는 것")
        for k in keys:
            try:
                d = h5[k]
                shape = getattr(d, "shape", "?")
                print(f"  {k:28s} shape={shape}")
            except Exception:
                print(f"  {k:28s} (읽기 실패)")

        if "temperature" not in h5 or "dfun" not in h5:
            print(f"\n{BAD} temperature 또는 dfun 이 없습니다. 키 이름이 다를 수 있으니 "
                  f"위 목록을 보고 configs 의 data.fields 를 고치세요.")
            return

        T = np.asarray(h5["temperature"][()], dtype=float)
        phi = np.asarray(h5["dfun"][()], dtype=float)

        # ---------- 2. 온도가 실제 온도인가 비율인가 ----------
        _line("2. 온도 필드 — 실제 온도인가, 비율(무차원)인가?")
        tmin, tmax = float(T.min()), float(T.max())
        print(f"  값 범위: {tmin:.4f} ~ {tmax:.4f}   (모양 {T.shape})")
        if tmax > 200:
            print(f"  {OK} 켈빈(K)으로 저장된 실제 온도로 보입니다.")
            print(f"       → configs 에 nondim.temperature_mode: kelvin")
            temp_mode = "kelvin"
        elif tmax > 40:
            print(f"  {OK} 섭씨(°C)로 저장된 실제 온도로 보입니다.")
            print(f"       → configs 에 nondim.temperature_mode: celsius")
            temp_mode = "celsius"
        else:
            print(f"  {OK} **비율(무차원)** 입니다. 실제 온도가 아닙니다.")
            print(f"       → 실제 온도로 되돌리려면 T_wall, T_bulk 를 알아야 합니다 (아래 3번).")
            print(f"       → configs 에 nondim.temperature_mode: nondimensional")
            temp_mode = "nondimensional"

        # ---------- 3. 벽 온도가 고정인가 (Dirichlet?) ----------
        _line("3. 벽 경계 — 벽 온도가 고정값인가? (Dirichlet 인가?)")
        bottom, top = T[:, 0, :], T[:, -1, :]
        for name, row in (("맨 아래 줄", bottom), ("맨 위 줄", top)):
            spread = float(row.max() - row.min())
            scale = max(abs(float(row.mean())), 1e-12)
            print(f"  {name}: 평균 {row.mean():.4f}, 편차폭 {spread:.2e} "
                  f"(상대 {spread / scale:.2e})")
        b_spread = float(bottom.max() - bottom.min()) / max(abs(float(bottom.mean())), 1e-12)

        if b_spread < 1e-6:
            print(f"\n  {OK} 맨 아래 줄이 위치·시간에 관계없이 **완전히 같은 값**입니다.")
            print(f"       → 벽 온도를 못 박고 돌린 것(Dirichlet)이고, 그 줄이 곧 벽입니다.")
            print(f"       → 정답 ΔT 를 이 값에서 바로 읽을 수 있습니다. 가장 좋은 경우입니다.")
        else:
            print(f"\n  {CHECK} 맨 아래 줄이 위치/시간마다 다릅니다. 두 가지 가능성:")
            print(f"       (a) 그 줄이 벽이 아니라 **벽에서 반 칸 떨어진 첫 셀**이다  ← 보통 이쪽")
            print(f"           그러면 벽 온도는 파일 밖(논문/설정)에서 가져와야 합니다.")
            print(f"       (b) 히터 고체까지 같이 푼 것(conjugate)이라 벽 온도가 필드다")
            print(f"           그러면 '정답 ΔT' 를 하나로 못 정하므로 정의를 바꿔야 합니다.")
            print(f"       구분법: 아래 '최댓값' 을 보세요.")

        near_max = float(np.mean(T >= T.max() * (1 - 1e-9))) if T.max() > 0 else 0.0
        print(f"\n  전체 최댓값 {tmax:.6f}, 그 값과 같은 셀의 비율 {near_max:.3%}")
        if temp_mode == "nondimensional" and abs(tmax - 1.0) < 1e-4:
            print(f"  {OK} 최댓값이 정확히 1.0 입니다 → 비율의 기준이 벽 온도라는 뜻입니다.")
            print(f"       즉 규약은 (T − T_기준) / (T_wall − T_기준) 형태가 맞습니다.")
        elif temp_mode == "nondimensional":
            print(f"  {CHECK} 최댓값이 1.0 이 아닙니다 ({tmax:.4f}). 규약이 다를 수 있으니 "
                  f"BubbleML 문서를 확인하세요.")

        # ---------- 4. 좌표 — 미터인가 비율인가 ----------
        _line("4. 좌표 — 미터인가 비율인가? 비율이면 기준 길이는?")
        for which in ("x", "y"):
            if which not in h5:
                print(f"  {CHECK} '{which}' 좌표가 파일에 없습니다. 격자 간격을 설정으로 줘야 합니다.")
                continue
            c = _axis(h5[which][()], which)
            if c.size < 2:
                continue
            d = float(np.mean(np.diff(c)))
            span = float(c.max() - c.min())
            print(f"  {which}: 범위 {c.min():.5g} ~ {c.max():.5g} (폭 {span:.5g}), 간격 {d:.6g}")

            if span < 0.1:
                print(f"     {OK} 폭이 0.1 미만 → 이미 **미터** 단위로 보입니다 (폭 {span * 1e3:.2f} mm).")
                print(f"        격자 간격 = {d * 1e6:.2f} µm")
                if expected_grid_um:
                    ratio = d * 1e6 / expected_grid_um
                    tag = OK if 0.9 < ratio < 1.1 else CHECK
                    print(f"        {tag} 알려진 격자 {expected_grid_um} µm 와 비교 → {ratio:.2f}배")
            else:
                print(f"     {OK} 폭이 {span:.3g} → **비율(무차원)** 입니다. 미터가 아닙니다.")
                if expected_grid_um:
                    implied = expected_grid_um * 1e-6 / d
                    cap = fluid.capillary_length
                    print(f"\n        ▶ 기준 길이 역산:")
                    print(f"          파일의 간격 {d:.6g} 이 실제로 {expected_grid_um} µm 라면,")
                    print(f"          기준 길이 = {expected_grid_um} µm ÷ {d:.6g} "
                          f"= {implied * 1e3:.4f} mm")
                    print(f"          {fluid.name} 의 모세관 길이 = {cap * 1e3:.4f} mm")
                    r = implied / cap
                    if 0.9 < r < 1.1:
                        print(f"          {OK} 거의 일치({r:.3f}배) → 기준 길이는 모세관 길이입니다.")
                        print(f"             configs 에 nondim.length_scale_m: {cap:.6g}")
                    else:
                        print(f"          {CHECK} {r:.2f}배 차이 → 기준이 모세관 길이가 아닙니다.")
                        print(f"             일단 nondim.length_scale_m: {implied:.6g} 로 두고,")
                        print(f"             par 파일에서 진짜 기준 길이를 찾으세요.")
                else:
                    print(f"        → --expected-grid-um 22.7 을 주면 기준 길이를 역산해 줍니다.")

        # ---------- 5. dfun 부호 ----------
        _line("5. dfun — 어느 쪽이 액체인가?")
        pos = float(np.mean(phi > 0))
        print(f"  φ > 0 인 셀 비율: {pos:.1%}   (범위 {phi.min():.4g} ~ {phi.max():.4g})")
        if pos > 0.6:
            print(f"  {OK} 대부분이 양수 → 액체가 양수(φ>0)일 가능성이 높습니다 (기포는 소수니까).")
            print(f"       → configs 에 phase.phi_positive_in: liquid  ← 현재 기본값")
        elif pos < 0.4:
            print(f"  {OK} 대부분이 음수 → **액체가 음수**입니다. 기본값을 바꿔야 합니다!")
            print(f"       → configs 에 phase.phi_positive_in: vapor")
        else:
            print(f"  {CHECK} 반반이라 판단이 안 섭니다. 한 프레임을 그림으로 그려 확인하세요.")

        # 온도와 교차검증: 증기(기포 안)는 벽 근처에서 액체보다 덜 뜨거운 경향
        wall_band = T[:, :3, :].ravel()
        phi_band = phi[:, :3, :].ravel()
        if phi_band.std() > 0:
            t_pos = wall_band[phi_band > 0].mean() if (phi_band > 0).any() else np.nan
            t_neg = wall_band[phi_band < 0].mean() if (phi_band < 0).any() else np.nan
            print(f"  (교차확인) 벽 근처 3줄 평균 온도 — φ>0 쪽 {t_pos:.4f}, φ<0 쪽 {t_neg:.4f}")

        # ---------- 6. runtime params ----------
        _line("6. 시뮬레이션 설정값(runtime params) — 물성이 들어 있나?")
        found = False
        for key in ("real-runtime-params", "int-runtime-params", "string-runtime-params",
                    "real runtime parameters", "integer runtime parameters"):
            if key not in h5:
                continue
            found = True
            print(f"  [{key}]")
            try:
                rows = h5[key][()]
                for row in rows[:60]:
                    name, value = row[0], row[1]
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", "ignore")
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", "ignore")
                    print(f"    {str(name).strip():32s} = {value}")
                if len(rows) > 60:
                    print(f"    ... (총 {len(rows)}개)")
            except Exception as exc:
                print(f"    (파싱 실패: {exc})")
        if not found:
            print(f"  {CHECK} runtime params 가 파일에 없습니다.")
            print(f"       → 물성(k_l, k_v, T_sat, 압력)과 기준 길이를 **BubbleML 논문이나**")
            print(f"         **Flash-X par 파일**에서 찾아 configs/fluids.yaml 에 적어야 합니다.")
            print(f"       → 지금 코드는 FC-72 문헌 대표값을 쓰고 있습니다 (실측값 아님).")

        # ---------- 마무리 ----------
        _line("요약 — configs 에 적을 것")
        print(f"  nondim.temperature_mode : {temp_mode}")
        print(f"  phase.phi_positive_in   : {'liquid' if pos > 0.5 else 'vapor'}")
        print(f"  nondim.case_table 에 이 파일의 T_wall_C / T_sat_C / T_bulk_C 를 적으세요.")
        if b_spread < 1e-6 and temp_mode != "nondimensional":
            print(f"  → 벽 온도는 데이터에서 직접 읽을 수 있습니다: {float(bottom.mean()):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="BubbleML HDF5 점검 도구")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--expected-grid-um", type=float, default=22.7,
                    help="문헌에 적힌 격자 크기 [µm]. 기준 길이 역산에 씁니다 (기본 22.7)")
    ap.add_argument("--fluid", default="FC-72")
    args = ap.parse_args()

    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"{BAD} 파일 없음: {p}")
            continue
        try:
            inspect(p, args.expected_grid_um, args.fluid)
        except Exception as exc:
            print(f"{BAD} {p.name} 점검 중 오류: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
