#!/usr/bin/env python3
"""Zimmermann 외 (2020) FC-72 매끈한 구리 풀비등 원자료 → 실측 비등곡선 CSV.

원자료: B2SHARE "Data for the publication: Influence of system pressure on pool boiling
regimes on a microstructured surface compared to a smooth surface by Zimmermann et al."
    https://b2share.eudat.eu/records/4znr8-k5h80   (CC BY-NC 4.0)
    zip md5 a7578ba95625855f7e92c3fcd158774d → "measurement and evaluated data.xls"
논문: Exp. Heat Transfer 33(4):318–334, doi:10.1080/08916152.2019.1635228

이 스크립트는 'smooth heater' 시트만 읽는다. 시트 안에 압력별 블록이 헤더 줄로
나뉘어 있고, 각 줄은 한 번의 정상 측정이다. 쓰는 열은 원자료가 계산해 둔 값 그대로다.
    pressure sensor [bar], q el [W/m²] (= V·I/A, 열손실 보정 없음),
    Tsat [°C], DT [K] (= 표면 온도 − Tsat, 표면 온도는 2.85 mm 아래 열전대에서 외삽)
블록 안에서 q 가 최대가 되는 줄까지를 상승(asc), 그 뒤를 하강(desc) 측정으로 둔다.

    pip install xlrd   # .xls 읽기용
    python scripts/convert_zimmermann.py "measurement and evaluated data.xls" \
        data/experiments/zimmermann2020_fc72_smooth.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEADER = """\
# Zimmermann, Heinz, Sielaff, Gambaryan-Roisman, Stephan (2020) Exp. Heat Transfer 33(4):318-334
# doi:10.1080/08916152.2019.1635228 ; data: https://b2share.eudat.eu/records/4znr8-k5h80 (CC BY-NC 4.0)
# FC-72, saturated pool boiling, smooth copper cylinder top face D=35 mm (A=9.62e-4 m2), heated from below.
# Orientation not stated in the source (upward-facing inferred). Roughness not in the dataset.
# q = electrical input V*I/A, no heat-loss correction. dT = extrapolated surface temperature - Tsat.
# Converted with scripts/convert_zimmermann.py from sheet 'smooth heater' (values unchanged; rows split into asc/desc).
"""

COLUMNS = ["source", "curve_id", "fluid", "surface", "orientation_deg", "branch",
           "p_Pa", "T_sat_C", "q_W_m2", "dT_K"]


def convert(xls: Path) -> pd.DataFrame:
    sheet = pd.read_excel(xls, sheet_name="smooth heater", header=None)
    blocks: list[tuple[list, list]] = []
    for _, row in sheet.iterrows():
        if isinstance(row[0], str) and "Thermostat" in row[0]:
            blocks.append((row.tolist(), []))
        elif blocks:
            blocks[-1][1].append(row.tolist())
    if not blocks:
        raise ValueError("'smooth heater' 시트에서 압력 블록 헤더를 찾지 못했습니다")

    out = []
    for header, rows in blocks:
        col = {name: i for i, name in enumerate(header) if isinstance(name, str)}
        df = pd.DataFrame(rows)
        p_bar = pd.to_numeric(df[col["pressure sensor"]], errors="coerce").to_numpy()
        q = pd.to_numeric(df[col["q el"]], errors="coerce").to_numpy()
        tsat = pd.to_numeric(df[col["Tsat"]], errors="coerce").to_numpy()
        dT = pd.to_numeric(df[col["DT"]], errors="coerce").to_numpy()
        ok = np.isfinite(p_bar) & np.isfinite(q) & np.isfinite(dT)
        p_bar, q, tsat, dT = p_bar[ok], q[ok], tsat[ok], dT[ok]
        top = int(np.argmax(q))
        label = f"Z2020_{np.mean(p_bar):.2f}bar"
        for i in range(q.size):
            out.append({
                "source": "Zimmermann2020",
                "curve_id": label,
                "fluid": "FC-72",
                "surface": "smooth Cu D35mm",
                "orientation_deg": 0,
                "branch": "asc" if i <= top else "desc",
                "p_Pa": round(float(p_bar[i]) * 1e5, 1),
                "T_sat_C": round(float(tsat[i]), 3),
                "q_W_m2": round(float(q[i]), 2),
                "dT_K": round(float(dT[i]), 4),
            })
    return pd.DataFrame(out, columns=COLUMNS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("xls", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    frame = convert(args.xls)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write(HEADER)
        frame.to_csv(fh, index=False)
    summary = frame.groupby("curve_id").agg(n=("q_W_m2", "size"), p_bar=("p_Pa", lambda s: s.mean() / 1e5),
                                            q_max=("q_W_m2", "max"))
    print(summary.to_string())
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
