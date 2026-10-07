"""정답 기포장 인계 표(``results/ground_truth_final.csv``) 로더.

정답 기포장 레포가 열유속 정의를 확정했다 (보고서 4~5쪽):
    q'' = k_mix · g_wall,   g_wall = 2(T_w − T_1)/Δy   (Flash-X 경계와 같은 1차 기울기)
    k_mix 는 phase-averaged — 산술이 기본, 조화가 동등 후보.
그래서 Cooper 게이트가 HDF5 에서 q'' 를 따로 다시 뽑을 이유가 없다. 이 모듈은 그 확정값을
그대로 읽어 '정답 q'' → Cooper → 예측 ΔT ↔ 정답 ΔT' 비교에 넘긴다.

같이 읽는 것: 조건별 **처방 사이트 수**. BubbleML 은 사이트 수를 입력으로 정하므로
비등곡선 기울기가 그 처방에 묶인다 — 게이트가 기울기를 판정에서 뺄지 정하는 근거다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["QDefinition", "Q_DEFINITIONS", "GroundTruthCondition", "load_ground_truth"]

DELTA_T_COL = "input_dT_sup_K"
TWALL_COL = "input_Twall_C"
SITES_COL = "input_n_sites_prescribed"
ACTIVE_SITES_COL = "confirmed_n_sites_active_2d"
CASE_ID_COLS = ("condition", "case_id", "case", "name")


@dataclass(frozen=True)
class QDefinition:
    column: str
    role: str          # 기본 | 동등 후보 | 참고
    description: str


# 보고서 5쪽의 네 정의. 판정은 '기본'으로, '동등 후보'는 판정이 갈리는지 보는 민감도로 쓴다.
# '참고' 둘은 표에만 남긴다 — 결과가 좋아지는 정의를 골라 쓰지 않는다.
Q_DEFINITIONS: dict[str, QDefinition] = {
    "arith": QDefinition("confirmed_q_mix_arith_2d_W_m2", "기본",
                         "1차 벽 기울기 × 산술 phase-averaged k"),
    "harm": QDefinition("confirmed_q_mix_harm_2d_W_m2", "동등 후보",
                        "1차 벽 기울기 × 조화 phase-averaged k"),
    "kl": QDefinition("confirmed_q_kl_all_2d_W_m2", "참고", "논문 본문 k_l 표기 변형"),
    "legacy": QDefinition("confirmed_q_legacy_2d_W_m2", "참고", "저자 저장소 legacy 환산"),
}


@dataclass
class GroundTruthCondition:
    case_id: str
    delta_t_sup_K: float
    q_W_m2: dict[str, float]                  # q 정의 키 -> 값
    n_sites_prescribed: float = float("nan")
    n_sites_active: float = float("nan")
    T_wall_C: float = float("nan")


def _orient(df: pd.DataFrame) -> pd.DataFrame:
    """조건이 행이 되게 맞춘다. 보고서 표처럼 조건이 열이면 뒤집는다."""
    if DELTA_T_COL in df.columns:
        return df
    first = df.columns[0]
    labels = df[first].astype(str).str.strip()
    if DELTA_T_COL in set(labels):
        out = df.set_index(labels).drop(columns=[first]).T
        out.index.name = "condition"
        return out.reset_index()
    raise KeyError(
        f"정답 CSV 에 {DELTA_T_COL!r} 가 없습니다 (행·열 모두 확인). "
        f"있는 컬럼: {list(df.columns)[:20]}"
    )


def _case_id(row: pd.Series, i: int) -> str:
    for col in CASE_ID_COLS:
        if col in row.index and str(row[col]).strip() not in ("", "nan"):
            return str(row[col]).strip()
    twall = row.get(TWALL_COL)
    if twall is not None and np.isfinite(pd.to_numeric(twall, errors="coerce")):
        return f"Twall_{float(twall):g}"
    return f"case{i}"


def _num(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return float("nan")
    return float(pd.to_numeric(row[col], errors="coerce"))


def load_ground_truth(path: str | Path) -> list[GroundTruthCondition]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"정답 CSV 가 없습니다: {path}. 정답 기포장 레포의 results/ground_truth_final.csv 를 "
            "받아 두거나 data/ground_truth/ground_truth_final_snapshot.csv 를 쓰세요."
        )
    df = _orient(pd.read_csv(path, comment="#"))

    available = {k: d for k, d in Q_DEFINITIONS.items() if d.column in df.columns}
    if not available:
        raise KeyError(
            "정답 CSV 에 열유속 컬럼이 하나도 없습니다. 찾은 이름: "
            f"{[d.column for d in Q_DEFINITIONS.values()]}"
        )

    conditions: list[GroundTruthCondition] = []
    for i, (_, row) in enumerate(df.iterrows()):
        case_id = _case_id(row, i)
        dt = _num(row, DELTA_T_COL)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"[{case_id}] {DELTA_T_COL} 가 비었거나 0 이하입니다: {row.get(DELTA_T_COL)}")
        q = {k: _num(row, d.column) for k, d in available.items()}
        bad = [k for k, v in q.items() if not np.isfinite(v) or v <= 0]
        if bad:
            # 조용히 빼면 조건 수와 평균이 바뀐다 — 멈춘다.
            raise ValueError(f"[{case_id}] 열유속 값이 비었거나 0 이하: {bad}")
        conditions.append(
            GroundTruthCondition(
                case_id=case_id,
                delta_t_sup_K=dt,
                q_W_m2=q,
                n_sites_prescribed=_num(row, SITES_COL),
                n_sites_active=_num(row, ACTIVE_SITES_COL),
                T_wall_C=_num(row, TWALL_COL),
            )
        )
    conditions.sort(key=lambda c: c.delta_t_sup_K)
    return conditions
