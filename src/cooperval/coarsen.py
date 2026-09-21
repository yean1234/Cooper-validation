"""뭉개기(coarsening) — 원본 미세 격자 → 거친 입력.

여기는 평균만 한다. Cooper 는 다음 단계다.
거친 격자(기본 2 mm)가 '보는' 값은
  - q''  : 히터 면적 평균 열유속  (면적 평균이 물리적으로 맞는 축약)
  - α_v  : 벽에 닿은 증기 분율
이고, 두 값 모두 시간 평균까지 해서 **조건당 숫자 한 개**로 만든다
(9/20 노트: 순간값이 아니라 조건당 대표값 1개로 고정).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .wallflux import WallFluxResult

__all__ = ["CoarseInput", "coarsen"]


@dataclass
class CoarseInput:
    case_id: str
    x_coarse: np.ndarray          # (n_coarse,) 거친 셀 중심 [m]
    q_cells: np.ndarray           # (n_coarse,) 거친 셀별 q'' [W/m^2]
    vapor_fraction_cells: np.ndarray
    q_heater_mean: float          # 히터 전체 평균 q'' — 게이트에 쓰는 값
    vapor_fraction_heater: float
    block_size: int               # 거친 셀 하나에 들어간 미세 셀 개수
    dx_coarse_m: float
    dx_fine_m: float

    def to_rows(self) -> list[dict]:
        return [
            {
                "case_id": self.case_id,
                "coarse_cell": i,
                "x_m": float(x),
                "q_W_m2": float(q),
                "vapor_fraction": float(a),
            }
            for i, (x, q, a) in enumerate(
                zip(self.x_coarse, self.q_cells, self.vapor_fraction_cells)
            )
        ]


def _block_mean(values: np.ndarray, block: int) -> np.ndarray:
    """앞에서부터 block 개씩 평균. 남는 꼬리는 버린다(부분 블록은 대표성이 없음)."""
    n = (values.size // block) * block
    if n == 0:
        return np.array([float(np.nanmean(values))])
    return np.nanmean(values[:n].reshape(-1, block), axis=1)


def coarsen(result: WallFluxResult, cfg: Config) -> CoarseInput:
    dx_coarse = float(cfg.get("coarsen.dx_m", 2e-3))
    x_heater = result.x[result.heater_mask]
    dx_fine = float(np.mean(np.diff(x_heater))) if x_heater.size > 1 else dx_coarse

    block = max(1, int(round(dx_coarse / dx_fine)))
    if block > x_heater.size:
        block = x_heater.size

    q_profile = result.heater_profile()                       # 시간 평균 q''(x)
    alpha_profile = 1.0 - np.nanmean(result.alpha_liquid[:, result.heater_mask], axis=0)

    q_cells = _block_mean(q_profile, block)
    a_cells = _block_mean(alpha_profile, block)
    x_cells = _block_mean(x_heater, block)

    return CoarseInput(
        case_id=result.case_id,
        x_coarse=x_cells,
        q_cells=q_cells,
        vapor_fraction_cells=a_cells,
        q_heater_mean=result.heater_mean(),
        vapor_fraction_heater=result.vapor_fraction_wall,
        block_size=block,
        dx_coarse_m=block * dx_fine,
        dx_fine_m=dx_fine,
    )
