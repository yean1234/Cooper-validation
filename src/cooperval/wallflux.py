"""벽 열유속 q'' 추출 — 이 파이프라인의 '정답 자(ruler)'.

핵심 규칙 (9/20 '정답 기포장' + 9/21 '거친 입력 검증' 노트)
  1. **섞인 셀을 빼지 않는다.** 기포 뿌리가 열이 가장 많이 빠지는 자리라서, 빼면
     오차가 우연히가 아니라 *항상 낮은 쪽으로* 생긴다. 대신 열전도도를 dfun 비율로
     섞어 쓴다.
  2. **차분 차수를 하나로 못 박는다.** 1차/2차가 20% 차이 나는 상태에서는 남의 예측을
     채점할 수 없다. 기준은 2차(Flash-X 가 공간 2차 정확도 스킴이므로)이고,
     3차까지 계산해서 |2차-3차| << |1차-2차| 인지(수렴했는지)를 매 실행마다 확인한다.

푸리에 법칙:  q'' = -k dT/dy|_wall   (벽에서 유체 쪽으로 빠져나가는 열이 +)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .bubbleml import CaseData
from .fd import wall_gradient_weights
from .fluids import Fluid
from .heaviside import liquid_fraction, mix_conductivity

log = logging.getLogger(__name__)

__all__ = ["WallFluxResult", "compute_wall_flux", "convergence_audit"]

MIXED_CELL_LO = 0.02
MIXED_CELL_HI = 0.98


@dataclass
class WallFluxResult:
    case_id: str
    q: dict[int, np.ndarray]          # 차분차수 -> q''(n_t, n_x) [W/m^2]
    alpha_liquid: np.ndarray          # 벽 인접 셀의 액체 분율 (n_t, n_x)
    heater_mask: np.ndarray           # (n_x,) bool
    x: np.ndarray                     # (n_x,) [m]
    dy: float
    order: int                        # 기준 차수
    diagnostics: dict = field(default_factory=dict)

    @property
    def q_reference(self) -> np.ndarray:
        return self.q[self.order]

    def heater_mean(self, order: int | None = None) -> float:
        """히터 전체 + 시간 평균 q'' — 거친 입력의 최종 숫자."""
        q = self.q[order or self.order][:, self.heater_mask]
        return float(np.nanmean(q))

    def heater_profile(self, order: int | None = None) -> np.ndarray:
        """시간 평균한 q''(x) — 거친 격자로 뭉개기 전의 프로파일."""
        q = self.q[order or self.order][:, self.heater_mask]
        return np.nanmean(q, axis=0)

    @property
    def vapor_fraction_wall(self) -> float:
        """벽에 닿은 증기 분율 (시간·히터 평균). 거친 입력의 두 번째 성분."""
        a = self.alpha_liquid[:, self.heater_mask]
        return float(1.0 - np.nanmean(a))


def _orient_wall(case: CaseData, wall: str) -> np.ndarray:
    """벽이 y 인덱스 0 쪽에 오도록 (n_t, n_y, n_x) 를 정렬해 반환용 슬라이스를 만든다."""
    if wall == "bottom":
        return slice(None, None, 1)
    if wall == "top":
        return slice(None, None, -1)
    raise ValueError(f"geometry.wall 은 'bottom' 또는 'top' — 받은 값 {wall!r}")


def _heater_mask(x: np.ndarray, x_range) -> np.ndarray:
    if not x_range:
        return np.ones_like(x, dtype=bool)
    lo, hi = float(x_range[0]), float(x_range[1])
    return (x >= lo) & (x <= hi)


def compute_wall_flux(case: CaseData, cfg: Config, fluid: Fluid) -> WallFluxResult:
    wall = cfg.get("geometry.wall", "bottom")
    flip = _orient_wall(case, wall)
    T = case.temperature[:, flip, :]        # (n_t, n_y, n_x), 인덱스 0 이 벽에 가장 가까움
    phi = case.dfun[:, flip, :]

    eps = float(cfg.get("phase.eps_cells", 1.5)) * case.dy
    alpha_all = liquid_fraction(phi, eps, cfg.get("phase.phi_positive_in", "liquid"))
    alpha_wall = alpha_all[:, 0, :]         # 벽 인접 셀의 액체 분율

    k_wall = mix_conductivity(
        alpha_wall, fluid.k_liquid, fluid.k_vapor, cfg.get("phase.mixing_rule", "arithmetic")
    )

    T_w = case.meta.T_wall_K                # Dirichlet 벽 온도
    orders = sorted(set(list(cfg.get("wall_flux.audit_orders", [1, 2, 3]))
                        + [int(cfg.get("wall_flux.order", 2))]))

    q: dict[int, np.ndarray] = {}
    for order in orders:
        w_wall, w_cells = wall_gradient_weights(case.dy, order)
        n_cells = w_cells.size
        if T.shape[1] < n_cells:
            raise ValueError(
                f"[{case.meta.case_id}] {order}차 차분에 셀 {n_cells}개가 필요한데 "
                f"y 방향 셀이 {T.shape[1]}개뿐입니다."
            )
        grad = w_wall * T_w + np.tensordot(w_cells, T[:, :n_cells, :], axes=([0], [1]))
        q[order] = -k_wall * grad           # (n_t, n_x)

    # 섞인 셀을 빼는 (편향된) 옛 방식을 재현하고 싶을 때만 마스크를 건다.
    if not cfg.get("wall_flux.include_mixed_cells", True):
        mixed = (alpha_wall > MIXED_CELL_LO) & (alpha_wall < MIXED_CELL_HI)
        log.warning(
            "[%s] include_mixed_cells=False — 섞인 셀 %.1f%% 를 결측 처리합니다. "
            "q'' 가 체계적으로 낮게 나옵니다(편향 재현용 모드).",
            case.meta.case_id, 100.0 * mixed.mean(),
        )
        for order in q:
            q[order] = np.where(mixed, np.nan, q[order])

    heater = _heater_mask(case.x, cfg.get("geometry.heater_x_range_m"))
    ref_order = int(cfg.get("wall_flux.order", 2))

    result = WallFluxResult(
        case_id=case.meta.case_id,
        q=q,
        alpha_liquid=alpha_wall,
        heater_mask=heater,
        x=case.x,
        dy=case.dy,
        order=ref_order,
        diagnostics={
            "mixing_rule": cfg.get("phase.mixing_rule"),
            "eps_m": eps,
            "mixed_cell_fraction": float(
                np.mean((alpha_wall > MIXED_CELL_LO) & (alpha_wall < MIXED_CELL_HI))
            ),
            "heater_cells": int(heater.sum()),
            "wall_temperature_K": T_w,
        },
    )
    result.diagnostics.update(convergence_audit(result))

    # 혼합 규칙 민감도: arithmetic vs harmonic 을 같은 자리에서 비교해 둔다.
    other_rule = "harmonic" if cfg.get("phase.mixing_rule") == "arithmetic" else "arithmetic"
    k_other = mix_conductivity(alpha_wall, fluid.k_liquid, fluid.k_vapor, other_rule)
    w_wall, w_cells = wall_gradient_weights(case.dy, ref_order)
    grad_ref = w_wall * T_w + np.tensordot(w_cells, T[:, : w_cells.size, :], axes=([0], [1]))
    q_other = float(np.nanmean((-k_other * grad_ref)[:, heater]))
    q_ref = result.heater_mean()
    result.diagnostics["mixing_rule_sensitivity"] = {
        "alternative_rule": other_rule,
        "q_alternative_W_m2": q_other,
        "relative_difference": abs(q_other - q_ref) / abs(q_ref) if q_ref else float("nan"),
    }
    return result


def convergence_audit(result: WallFluxResult) -> dict:
    """1차/2차/3차 차이로 '2차가 이미 수렴했는지'를 판정한다.

    |2차-3차| 가 |1차-2차| 보다 훨씬 작으면 2차는 수렴한 것 → 기준으로 써도 된다.
    둘 다 크면 격자가 경계층을 못 잡고 있다는 뜻이고, 이건 차분 차수 문제가 아니라
    더 큰 문제다 (게이트 결과를 해석하면 안 됨).
    """
    means = {o: float(np.nanmean(q[:, result.heater_mask])) for o, q in result.q.items()}
    audit: dict = {"heater_mean_q_by_order": means}
    if {1, 2, 3} <= set(means):
        d12 = abs(means[2] - means[1]) / abs(means[2])
        d23 = abs(means[3] - means[2]) / abs(means[2])
        audit.update(
            {
                "rel_diff_1st_2nd": d12,
                "rel_diff_2nd_3rd": d23,
                "converged": bool(d23 < 0.25 * d12 or d23 < 0.02),
                "verdict": (
                    "2차 수렴 — 기준으로 사용 가능"
                    if (d23 < 0.25 * d12 or d23 < 0.02)
                    else "미수렴 — 격자가 경계층을 못 잡고 있을 가능성. 게이트 해석 주의"
                ),
            }
        )
    return audit
