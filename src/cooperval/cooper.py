"""Cooper (1984) 풀비등 상관식 — 정방향과 역산.

    h = 55 * p_r^(0.12 - 0.2 log10 R_p) * (-log10 p_r)^(-0.55) * M^(-0.5) * q''^0.67
    q'' = h * ΔT_sup   ==>   ΔT_sup = q''^0.33 / K,
    K = 55 * p_r^(0.12 - 0.2 log10 R_p) * (-log10 p_r)^(-0.55) * M^(-0.5)

단위: q'' [W/m^2], h [W/m^2/K], M [g/mol], R_p [µm], ΔT [K].

R_p 규칙 (9/21 노트의 '함정'):
    시뮬레이션에는 표면조도 정의가 없으므로 R_p 는 사실상 자유 파라미터다.
    맞추려고 튜닝하면 '유체별 상수 없음' 주장이 그 자리에서 깨진다.
    따라서 **R_p = 1 µm 고정**이고, 코드가 이를 강제한다(``allow_tuning=True`` 명시적
    허용이 없으면 1.0 이외의 값은 예외). 진단용으로 '만약 튜닝했다면 얼마였을까'를
    역산해 리포트에만 남긴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fluids import Fluid

__all__ = [
    "RP_FIXED_UM",
    "COOPER_Q_EXPONENT",
    "DELTA_T_EXPONENT",
    "CooperModel",
    "implied_roughness_um",
]

RP_FIXED_UM = 1.0
COOPER_Q_EXPONENT = 0.67          # h ∝ q''^0.67
DELTA_T_EXPONENT = 1.0 - COOPER_Q_EXPONENT   # ΔT ∝ q''^0.33  ← log-log 게이트의 기준 기울기


@dataclass(frozen=True)
class CooperModel:
    """한 유체/압력/조도 조합에 대해 고정된 Cooper 식."""

    fluid: Fluid
    roughness_um: float = RP_FIXED_UM
    allow_tuning: bool = False

    def __post_init__(self) -> None:
        if not self.allow_tuning and not math.isclose(self.roughness_um, RP_FIXED_UM):
            raise ValueError(
                f"R_p 는 {RP_FIXED_UM} µm 고정입니다 (받은 값 {self.roughness_um}). "
                "튜닝은 '유체별 상수 없음' 주장을 깨뜨립니다. 의도적 민감도 분석이라면 "
                "allow_tuning=True 를 명시하세요."
            )
        pr = self.fluid.reduced_pressure
        if not 0.0 < pr < 1.0:
            raise ValueError(f"환산압력 p_r = {pr} 이 (0,1) 밖입니다 — 압력/임계압 확인 필요")

    # ------------------------------------------------------------------
    @property
    def prefactor(self) -> float:
        """K — q''^0.67 앞에 붙는 상수 덩어리.

        문헌 산포 ±30~40% 는 대부분 이 K 에 들어 있고, ΔT 로 **감쇠 없이** 전달된다.
        """
        pr = self.fluid.reduced_pressure
        m = self.fluid.molar_mass_g_mol
        exponent = 0.12 - 0.2 * math.log10(self.roughness_um)
        return (
            55.0
            * pr**exponent
            * (-math.log10(pr)) ** (-0.55)
            * m**-0.5
        )

    # --- 정방향 -------------------------------------------------------
    def h_from_q(self, q_flux):
        """q'' → 열전달계수 h."""
        q = np.asarray(q_flux, dtype=float)
        return self.prefactor * np.power(q, COOPER_Q_EXPONENT)

    def q_from_delta_t(self, delta_t):
        """ΔT_sup → q''  (정방향; 비등곡선을 그릴 때 사용)."""
        dt = np.asarray(delta_t, dtype=float)
        return np.power(self.prefactor * dt, 1.0 / DELTA_T_EXPONENT)

    # --- 역산 (이 게이트의 본체) ---------------------------------------
    def delta_t_from_q(self, q_flux):
        """q'' → 예측 벽 과열도 ΔT_sup = q''^0.33 / K."""
        q = np.asarray(q_flux, dtype=float)
        out = np.full(q.shape, np.nan, dtype=float)
        ok = q > 0
        out[ok] = np.power(q[ok], DELTA_T_EXPONENT) / self.prefactor
        return out if out.ndim else float(out)

    def h_from_delta_t(self, delta_t):
        dt = np.asarray(delta_t, dtype=float)
        return self.q_from_delta_t(dt) / dt

    def describe(self) -> dict:
        return {
            "fluid": self.fluid.name,
            "molar_mass_g_mol": self.fluid.molar_mass_g_mol,
            "p_operating_Pa": self.fluid.p_operating_Pa,
            "p_critical_Pa": self.fluid.p_critical_Pa,
            "reduced_pressure": self.fluid.reduced_pressure,
            "roughness_um": self.roughness_um,
            "roughness_policy": "고정 (튜닝 금지)" if not self.allow_tuning else "튜닝 허용됨",
            "prefactor_K": self.prefactor,
            "q_exponent": COOPER_Q_EXPONENT,
            "delta_t_exponent": DELTA_T_EXPONENT,
        }


def implied_roughness_um(model: CooperModel, q_flux: float, delta_t_truth: float) -> float:
    """진단 전용 — 이 한 점을 정확히 맞추려면 R_p 가 얼마여야 했나.

    **절대 예측에 쓰지 말 것.** 리포트에만 남겨서 "R_p 를 튜닝했다면 물리적으로
    말이 되는 범위(0.1~10 µm)였는가, 아니면 터무니없었는가"를 보여주는 용도다.
    """
    if q_flux <= 0 or delta_t_truth <= 0:
        return float("nan")
    k_required = q_flux**DELTA_T_EXPONENT / delta_t_truth
    pr = model.fluid.reduced_pressure
    base = 55.0 * (-math.log10(pr)) ** (-0.55) * model.fluid.molar_mass_g_mol**-0.5
    # k_required = base * pr^(0.12 - 0.2 log10 Rp)
    ratio = k_required / base
    if ratio <= 0:
        return float("nan")
    exponent = math.log(ratio) / math.log(pr)          # = 0.12 - 0.2 log10 Rp
    return 10.0 ** ((0.12 - exponent) / 0.2)
