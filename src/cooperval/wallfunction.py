"""두 번째 경로 — 온도 벽함수(Kader, 1981). **10월 정식 게이트용, 기본 비활성.**

왜 두는가 (멘토 조언 9/9, 9/21 정리)
    Cooper 역산은 "열이 이만큼 나가니 경험상 벽은 이 정도 뜨거울 것"이라 **격자 크기를
    아예 안 본다.** 반면 우리 문제는 "굵은 격자가 보는 평균값에서 벽 온도 되찾기"이고,
    이건 CFD 업계가 수십 년 다뤄 표준 도구(온도 벽함수)가 이미 있는 문제다.
    두 경로를 나란히 비교하면 게이트가 '중단/진행'이 아니라 '무엇이 더 잘 복원하나'가
    되고, 실패해도 결론이 남는다.

주의 — 이건 원래 **액체 단상 난류**용이다. 비등에 그대로 쓰면 논란 여지가 있으므로
정답으로 놓지 말고 비교 대상으로만 쓴다.

현재 상태: 공식은 구현했지만 **데이터 배선은 안 돼 있다.** u_tau 가 필요하고 그건
velx/vely 로 벽 전단을 계산해야 나오는데, 로더가 아직 속도장을 읽지 않는다.
9/21 일정에서는 Cooper 한 경로만 돌리기로 했으므로 의도적으로 남겨 둔 구멍이다.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kader_t_plus", "delta_t_from_wall_function", "WallFunctionNotWired"]


class WallFunctionNotWired(NotImplementedError):
    """벽함수 경로는 10월 게이트 항목 — 아직 데이터에 연결되지 않았다."""


def kader_t_plus(y_plus, prandtl: float):
    """Kader 의 무차원 온도 T+ (점성층~로그층을 매끄럽게 잇는 blending)."""
    y_plus = np.asarray(y_plus, dtype=float)
    pr = float(prandtl)
    beta = (3.85 * pr ** (1.0 / 3.0) - 1.3) ** 2 + 2.12 * np.log(pr)
    gamma = 0.01 * (pr * y_plus) ** 4 / (1.0 + 5.0 * pr**3 * y_plus)
    return pr * y_plus * np.exp(-gamma) + (2.12 * np.log(np.maximum(y_plus, 1e-12)) + beta) * np.exp(
        -1.0 / np.maximum(gamma, 1e-12)
    )


def delta_t_from_wall_function(q_flux, T1, y1, u_tau, fluid, T_sat: float):
    """첫 셀 온도 T1 과 셀 중심 거리 y1 에서 벽 과열도를 되찾는다.

        T+ = (T_w - T1) * rho * c_p * u_tau / q''
        => T_w = T1 + q'' * T+(y+) / (rho * c_p * u_tau),   y+ = y1 * u_tau / nu
    """
    nu = fluid.mu_liquid / fluid.rho_liquid
    y_plus = np.asarray(y1, dtype=float) * u_tau / nu
    t_plus = kader_t_plus(y_plus, fluid.prandtl_liquid)
    T_wall = np.asarray(T1, dtype=float) + np.asarray(q_flux, dtype=float) * t_plus / (
        fluid.rho_liquid * fluid.cp_liquid * u_tau
    )
    return T_wall - T_sat
