"""level-set φ(dfun) → 상(phase) 지시함수와 혼합 물성.

'거친 입력 검증' 9/21 노트의 1번 문제 해결안:
    섞인 셀을 결측 처리하지 말고, 열전도도를 dfun 비율대로 섞어서 전부 계산한다.

혼합 규칙은 두 가지를 모두 제공한다.
  - ``arithmetic`` : k = k_v + (k_l - k_v) * H   (노트에 적힌 "비율로 섞은 값", 기본값)
  - ``harmonic``   : 1/k = H/k_l + (1-H)/k_v     (벽에 수직한 직렬 전도의 물리적 형태)
둘의 차이는 진단 리포트에 함께 기록된다 (민감도 항목).
"""

from __future__ import annotations

import numpy as np

__all__ = ["smoothed_heaviside", "liquid_fraction", "mix_conductivity"]


def smoothed_heaviside(phi: np.ndarray, eps: float) -> np.ndarray:
    """정규화된 smoothed Heaviside H_eps(phi) ∈ [0, 1].

    phi > 0 인 쪽이 1이 된다 (호출부에서 액체가 양수가 되도록 부호를 맞춰 넘길 것).
    eps -> 0 이면 계단함수. 통상 eps = 1.5 * Δ 를 쓴다.
    """
    phi = np.asarray(phi, dtype=float)
    if eps <= 0:
        return (phi > 0).astype(float)
    h = np.empty_like(phi)
    inner = np.abs(phi) <= eps
    h[phi > eps] = 1.0
    h[phi < -eps] = 0.0
    p = phi[inner] / eps
    h[inner] = 0.5 * (1.0 + p + np.sin(np.pi * p) / np.pi)
    return h


def liquid_fraction(phi: np.ndarray, eps: float, positive_in: str = "liquid") -> np.ndarray:
    """dfun 에서 액체 분율 alpha_l ∈ [0, 1] 을 만든다."""
    phi = np.asarray(phi, dtype=float)
    if positive_in == "liquid":
        signed = phi
    elif positive_in == "vapor":
        signed = -phi
    else:
        raise ValueError(f"positive_in 은 'liquid' 또는 'vapor' — 받은 값: {positive_in}")
    return smoothed_heaviside(signed, eps)


def mix_conductivity(
    alpha_l: np.ndarray, k_liquid: float, k_vapor: float, rule: str = "arithmetic"
) -> np.ndarray:
    """액체 분율로 열전도도를 혼합한다."""
    alpha_l = np.clip(np.asarray(alpha_l, dtype=float), 0.0, 1.0)
    if rule == "arithmetic":
        return k_vapor + (k_liquid - k_vapor) * alpha_l
    if rule == "harmonic":
        return 1.0 / (alpha_l / k_liquid + (1.0 - alpha_l) / k_vapor)
    raise ValueError(f"알 수 없는 혼합 규칙 {rule!r} (arithmetic | harmonic)")
