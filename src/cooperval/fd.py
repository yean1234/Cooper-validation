"""유한차분 가중치.

벽(경계면)에서의 온도 기울기를 1/2/3차 정확도로 계산하기 위한 일방향(one-sided)
차분 가중치를 Fornberg 알고리즘으로 생성한다.

격자가 셀 중심(cell-centred)이므로 벽은 y=0, 첫 셀 중심은 y=Δ/2, 그 다음이 3Δ/2 …
처럼 **비균등 간격**이 된다. 교과서의 균등 간격 계수를 쓰면 안 되므로
임의 노드 위치를 받는 일반식을 쓴다.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fd_weights", "wall_gradient_weights", "STENCIL_ORDERS"]

# 차수 → 벽에서부터 쓰는 셀 개수 (벽 값 T_w 는 항상 포함)
STENCIL_ORDERS = {1: 1, 2: 2, 3: 3}


def fd_weights(z: float, nodes, max_deriv: int = 1) -> np.ndarray:
    """Fornberg (1988) 유한차분 가중치.

    Parameters
    ----------
    z : 미분을 평가할 위치 (여기서는 벽, z=0)
    nodes : 값을 알고 있는 노드 좌표 배열
    max_deriv : 최대 미분 차수

    Returns
    -------
    (len(nodes), max_deriv+1) 배열. ``c[:, m]`` 이 m차 미분 가중치.
    """
    x = np.asarray(nodes, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("nodes 가 비어 있습니다")
    if n <= max_deriv:
        raise ValueError(f"{max_deriv}차 미분에는 노드가 최소 {max_deriv + 1}개 필요합니다")

    c = np.zeros((n, max_deriv + 1))
    c1 = 1.0
    c4 = x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, max_deriv)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


def wall_gradient_weights(dy: float, order: int) -> tuple[float, np.ndarray]:
    """벽에서의 dT/dy 에 대한 (벽 값 가중치, 셀 값 가중치들).

    노드 배치: y = 0 (벽, Dirichlet T_w), Δ/2, 3Δ/2, 5Δ/2, …

    Returns
    -------
    (w_wall, w_cells) — ``dT/dy|_wall = w_wall * T_w + sum(w_cells * T_cells)``

    Examples
    --------
    2차(3점)의 해석해는 (-8/3, 3, -1/3)/Δ 이다.
    """
    if order not in STENCIL_ORDERS:
        raise ValueError(f"지원하지 않는 차수 {order}. 가능: {sorted(STENCIL_ORDERS)}")
    n_cells = STENCIL_ORDERS[order]
    nodes = np.concatenate(([0.0], (np.arange(n_cells) + 0.5) * dy))
    w = fd_weights(0.0, nodes, max_deriv=1)[:, 1]
    return float(w[0]), w[1:]
