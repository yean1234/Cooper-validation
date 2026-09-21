import numpy as np
import pytest

from cooperval.fd import fd_weights, wall_gradient_weights


def test_second_order_weights_match_analytic():
    """벽(y=0)과 셀 중심 Δ/2, 3Δ/2 에 대한 해석해는 (-8/3, 3, -1/3)/Δ."""
    dy = 2.0
    w_wall, w_cells = wall_gradient_weights(dy, order=2)
    assert w_wall == pytest.approx(-8 / 3 / dy)
    assert w_cells == pytest.approx(np.array([3.0, -1 / 3]) / dy)


def test_weights_sum_to_zero():
    """상수 온도장의 기울기는 0 이어야 한다 — 가중치 합이 0."""
    for order in (1, 2, 3):
        w_wall, w_cells = wall_gradient_weights(1.0, order)
        assert w_wall + w_cells.sum() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("order,degree", [(1, 1), (2, 2), (3, 3)])
def test_exact_for_polynomials_of_matching_degree(order, degree):
    """n차 스텐실은 n차 다항식의 기울기를 정확히 재현해야 한다."""
    dy = 0.37
    w_wall, w_cells = wall_gradient_weights(dy, order)
    y_cells = (np.arange(w_cells.size) + 0.5) * dy
    coeffs = np.array([0.3, -1.7, 0.9, 2.1][: degree + 1])

    def f(y):
        return sum(c * y**i for i, c in enumerate(coeffs))

    got = w_wall * f(0.0) + float(w_cells @ f(y_cells))
    assert got == pytest.approx(coeffs[1], rel=1e-9)


def test_fd_weights_rejects_too_few_nodes():
    with pytest.raises(ValueError):
        fd_weights(0.0, [0.0], max_deriv=1)
