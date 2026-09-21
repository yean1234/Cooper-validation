import numpy as np
import pytest

from cooperval.bubbleml import CaseData, CaseMeta
from cooperval.coarsen import coarsen
from cooperval.config import load_config
from cooperval.fluids import get_fluid
from cooperval.wallflux import compute_wall_flux

FLUID = get_fluid("FC-72")


def _exponential_case(q_target=2.0e5, delta_t=30.0, n_y=64, n_x=32, layer_cells=8.0,
                      vapor_columns=()):
    """해석해를 아는 벽 경계층: T = T_sat + ΔT exp(-y/δ), δ = k ΔT / q.

    이렇게 만들면 -k dT/dy|_wall 이 정확히 q_target 이므로, 추출값이 이 값에서
    벗어난 만큼이 곧 수치 오차다.
    """
    T_sat = FLUID.T_sat_K
    k_wall = np.full(n_x, FLUID.k_liquid)
    phi = np.full((1, n_y, n_x), 10.0)
    for col in vapor_columns:
        k_wall[col] = FLUID.k_vapor
        phi[0, :, col] = -10.0

    delta = k_wall * delta_t / q_target
    dy = float(delta.max() / layer_cells)
    y = (np.arange(n_y) + 0.5) * dy
    T = T_sat + delta_t * np.exp(-y[None, :, None] / delta[None, None, :])

    meta = CaseMeta("analytic", "<memory>", T_sat + delta_t, T_sat, T_sat)
    return CaseData(
        meta=meta, temperature=T, dfun=phi,
        x=(np.arange(n_x) + 0.5) * dy, y=y, dx=dy, dy=dy,
    )


def test_second_order_recovers_analytic_flux():
    q_target = 2.0e5
    case = _exponential_case(q_target=q_target, layer_cells=8.0)
    cfg = load_config(overrides={"nondim": {"temperature_mode": "kelvin"}})
    result = compute_wall_flux(case, cfg, FLUID)
    assert result.heater_mean() == pytest.approx(q_target, rel=0.01)


def test_higher_order_is_closer_to_truth():
    """1차 < 2차 < 3차 순으로 정답에 가까워야 한다 (9/20 노트의 수렴 확인)."""
    q_target = 2.0e5
    case = _exponential_case(q_target=q_target, layer_cells=3.0)
    cfg = load_config(overrides={"nondim": {"temperature_mode": "kelvin"}})
    result = compute_wall_flux(case, cfg, FLUID)
    err = {o: abs(result.heater_mean(o) - q_target) / q_target for o in (1, 2, 3)}
    assert err[1] > err[2] > err[3]
    assert err[1] > 0.05          # 1차는 눈에 띄게 틀린다
    assert err[2] < 0.02          # 2차는 이미 쓸 만하다


def test_first_order_underestimates_a_convex_profile():
    """지수 프로파일에서 1차 차분은 기울기를 과소평가한다 → q'' 가 낮게 나온다."""
    case = _exponential_case(layer_cells=3.0)
    cfg = load_config(overrides={"nondim": {"temperature_mode": "kelvin"}})
    result = compute_wall_flux(case, cfg, FLUID)
    assert result.heater_mean(1) < result.heater_mean(2) < result.heater_mean(3)


def test_mixed_cell_conductivity_is_actually_used():
    """증기 열이 섞이면 q''(x) 는 유지되고(설계상) k 가 달라진 것이 반영돼야 한다."""
    q_target = 2.0e5
    vapor = tuple(range(0, 8))
    case = _exponential_case(q_target=q_target, vapor_columns=vapor, layer_cells=8.0)
    cfg = load_config(overrides={"nondim": {"temperature_mode": "kelvin"}})
    result = compute_wall_flux(case, cfg, FLUID)
    # 액체 k 를 증기 열에도 잘못 적용했다면 q'' 가 k_l/k_v ≈ 4.2 배로 튄다.
    assert result.heater_mean() == pytest.approx(q_target, rel=0.05)
    assert result.vapor_fraction_wall == pytest.approx(len(vapor) / 32, abs=0.02)


def test_coarsening_preserves_the_heater_mean():
    """뭉개기는 평균일 뿐이므로 히터 평균을 바꾸면 안 된다."""
    case = _exponential_case(n_x=64)
    cfg = load_config(overrides={
        "nondim": {"temperature_mode": "kelvin"},
        "coarsen": {"dx_m": 8 * case.dx},
    })
    result = compute_wall_flux(case, cfg, FLUID)
    coarse = coarsen(result, cfg)
    assert coarse.block_size == 8
    assert float(np.mean(coarse.q_cells)) == pytest.approx(result.heater_mean(), rel=1e-9)
