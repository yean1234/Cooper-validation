import numpy as np
import pytest

from cooperval.config import load_config
from cooperval.cooper import CooperModel
from cooperval.fluids import get_fluid
from cooperval.gate import evaluate_gate, fit_loglog, make_point

MODEL = CooperModel(fluid=get_fluid("FC-72"))
CFG = load_config()


def _points(scale=1.0, exponent=None):
    """정답 ΔT 목록에서 Cooper 정합 q'' 를 만들고, 필요하면 일부러 비튼다."""
    truths = np.array([20.0, 26.0, 32.0, 38.0, 44.0, 50.0])
    q = MODEL.q_from_delta_t(truths)
    if exponent is not None:
        q = q[len(q) // 2] * (truths / truths[len(truths) // 2]) ** exponent
    q = q * scale
    return [
        make_point(f"c{i}", qi, ti, MODEL, vapor_fraction=0.1)
        for i, (qi, ti) in enumerate(zip(q, truths))
    ]


def test_loglog_slope_of_cooper_consistent_data_is_one_third():
    pts = _points()
    fit = fit_loglog(
        np.array([p.q_coarse_W_m2 for p in pts]),
        np.array([p.delta_t_truth_K for p in pts]),
    )
    assert fit.slope == pytest.approx(0.33, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_perfect_data_passes():
    gate = evaluate_gate(_points(), CFG)
    assert gate.verdict == "통과"
    assert gate.mape == pytest.approx(0.0, abs=1e-9)


def test_prefactor_offset_is_conditional_pass_not_failure():
    """프리팩터만 어긋나면 기울기는 살아 있다 → 중단이 아니라 조건부 통과."""
    gate = evaluate_gate(_points(scale=4.0), CFG)
    assert gate.passed_slope and not gate.passed_mape
    assert gate.verdict.startswith("조건부 통과")


def test_wrong_exponent_fails_the_slope_gate():
    """형태가 다르면 상수로 못 속인다 — 이게 단일점 비교보다 강한 이유."""
    gate = evaluate_gate(_points(exponent=2.0), CFG)
    assert not gate.passed_slope
    assert gate.verdict.startswith("불일치")
    assert "벽함수" in gate.route


def test_single_point_is_never_conclusive():
    """점이 하나면 프리팩터로 무조건 맞출 수 있으므로 판정 불가여야 한다."""
    gate = evaluate_gate(_points()[:2], CFG)
    assert gate.verdict == "판정 불가"
    assert not gate.enough_cases


def test_slope_tolerance_is_respected():
    pts = _points(exponent=1.0 / 0.36)      # 기울기 0.36 → 0.33 ± 0.05 안
    gate = evaluate_gate(pts, CFG)
    assert gate.passed_slope
