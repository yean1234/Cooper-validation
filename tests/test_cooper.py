import numpy as np
import pytest

from cooperval.cooper import CooperModel, DELTA_T_EXPONENT, implied_roughness_um
from cooperval.fluids import get_fluid


def test_water_delta_t_is_physically_sane():
    """물, 1 atm, 100 kW/m^2 → 핵비등 과열도는 문헌상 대략 10 K 근처."""
    model = CooperModel(fluid=get_fluid("water"))
    dt = float(model.delta_t_from_q(np.array([1.0e5]))[0])
    assert 6.0 < dt < 18.0


def test_fc72_delta_t_is_physically_sane():
    """FC-72 는 물보다 h 가 훨씬 낮아 같은 열유속에서 더 뜨겁다."""
    water = CooperModel(fluid=get_fluid("water"))
    fc72 = CooperModel(fluid=get_fluid("FC-72"))
    q = np.array([1.0e5])
    assert float(fc72.delta_t_from_q(q)[0]) > float(water.delta_t_from_q(q)[0])
    assert 10.0 < float(fc72.delta_t_from_q(q)[0]) < 50.0


def test_forward_inverse_roundtrip():
    model = CooperModel(fluid=get_fluid("FC-72"))
    dt = np.array([10.0, 20.0, 35.0])
    assert model.delta_t_from_q(model.q_from_delta_t(dt)) == pytest.approx(dt, rel=1e-10)


def test_delta_t_scales_as_q_to_the_one_third():
    """q'' 가 8배면 ΔT 는 2배 (세제곱근) — log-log 기울기 0.33 의 근거."""
    model = CooperModel(fluid=get_fluid("FC-72"))
    dt = model.delta_t_from_q(np.array([1.0e5, 8.0e5]))
    assert dt[1] / dt[0] == pytest.approx(8.0**DELTA_T_EXPONENT, rel=1e-12)


def test_roughness_is_locked_by_default():
    """R_p 튜닝은 '유체별 상수 없음' 주장을 깨므로 코드가 막는다."""
    with pytest.raises(ValueError, match="R_p"):
        CooperModel(fluid=get_fluid("FC-72"), roughness_um=3.0)
    CooperModel(fluid=get_fluid("FC-72"), roughness_um=3.0, allow_tuning=True)


def test_implied_roughness_round_trips():
    """정답이 Cooper 자신의 예측이면 역산 조도는 1 µm 로 돌아와야 한다."""
    model = CooperModel(fluid=get_fluid("FC-72"))
    q = 2.0e5
    dt = float(model.delta_t_from_q(np.array([q]))[0])
    assert implied_roughness_um(model, q, dt) == pytest.approx(1.0, rel=1e-6)
