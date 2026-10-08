"""실측 비등곡선 → Cooper → 곡선별 게이트.

Cooper 와 정확히 같은 곡선은 통과하고, 기울기가 다른 곡선은 상수로 못 맞추고,
압력 항은 프리팩터와 무관하게 잡히는지를 고정한다. Zimmermann 공개 자료는
변환 결과와 핵심 숫자를 고정한다 (자료가 바뀌면 바로 보이게).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cooperval.config import load_config
from cooperval.cooper import CooperModel
from cooperval.experiments import (
    evaluate_curve, load_curves, pressure_trends, run_experiments, window_mask,
)
from cooperval.fluids import get_fluid
from cooperval.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
ZIMMERMANN = ROOT / "data" / "experiments" / "zimmermann2020_fc72_smooth.csv"


def _write(tmp_path, rows, name="curves.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _cooper_rows(p_Pa, q, curve_id="c", scale=1.0, exponent=None, source="synthetic"):
    """그 압력의 Cooper 곡선 위의 점. scale 로 ΔT 를 상수배, exponent 로 기울기를 비튼다."""
    model = CooperModel(get_fluid("FC-72").replace(p_operating_Pa=p_Pa))
    dT = np.asarray(model.delta_t_from_q(q), dtype=float) * scale
    if exponent is not None:
        mid = len(q) // 2
        dT = dT[mid] * (np.asarray(q) / q[mid]) ** exponent
    return [{"source": source, "curve_id": curve_id, "p_Pa": p_Pa, "q_W_m2": qi, "dT_K": ti}
            for qi, ti in zip(q, dT)]


Q = np.geomspace(2.5e4, 1.0e5, 8)


def test_cooper_consistent_curve_passes(tmp_path):
    cfg = load_config(overrides={"experiments": {"q_max_frac": 1.0}})
    curve = load_curves(_write(tmp_path, _cooper_rows(101325.0, Q)))[0]
    r = evaluate_curve(curve, cfg)
    assert r.verdict == "통과"
    assert r.gate.mape == pytest.approx(0.0, abs=1e-9)
    assert r.gate.fit.slope == pytest.approx(0.33, abs=1e-6)
    assert r.gate.slope_role == "criterion"      # 실측에는 사이트 수 처방이 없다


def test_constant_offset_is_conditional_and_slope_mismatch_is_not(tmp_path):
    cfg = load_config(overrides={"experiments": {"q_max_frac": 1.0}})
    offset = load_curves(_write(tmp_path, _cooper_rows(101325.0, Q, scale=0.5), "a.csv"))[0]
    flat = load_curves(_write(tmp_path, _cooper_rows(101325.0, Q, exponent=0.15), "b.csv"))[0]
    assert evaluate_curve(offset, cfg).verdict.startswith("조건부")
    assert evaluate_curve(flat, cfg).verdict.startswith("불일치")


def test_window_keeps_only_ascending_points_inside_limits(tmp_path):
    rows = _cooper_rows(101325.0, np.array([5e3, 2e4, 5e4, 1e5, 2e5, 5e4]))
    for row, branch in zip(rows, ["asc"] * 5 + ["desc"]):
        row["branch"] = branch
    curve = load_curves(_write(tmp_path, rows))[0]
    mask = window_mask(curve, q_min=2e4, q_max_frac=0.8)
    # 2e5 는 최대의 100% 라 빠지고, 마지막 5e4 는 하강 측정이라 빠진다
    assert mask.tolist() == [False, True, True, True, False, False]


def test_pressure_term_is_measured_independently_of_the_prefactor(tmp_path):
    rows = []
    for p in (5e4, 1e5, 2e5):
        rows += _cooper_rows(p, Q, curve_id=f"p{p:g}", scale=0.5)   # 프리팩터만 틀린 표면
    cfg = load_config(overrides={"experiments": {"q_max_frac": 1.0}})
    results = [evaluate_curve(c, cfg) for c in load_curves(_write(tmp_path, rows))]
    (trend,) = pressure_trends(results)
    assert trend.fit_meas.slope == pytest.approx(trend.fit_cooper.slope, abs=1e-3)


def test_bad_rows_stop_instead_of_being_dropped(tmp_path):
    rows = _cooper_rows(101325.0, Q)
    rows[3]["dT_K"] = np.nan
    with pytest.raises(ValueError, match="dT_K"):
        load_curves(_write(tmp_path, rows))


def test_mixed_surface_inside_one_curve_is_rejected(tmp_path):
    rows = _cooper_rows(101325.0, Q)
    for i, row in enumerate(rows):
        row["surface"] = "polished" if i < 4 else "sanded"
    with pytest.raises(ValueError, match="surface"):
        load_curves(_write(tmp_path, rows))


def test_zimmermann_snapshot_numbers():
    curves = load_curves(ZIMMERMANN)
    assert len(curves) == 6
    assert sum(c.q.size for c in curves) == 98
    run = run_experiments(curves, load_config())
    by_p = {round(r.curve.p_Pa / 1e5, 2): r for r in run.results}
    r = by_p[0.95]
    assert r.dT_meas_at_ref == pytest.approx(9.57, abs=0.02)
    assert r.dT_cooper_at_ref == pytest.approx(21.94, abs=0.02)
    assert all(res.bias > 0.5 for res in run.results if res.gate)   # Cooper 가 과열도를 크게 낸다
    (trend,) = run.trends
    assert trend.fit_meas.slope == pytest.approx(-0.47, abs=0.02)
    assert trend.fit_cooper.slope == pytest.approx(-0.31, abs=0.02)


def test_pipeline_experiment_mode_end_to_end(tmp_path):
    cfg = load_config(
        ROOT / "configs" / "experiments.yaml",
        overrides={
            "experiments": {"csv": [str(ZIMMERMANN)]},
            "cooper": {"fluid_overrides": str(ROOT / "configs" / "fluids.yaml")},
            "run": {"out_dir": str(tmp_path), "run_id": "exp"},
            "report": {"make_plots": False},
        },
    )
    result = run_pipeline(cfg)
    out = tmp_path / "exp"
    for name in ("summary.md", "curves.csv", "points.csv", "experiment_result.json", "run_manifest.json"):
        assert (out / name).exists(), name
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "압력 항" in summary and "평가 구간 민감도" in summary
    assert sum(result["experiments"].verdict_counts.values()) == 6
