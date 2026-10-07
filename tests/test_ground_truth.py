"""정답 기포장 인계 표 → Cooper → 정답 ΔT 비교.

9/22 이후 정답 기포장 레포가 열유속 정의를 확정했고, 그 값을 그대로 넣었더니
'불일치 (기울기 이탈)' 가 나왔다. 아래 테스트는
  (1) 그 숫자가 이 코드로 그대로 재현되는지,
  (2) 사이트 수가 처방된 자료에서 기울기가 판정에서 빠지는지,
  (3) 그렇다고 조건별 크기 이탈까지 지워지지는 않는지
를 고정한다.
"""

from pathlib import Path

import numpy as np
import pytest

from cooperval.config import load_config
from cooperval.cooper import CooperModel
from cooperval.fluids import get_fluid
from cooperval.gate import evaluate_gate, make_point
from cooperval.ground_truth import load_ground_truth
from cooperval.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "ground_truth" / "ground_truth_final_snapshot.csv"
MODEL = CooperModel(fluid=get_fluid("FC-72"))


def _points(conditions, key="arith"):
    return [
        make_point(c.case_id, c.q_W_m2[key], c.delta_t_sup_K, MODEL, float("nan"),
                   n_sites_prescribed=c.n_sites_prescribed)
        for c in conditions
    ]


def test_snapshot_loads_five_conditions_in_superheat_order():
    conds = load_ground_truth(SNAPSHOT)
    assert [c.delta_t_sup_K for c in conds] == [22, 28, 32, 36, 42]
    assert [c.n_sites_prescribed for c in conds] == [10, 16, 20, 24, 30]
    assert conds[0].q_W_m2["arith"] == pytest.approx(39016.1)
    assert set(conds[0].q_W_m2) == {"arith", "harm", "kl", "legacy"}


def test_reproduces_the_reported_mismatch():
    """이미지의 숫자: Cooper 17.6/20.1/21.2/22.4/23.8 K, 평균 오차 32.6%, 기울기 0.70."""
    gate = evaluate_gate(_points(load_ground_truth(SNAPSHOT)), load_config())
    pred = [p.delta_t_cooper_K for p in gate.points]
    assert pred == pytest.approx([17.6, 20.1, 21.2, 22.4, 23.8], abs=0.05)
    assert gate.mape == pytest.approx(0.326, abs=5e-4)
    assert gate.fit.slope == pytest.approx(0.700, abs=5e-4)


def test_prescribed_sites_move_slope_to_reference_but_keep_magnitude_verdict():
    gate = evaluate_gate(_points(load_ground_truth(SNAPSHOT)), load_config())
    assert gate.slope_role == "reference"
    assert not gate.verdict.startswith("불일치")        # 기울기로 판정하지 않는다
    assert gate.verdict.startswith("불통과")           # 그래도 MAPE 32.6% > 30% 는 남는다
    assert gate.n_within == 2
    c = gate.confound
    assert c.sites_vs_delta_t.slope == pytest.approx(1.70, abs=0.01)   # 보고서 8쪽 저장 적합
    assert c.q_vs_sites.slope == pytest.approx(0.836, abs=0.001)
    assert c.error_site_spearman == pytest.approx(-1.0)


def test_forcing_slope_criterion_reproduces_old_verdict_with_warning():
    cfg = load_config(overrides={"gate": {"slope_role": "criterion"}})
    gate = evaluate_gate(_points(load_ground_truth(SNAPSHOT)), cfg)
    assert gate.verdict.startswith("불일치")
    assert any("사이트 처방을 채점" in r for r in gate.reasons)


def _ramped_points(n_sites):
    """크기는 ±10% 안인데 기울기는 0.33 에서 크게 벗어나는 점들."""
    truths = np.array([20.0, 22.0, 24.0, 26.0, 28.0])
    pred = truths * (1.0 + np.linspace(-0.1, 0.1, truths.size))
    q = MODEL.q_from_delta_t(pred)
    return [
        make_point(f"c{i}", qi, ti, MODEL, float("nan"), n_sites_prescribed=ni)
        for i, (qi, ti, ni) in enumerate(zip(q, truths, n_sites))
    ]


def test_slope_off_but_magnitude_ok_passes_only_when_sites_are_prescribed():
    cfg = load_config()
    prescribed = evaluate_gate(_ramped_points([10, 14, 18, 22, 26]), cfg)
    assert not prescribed.passed_slope
    assert prescribed.verdict.startswith("통과")

    physical = evaluate_gate(_ramped_points([np.nan] * 5), cfg)     # 사이트 정보 없음
    assert physical.slope_role == "criterion"
    assert physical.verdict.startswith("불일치")

    constant = evaluate_gate(_ramped_points([20] * 5), cfg)         # 처방이 조건마다 같음
    assert constant.slope_role == "criterion"


def test_unknown_slope_role_is_rejected():
    cfg = load_config(overrides={"gate": {"slope_role": "maybe"}})
    with pytest.raises(ValueError, match="slope_role"):
        evaluate_gate(_points(load_ground_truth(SNAPSHOT)), cfg)


def test_transposed_table_reads_the_same(tmp_path):
    """보고서 표처럼 조건이 열로 놓인 CSV 도 같은 값으로 읽힌다."""
    import pandas as pd

    rows = pd.read_csv(SNAPSHOT, comment="#").set_index("condition").T
    rows.index.name = "item"
    path = tmp_path / "transposed.csv"
    rows.to_csv(path)
    a, b = load_ground_truth(SNAPSHOT), load_ground_truth(path)
    assert [c.case_id for c in a] == [c.case_id for c in b]
    assert [c.q_W_m2 for c in a] == [c.q_W_m2 for c in b]
    assert [c.n_sites_prescribed for c in a] == [c.n_sites_prescribed for c in b]


def test_missing_flux_value_stops_instead_of_dropping(tmp_path):
    text = SNAPSHOT.read_text(encoding="utf-8").replace("69424.7", "")
    path = tmp_path / "hole.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Twall_90"):
        load_ground_truth(path)


def test_pipeline_ground_truth_mode_end_to_end(tmp_path):
    cfg = load_config(
        ROOT / "configs" / "ground_truth.yaml",
        overrides={
            "ground_truth": {"csv": str(SNAPSHOT)},
            "cooper": {"fluid_overrides": str(ROOT / "configs" / "fluids.yaml")},
            "run": {"out_dir": str(tmp_path), "run_id": "gt"},
            "report": {"make_plots": False},
        },
    )
    result = run_pipeline(cfg)
    assert result["gate"].verdict.startswith("불통과")
    sens = {r["q_definition"]: r for r in result["manifest"]["q_definition_sensitivity"]}
    assert sens["arith"]["used_for_verdict"] and not sens["harm"]["used_for_verdict"]
    assert sens["harm"]["verdict"] == result["gate"].verdict        # 동등 후보도 같은 결론
    summary = (tmp_path / "gt" / "summary.md").read_text(encoding="utf-8")
    assert "사이트 처방 교란 분해" in summary and "q 정의 민감도" in summary
