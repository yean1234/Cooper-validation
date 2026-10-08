"""전체 파이프라인 오케스트레이션.

    BubbleML 원본
      → [wallflux]  벽 열유속 q''(t, x)          (섞인 셀 포함, 2차 차분)
      → [coarsen]   거친 입력 q''_bar, α_v       (뭉개기; 여기까지는 그냥 평균)
      → [cooper]    ΔT_sup 예측                  (Cooper 역산)
      → [gate]      정답 ΔT_sup 과 비교           (MAPE + log-log 기울기)
      → [report]    CSV / JSON / 그림 / 요약

실측 모드 (``experiments.csv`` 를 주면):
    실제 표면에서 잰 (q'', ΔT) 비등곡선 → [cooper] 곡선마다 그 압력으로 ΔT 예측
      → [gate] 곡선별 MAPE + 기울기, 압력 항, 평가 구간 민감도 → [report]
    BubbleML 은 사이트 수를 처방해서 Cooper 를 채점할 수 없으므로 Cooper 검증은 이 경로로 한다.

정답 표 모드 (``ground_truth.csv`` 를 주면):
    정답 기포장 레포가 확정한 q'' (1차 벽 기울기 × phase-averaged k)
      → [cooper] ΔT_sup 예측 → [gate] 정답 ΔT_sup 과 비교 → [report]
    HDF5 를 다시 읽지 않는다. 정답 쪽이 이미 확정한 열유속을 우리가 다른 식으로
    다시 뽑으면 '정답'이 두 개가 되기 때문이다.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .bubbleml import discover_cases, load_case
from .coarsen import coarsen
from .config import Config
from .cooper import CooperModel
from .fluids import get_fluid, load_fluid_library
from .experiments import load_curves, run_experiments
from .gate import evaluate_gate, make_point
from .ground_truth import Q_DEFINITIONS, load_ground_truth
from .report import write_experiment_outputs, write_outputs
from .wallflux import compute_wall_flux

log = logging.getLogger(__name__)

__all__ = ["run_pipeline"]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _file_digest(path: Path, limit_bytes: int = 4 << 20) -> str:
    """파일 앞부분 해시 — 대용량 HDF5 전체를 읽지 않으면서 버전을 구분한다."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit_bytes))
    return f"sha256:{h.hexdigest()[:16]} (first {limit_bytes} B)"


def run_pipeline(cfg: Config) -> dict:
    overrides = cfg.get("cooper.fluid_overrides")
    if overrides:
        load_fluid_library(overrides)
    fluid = get_fluid(cfg.get("cooper.fluid"))
    model = CooperModel(
        fluid=fluid,
        roughness_um=float(cfg.get("cooper.roughness_um", 1.0)),
        allow_tuning=bool(cfg.get("cooper.allow_tuning", False)),
    )

    if cfg.get("wall_function.enabled"):
        from .wallfunction import WallFunctionNotWired

        raise WallFunctionNotWired(
            "벽함수 경로는 아직 데이터에 연결되지 않았습니다 (u_tau 를 위한 속도장 미로딩). "
            "9/21 일정에서는 Cooper 한 경로만 돌립니다 — wall_function.enabled=false 로 두세요."
        )

    run_id = cfg.get("run.run_id") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg.get("run.out_dir", "outputs")) / run_id

    if cfg.get("experiments.csv"):
        return _run_from_experiments(cfg, model, run_id, out_dir)
    if cfg.get("ground_truth.csv"):
        return _run_from_ground_truth(cfg, model, fluid, run_id, out_dir)

    paths = discover_cases(cfg)
    log.info("케이스 %d개 발견", len(paths))

    points, coarse_rows = [], []
    case_meta, diagnostics, inputs = {}, {}, {}

    for path in paths:
        log.info("처리 중: %s", path.name)
        case = load_case(path, cfg, fluid)
        flux = compute_wall_flux(case, cfg, fluid)
        coarse = coarsen(flux, cfg)

        points.append(
            make_point(
                case_id=case.meta.case_id,
                q_coarse=coarse.q_heater_mean,
                delta_t_truth=case.meta.delta_t_sup_truth,
                model=model,
                vapor_fraction=coarse.vapor_fraction_heater,
            )
        )
        coarse_rows.extend(coarse.to_rows())
        case_meta[case.meta.case_id] = case.meta.to_dict()
        diagnostics[case.meta.case_id] = flux.diagnostics
        inputs[case.meta.case_id] = {
            "file": str(path),
            "digest": _file_digest(path),
            "block_size_fine_cells": coarse.block_size,
            "dx_coarse_m": coarse.dx_coarse_m,
            "dx_fine_m": coarse.dx_fine_m,
            "q_heater_mean_W_m2": coarse.q_heater_mean,
            "vapor_fraction_heater": coarse.vapor_fraction_heater,
        }

    gate = evaluate_gate(points, cfg)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "label": cfg.get("run.label"),
        "config": cfg.to_dict(),
        "config_source": cfg.source_path,
        "cooper_model": model.describe(),
        "fluid": fluid.to_dict(),
        "cases": case_meta,
        "inputs": inputs,
        "wall_flux_diagnostics": diagnostics,
        "provenance": {
            "reference_difference_order": cfg.get("wall_flux.order"),
            "mixed_cells_included": cfg.get("wall_flux.include_mixed_cells"),
            "conductivity_mixing_rule": cfg.get("phase.mixing_rule"),
            "roughness_um": cfg.get("cooper.roughness_um"),
            "roughness_tuned": cfg.get("cooper.allow_tuning"),
            "temperature_scales": {k: v["provenance"] for k, v in case_meta.items()},
        },
    }

    written = write_outputs(
        out_dir=out_dir,
        gate=gate,
        coarse_rows=coarse_rows,
        manifest=manifest,
        make_plots=bool(cfg.get("report.make_plots", True)),
        dpi=int(cfg.get("report.dpi", 160)),
    )
    return {"run_id": run_id, "out_dir": str(out_dir), "gate": gate, "files": written,
            "manifest": manifest}


def _ground_truth_points(conditions, key: str, model: CooperModel) -> list:
    return [
        make_point(
            case_id=c.case_id,
            q_coarse=c.q_W_m2[key],
            delta_t_truth=c.delta_t_sup_K,
            model=model,
            vapor_fraction=float("nan"),        # 인계 표에 벽 증기분율은 없다
            n_sites_prescribed=c.n_sites_prescribed,
        )
        for c in conditions
    ]


def _run_from_ground_truth(cfg: Config, model: CooperModel, fluid, run_id: str, out_dir: Path) -> dict:
    csv_path = Path(cfg.get("ground_truth.csv"))
    conditions = load_ground_truth(csv_path)
    log.info("정답 표에서 조건 %d개를 읽었습니다: %s", len(conditions), csv_path)

    key = str(cfg.get("ground_truth.q_definition", "arith"))
    if key not in Q_DEFINITIONS:
        raise ValueError(f"ground_truth.q_definition 은 {sorted(Q_DEFINITIONS)} 중 하나 — 받은 값 {key!r}")
    if key not in conditions[0].q_W_m2:
        raise KeyError(f"정답 표에 {Q_DEFINITIONS[key].column!r} 컬럼이 없습니다")

    gate = evaluate_gate(_ground_truth_points(conditions, key, model), cfg)

    # q 정의 민감도: 네 정의를 전부 같은 게이트로 돌려 나란히 남긴다. 판정은 위의 하나로만 한다.
    sensitivity = []
    for other in Q_DEFINITIONS:
        if other not in conditions[0].q_W_m2:
            continue
        g = gate if other == key else evaluate_gate(_ground_truth_points(conditions, other, model), cfg)
        sensitivity.append({
            "q_definition": other,
            "column": Q_DEFINITIONS[other].column,
            "role": Q_DEFINITIONS[other].role,
            "used_for_verdict": other == key,
            "mape": g.mape,
            "n_within_band": g.n_within,
            "verdict": g.verdict,
            "delta_t_cooper_K": [p.delta_t_cooper_K for p in g.points],
        })
    peers = [r for r in sensitivity
             if r["role"] in ("기본", "동등 후보") and not r["used_for_verdict"]]
    split = [r["q_definition"] for r in peers if r["verdict"] != gate.verdict]
    if split:
        gate.reasons.append(
            f"주의: 동등 후보 q 정의({', '.join(split)})로는 판정이 달라집니다 — "
            "산술/조화 선택이 확정되기 전까지 이 판정은 정의 선택에 기대고 있습니다."
        )
    elif peers:
        gate.reasons.append(
            f"동등 후보 q 정의({', '.join(r['q_definition'] for r in peers)})로도 판정이 같습니다 "
            "(산술/조화 선택에 기대지 않는 결론)."
        )

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "label": cfg.get("run.label"),
        "mode": "ground_truth_table",
        "config": cfg.to_dict(),
        "config_source": cfg.source_path,
        "cooper_model": model.describe(),
        "fluid": fluid.to_dict(),
        "ground_truth": {
            "csv": str(csv_path),
            "digest": _file_digest(csv_path),
            "q_definition": key,
            "q_column": Q_DEFINITIONS[key].column,
            "conditions": [
                {
                    "case_id": c.case_id,
                    "T_wall_C": c.T_wall_C,
                    "delta_t_sup_K": c.delta_t_sup_K,
                    "n_sites_prescribed": c.n_sites_prescribed,
                    "n_sites_active_2d": c.n_sites_active,
                    "q_W_m2": c.q_W_m2,
                }
                for c in conditions
            ],
        },
        "q_definition_sensitivity": sensitivity,
        "provenance": {
            "q_source": "정답 기포장 인계 표 (1차 벽 기울기, phase-averaged k)",
            "delta_t_source": "인계 표 input_dT_sup_K (시뮬레이션 입력)",
            "slope_role": gate.slope_role,
            "roughness_um": cfg.get("cooper.roughness_um"),
            "roughness_tuned": cfg.get("cooper.allow_tuning"),
        },
    }

    written = write_outputs(
        out_dir=out_dir,
        gate=gate,
        coarse_rows=[],
        manifest=manifest,
        make_plots=bool(cfg.get("report.make_plots", True)),
        dpi=int(cfg.get("report.dpi", 160)),
    )
    return {"run_id": run_id, "out_dir": str(out_dir), "gate": gate, "files": written,
            "manifest": manifest}


def _run_from_experiments(cfg: Config, model: CooperModel, run_id: str, out_dir: Path) -> dict:
    paths = cfg.get("experiments.csv")
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    curves = load_curves(paths, default_fluid=cfg.get("cooper.fluid", "FC-72"))
    log.info("실측 곡선 %d개를 읽었습니다 (%d개 파일)", len(curves), len(paths))
    run = run_experiments(curves, cfg)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "label": cfg.get("run.label"),
        "mode": "experiments",
        "config": cfg.to_dict(),
        "config_source": cfg.source_path,
        "cooper_model_at_1atm": model.describe(),
        "inputs": [str(p) for p in paths],
        "input_digests": {str(p): _file_digest(Path(p)) for p in paths},
        "curves": [
            {"key": c.key, "file": c.file, "fluid": c.fluid, "surface": c.surface,
             "orientation_deg": c.orientation_deg, "p_Pa": c.p_Pa, "n_points": int(c.q.size)}
            for c in curves
        ],
        "provenance": {
            "dT_source": "실측 (각 출처의 벽 과열도 정의 그대로)",
            "q_source": "실측 (각 출처의 열유속 정의 그대로)",
            "window": {"q_min_W_m2": run.window[0], "q_max_frac": run.window[1]},
            "q_ref_W_m2": run.q_ref,
            "roughness_um": cfg.get("cooper.roughness_um"),
            "roughness_tuned": cfg.get("cooper.allow_tuning"),
            "slope_role": "criterion (실측 곡선에는 사이트 수 처방이 없음)",
        },
    }
    written = write_experiment_outputs(
        out_dir=out_dir, run=run, cfg=cfg, manifest=manifest,
        make_plots=bool(cfg.get("report.make_plots", True)), dpi=int(cfg.get("report.dpi", 160)),
    )
    return {"run_id": run_id, "out_dir": str(out_dir), "experiments": run, "files": written,
            "manifest": manifest}
