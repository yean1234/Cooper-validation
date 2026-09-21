"""전체 파이프라인 오케스트레이션.

    BubbleML 원본
      → [wallflux]  벽 열유속 q''(t, x)          (섞인 셀 포함, 2차 차분)
      → [coarsen]   거친 입력 q''_bar, α_v       (뭉개기; 여기까지는 그냥 평균)
      → [cooper]    ΔT_sup 예측                  (Cooper 역산)
      → [gate]      정답 ΔT_sup 과 비교           (MAPE + log-log 기울기)
      → [report]    CSV / JSON / 그림 / 요약
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
from .gate import evaluate_gate, make_point
from .report import write_outputs
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
