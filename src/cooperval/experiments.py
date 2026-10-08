"""실측 비등곡선 → Cooper → 곡선별 게이트.

BubbleML 은 핵생성 자리 수를 입력으로 처방해서 "실제 표면에서 q'' → ΔT" 를 채점할 수 없다
(docs/GROUND_TRUTH_CHECK.md). 이 모듈은 실제 표면에서 잰 (q'', ΔT) 곡선으로 같은 게이트를 돈다.

입력 CSV (한 줄 = 한 측정점, '#' 로 시작하는 줄은 출처 주석)
    필수: curve_id, p_Pa, q_W_m2, dT_K
    선택: source, fluid(기본 cooper.fluid), surface, orientation_deg, branch(asc|desc, 기본 asc), T_sat_C
곡선 하나 = 같은 표면·압력·방향에서 열유속을 올려 가며 잰 점들. 곡선마다 그 압력으로 p_r 을
다시 계산한 Cooper 식을 쓰고, 곡선마다 따로 판정한다. 표면이 다른 곡선을 섞으면 표면 차이가
Cooper 오차처럼 보이기 때문이다 (같은 열유속에서 실험끼리 과열도가 3배 넘게 다르다).

평가 구간 (판단값 — 결과에 민감도 표를 같이 남긴다)
    상승(asc) 측정만, q ≥ q_min_W_m2 (끓기 시작 직후의 부분 핵비등 제외),
    q ≤ q_max_frac × 곡선 최대 q (임계열유속에 다가가며 곡선이 눕는 구간 제외).
실측 곡선에는 사이트 수 처방이 없으므로 기울기 기준이 다시 판정에 쓰인다 (slope_role auto → criterion).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .cooper import CooperModel, implied_roughness_um
from .fluids import Fluid, get_fluid
from .gate import GateResult, LogLogFit, evaluate_gate, fit_loglog, make_point

log = logging.getLogger(__name__)

__all__ = [
    "Curve", "CurveResult", "PressureTrend", "load_curves", "window_mask",
    "evaluate_curve", "pressure_trends", "window_sensitivity", "ExperimentRun", "run_experiments",
]

REQUIRED = ("curve_id", "p_Pa", "q_W_m2", "dT_K")
PRESSURE_SPREAD_WARN = 0.05


@dataclass
class Curve:
    curve_id: str
    source: str
    fluid: str
    surface: str
    orientation_deg: float
    p_Pa: float                   # 곡선 평균 압력
    q: np.ndarray                 # 측정 순서 그대로 [W/m²]
    dT: np.ndarray                # [K]
    branch: np.ndarray            # 'asc' | 'desc'
    file: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}:{self.curve_id}"

    @property
    def ascending(self) -> np.ndarray:
        return self.branch == "asc"


def read_commented_csv(path: Path) -> pd.DataFrame:
    """'#' 로 시작하는 줄만 출처 주석으로 건너뛴다.

    pandas 의 comment='#' 는 줄 중간의 '#' 뒤도 잘라 버려서 값에 '#' 가 들어가면
    (예: '#1500 emery') 컬럼이 밀린다.
    """
    with open(path, encoding="utf-8") as fh:
        body = "".join(line for line in fh if not line.lstrip().startswith("#"))
    return pd.read_csv(io.StringIO(body))


def _read(path: Path) -> pd.DataFrame:
    df = read_commented_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: 필수 컬럼 {missing} 이 없습니다. 있는 컬럼: {list(df.columns)}")
    for col in ("q_W_m2", "dT_K", "p_Pa"):
        values = pd.to_numeric(df[col], errors="coerce")
        bad = ~np.isfinite(values) | (values <= 0)
        if bad.any():
            # 조용히 빼면 곡선 모양이 바뀐다 — 멈춘다.
            raise ValueError(f"{path}: {col} 이 비었거나 0 이하인 줄이 {int(bad.sum())}개 있습니다 "
                             f"(첫 줄 번호 {int(np.flatnonzero(bad)[0])})")
        df[col] = values
    df["_file"] = str(path)
    return df


def load_curves(paths, default_fluid: str = "FC-72") -> list[Curve]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = [_read(Path(p)) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    defaults = {"source": "unknown", "fluid": default_fluid, "surface": "unknown",
                "orientation_deg": np.nan, "branch": "asc"}
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
        df[col] = df[col].fillna(value)
    df["branch"] = df["branch"].astype(str).str.strip().str.lower()
    bad_branch = ~df["branch"].isin(["asc", "desc"])
    if bad_branch.any():
        raise ValueError(f"branch 는 asc/desc 만 됩니다: {sorted(df.loc[bad_branch, 'branch'].unique())}")

    curves: list[Curve] = []
    for (source, curve_id), g in df.groupby(["source", "curve_id"], sort=False):
        p = g["p_Pa"].to_numpy(float)
        spread = (p.max() - p.min()) / p.mean()
        if spread > PRESSURE_SPREAD_WARN:
            log.warning("[%s:%s] 곡선 안 압력이 %.1f%% 흔들립니다 — 평균 %.0f Pa 를 씁니다",
                        source, curve_id, 100 * spread, p.mean())
        for col in ("fluid", "surface", "orientation_deg"):
            if g[col].nunique(dropna=False) > 1:
                raise ValueError(f"[{source}:{curve_id}] 한 곡선 안에 {col} 값이 여러 개입니다: "
                                 f"{g[col].unique().tolist()}")
        curves.append(Curve(
            curve_id=str(curve_id),
            source=str(source),
            fluid=str(g["fluid"].iloc[0]),
            surface=str(g["surface"].iloc[0]),
            orientation_deg=float(g["orientation_deg"].iloc[0]),
            p_Pa=float(p.mean()),
            q=g["q_W_m2"].to_numpy(float),
            dT=g["dT_K"].to_numpy(float),
            branch=g["branch"].to_numpy(str),
            file=str(g["_file"].iloc[0]),
        ))
    return curves


def window_mask(curve: Curve, q_min: float, q_max_frac: float) -> np.ndarray:
    asc = curve.ascending
    if not asc.any():
        return asc
    q_top = curve.q[asc].max()
    return asc & (curve.q >= q_min) & (curve.q <= q_max_frac * q_top * (1 + 1e-12))


def _interp_loglog(q: np.ndarray, dT: np.ndarray, q_ref: float) -> float:
    """상승 곡선에서 q_ref 의 ΔT 를 log-log 선형보간 (범위 밖이면 nan, 외삽하지 않는다)."""
    order = np.argsort(q, kind="stable")
    q, dT = q[order], dT[order]
    if q.size < 2 or not (q[0] <= q_ref <= q[-1]):
        return float("nan")
    return float(np.exp(np.interp(np.log(q_ref), np.log(q), np.log(dT))))


@dataclass
class CurveResult:
    curve: Curve
    model: CooperModel
    in_window: np.ndarray
    dT_cooper: np.ndarray                 # 모든 점의 Cooper 예측
    gate: GateResult | None               # 구간 안 점이 2개 미만이면 None
    dT_meas_at_ref: float
    dT_cooper_at_ref: float
    implied_rp_at_ref_um: float

    @property
    def verdict(self) -> str:
        return self.gate.verdict if self.gate else "판정 불가"

    @property
    def n_window(self) -> int:
        return int(self.in_window.sum())

    @property
    def bias(self) -> float:
        """구간 안 평균 부호 오차 — +면 Cooper 가 과열도를 높게 낸다."""
        if not self.gate:
            return float("nan")
        return float(np.mean([p.rel_error for p in self.gate.points]))

    def to_row(self) -> dict:
        c, g = self.curve, self.gate
        return {
            "source": c.source, "curve_id": c.curve_id, "fluid": c.fluid, "surface": c.surface,
            "orientation_deg": c.orientation_deg, "p_Pa": c.p_Pa,
            "reduced_pressure": self.model.fluid.reduced_pressure,
            "n_points": int(c.q.size), "n_window": self.n_window,
            "q_window_min_W_m2": float(c.q[self.in_window].min()) if self.n_window else float("nan"),
            "q_window_max_W_m2": float(c.q[self.in_window].max()) if self.n_window else float("nan"),
            "mape": g.mape if g else float("nan"),
            "bias": self.bias,
            "slope": g.fit.slope if g else float("nan"),
            "slope_stderr": g.fit.slope_stderr if g else float("nan"),
            "slope_r2": g.fit.r_squared if g else float("nan"),
            "passed_mape": g.passed_mape if g else False,
            "passed_slope": g.passed_slope if g else False,
            "verdict": self.verdict,
            "dT_meas_at_ref_K": self.dT_meas_at_ref,
            "dT_cooper_at_ref_K": self.dT_cooper_at_ref,
            "implied_rp_at_ref_um": self.implied_rp_at_ref_um,
        }


def _model_for(curve: Curve, cfg) -> CooperModel:
    fluid: Fluid = get_fluid(curve.fluid).replace(p_operating_Pa=curve.p_Pa)
    return CooperModel(
        fluid=fluid,
        roughness_um=float(cfg.get("cooper.roughness_um", 1.0)),
        allow_tuning=bool(cfg.get("cooper.allow_tuning", False)),
    )


def evaluate_curve(curve: Curve, cfg, q_min: float | None = None,
                   q_max_frac: float | None = None) -> CurveResult:
    q_min = float(cfg.get("experiments.q_min_W_m2", 2.0e4) if q_min is None else q_min)
    q_max_frac = float(cfg.get("experiments.q_max_frac", 0.8) if q_max_frac is None else q_max_frac)
    q_ref = float(cfg.get("experiments.q_ref_W_m2", 7.2e4))

    model = _model_for(curve, cfg)
    mask = window_mask(curve, q_min, q_max_frac)
    points = [
        make_point(f"{curve.curve_id}#{i}", curve.q[i], curve.dT[i], model, float("nan"))
        for i in np.flatnonzero(mask)
    ]
    gate = evaluate_gate(points, cfg) if len(points) >= 2 else None

    asc = curve.ascending
    dT_ref = _interp_loglog(curve.q[asc], curve.dT[asc], q_ref)
    return CurveResult(
        curve=curve,
        model=model,
        in_window=mask,
        dT_cooper=np.asarray(model.delta_t_from_q(curve.q), dtype=float),
        gate=gate,
        dT_meas_at_ref=dT_ref,
        dT_cooper_at_ref=float(model.delta_t_from_q(q_ref)),
        implied_rp_at_ref_um=implied_roughness_um(model, q_ref, dT_ref) if np.isfinite(dT_ref) else float("nan"),
    )


@dataclass
class PressureTrend:
    """같은 표면에서 압력만 바꾼 곡선들 — Cooper 의 p_r 항이 맞는지.

    q_ref 에서 ΔT ∝ p_r^m 로 맞춘 지수 m 을 실측과 Cooper 각각 구한다. 프리팩터는 ΔT 를
    평행이동만 시키므로 m 은 상수로 속일 수 없다 — 기울기 기준과 같은 논리다.
    """

    group: str
    curve_ids: list[str]
    reduced_pressure: list[float]
    dT_meas: list[float]
    dT_cooper: list[float]
    fit_meas: LogLogFit
    fit_cooper: LogLogFit

    def to_dict(self) -> dict:
        return {
            "group": self.group, "curve_ids": self.curve_ids,
            "reduced_pressure": self.reduced_pressure,
            "dT_meas_at_ref_K": self.dT_meas, "dT_cooper_at_ref_K": self.dT_cooper,
            "exponent_meas": self.fit_meas.slope, "exponent_meas_stderr": self.fit_meas.slope_stderr,
            "exponent_meas_r2": self.fit_meas.r_squared,
            "exponent_cooper": self.fit_cooper.slope,
        }


def pressure_trends(results: list[CurveResult], min_curves: int = 3) -> list[PressureTrend]:
    groups: dict[str, list[CurveResult]] = {}
    for r in results:
        c = r.curve
        groups.setdefault(f"{c.source} · {c.surface} · {c.orientation_deg:g}°", []).append(r)
    trends = []
    for name, rs in groups.items():
        rs = [r for r in rs if np.isfinite(r.dT_meas_at_ref)]
        pressures = {round(r.curve.p_Pa, -2) for r in rs}
        if len(rs) < min_curves or len(pressures) < min_curves:
            continue
        rs.sort(key=lambda r: r.curve.p_Pa)
        pr = np.array([r.model.fluid.reduced_pressure for r in rs])
        meas = np.array([r.dT_meas_at_ref for r in rs])
        coop = np.array([r.dT_cooper_at_ref for r in rs])
        trends.append(PressureTrend(
            group=name,
            curve_ids=[r.curve.curve_id for r in rs],
            reduced_pressure=pr.tolist(),
            dT_meas=meas.tolist(),
            dT_cooper=coop.tolist(),
            fit_meas=fit_loglog(pr, meas),
            fit_cooper=fit_loglog(pr, coop),
        ))
    return trends


def window_sensitivity(curves: list[Curve], cfg) -> list[dict]:
    """평가 구간을 바꿔도 결론이 같은지 — 구간은 판단값이라 숨기지 않고 표로 남긴다."""
    rows = []
    for q_min, q_max_frac in cfg.get("experiments.window_sensitivity") or []:
        rs = [evaluate_curve(c, cfg, q_min=float(q_min), q_max_frac=float(q_max_frac)) for c in curves]
        judged = [r for r in rs if r.gate and r.gate.enough_cases]
        verdicts = [r.verdict for r in judged]
        per_curve = {
            r.curve.key: {
                "slope": r.gate.fit.slope if r.gate else float("nan"),
                "mape": r.gate.mape if r.gate else float("nan"),
                "bias": r.bias,
                "verdict": r.verdict,
            }
            for r in rs
        }
        rows.append({
            "per_curve": per_curve,
            "q_min_W_m2": float(q_min), "q_max_frac": float(q_max_frac),
            "n_judged": len(judged), "n_curves": len(rs),
            "n_pass": sum(v == "통과" for v in verdicts),
            "n_conditional": sum(v.startswith("조건부") for v in verdicts),
            "n_mismatch": sum(v.startswith("불일치") for v in verdicts),
            "median_mape": float(np.median([r.gate.mape for r in judged])) if judged else float("nan"),
            "median_slope": float(np.median([r.gate.fit.slope for r in judged])) if judged else float("nan"),
        })
    return rows


@dataclass
class ExperimentRun:
    results: list[CurveResult]
    trends: list[PressureTrend]
    sensitivity: list[dict]
    q_ref: float
    window: tuple[float, float]
    pooled_mape: float = float("nan")
    pooled_bias: float = float("nan")
    verdict_counts: dict = field(default_factory=dict)


def run_experiments(curves: list[Curve], cfg) -> ExperimentRun:
    curves = sorted(curves, key=lambda c: (c.source, c.surface, c.orientation_deg, c.p_Pa))
    results = [evaluate_curve(c, cfg) for c in curves]
    errors = np.concatenate([
        [p.rel_error for p in r.gate.points] for r in results if r.gate
    ]) if any(r.gate for r in results) else np.array([])
    counts = {"통과": 0, "조건부": 0, "불일치": 0, "판정 불가": 0}
    for r in results:
        v = r.verdict
        key = "조건부" if v.startswith("조건부") else "불일치" if v.startswith("불일치") else v
        counts[key] = counts.get(key, 0) + 1
    return ExperimentRun(
        results=results,
        trends=pressure_trends(results),
        sensitivity=window_sensitivity(curves, cfg),
        q_ref=float(cfg.get("experiments.q_ref_W_m2", 7.2e4)),
        window=(float(cfg.get("experiments.q_min_W_m2", 2.0e4)), float(cfg.get("experiments.q_max_frac", 0.8))),
        pooled_mape=float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        pooled_bias=float(np.mean(errors)) if errors.size else float("nan"),
        verdict_counts=counts,
    )
