"""게이트 판정 — 두 개의 기준을 동시에 본다.

기준 1. **MAPE ≤ 30%**
    Cooper 의 문헌 산포가 h 기준 ±30~40% 다. 우리 데이터가 나빠서가 아니라 식 자체의
    정밀도가 그렇다. ±10% 를 걸면 Cooper 가 정상 작동해도 떨어진다 — 그래서 30% 는
    봐주는 게 아니라 '도구의 실제 정밀도에 맞춘' 기준이다.

기준 2. **log-log 기울기 0.33 ± 0.05**
    ΔT = (상수) × q''^0.33 이므로 log-log 에서 기울기 0.33 인 직선이어야 한다.
    프리팩터는 직선을 위아래로 평행이동만 시키고 기울기는 못 바꾼다.
    → **기울기는 상수로 속일 수 없는 부분**이라 단일점 일치보다 훨씬 강한 증거다.
    (단일점은 프리팩터 하나로 무조건 맞출 수 있어서 통과해도 의미가 없다.)

판정은 '중단/진행'의 이진이 아니라 **경로 선택**이다. 멘토 조언(9/9)대로
실패도 결론이 되게 만든다 — 자폭 분기를 만들지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .cooper import CooperModel, DELTA_T_EXPONENT, implied_roughness_um

__all__ = ["CasePoint", "LogLogFit", "GateResult", "fit_loglog", "evaluate_gate"]


@dataclass
class CasePoint:
    case_id: str
    q_coarse_W_m2: float
    delta_t_truth_K: float
    delta_t_cooper_K: float
    vapor_fraction: float
    rel_error: float = field(init=False)
    implied_roughness_um: float = float("nan")

    def __post_init__(self) -> None:
        self.rel_error = (
            (self.delta_t_cooper_K - self.delta_t_truth_K) / self.delta_t_truth_K
            if self.delta_t_truth_K
            else float("nan")
        )


@dataclass
class LogLogFit:
    slope: float
    intercept: float
    slope_stderr: float
    r_squared: float
    n: int

    @property
    def slope_ci95(self) -> tuple[float, float]:
        # 표본이 5~10개라 정규근사는 낙관적이다. 초안에서는 ±2σ 로 두고,
        # 정식 게이트에서는 t 분포(자유도 n-2)로 바꿀 것.
        return (self.slope - 2 * self.slope_stderr, self.slope + 2 * self.slope_stderr)


def fit_loglog(q: np.ndarray, delta_t: np.ndarray) -> LogLogFit:
    """log(ΔT) = a + b log(q'') 의 최소제곱 적합. b 가 우리가 보려는 기울기."""
    q = np.asarray(q, dtype=float)
    dt = np.asarray(delta_t, dtype=float)
    ok = np.isfinite(q) & np.isfinite(dt) & (q > 0) & (dt > 0)
    x, y = np.log(q[ok]), np.log(dt[ok])
    n = x.size
    if n < 2:
        raise ValueError("log-log 적합에는 유효한 점이 최소 2개 필요합니다")

    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = float(np.sqrt(ss_res / (n - 2) / sxx)) if n > 2 and sxx > 0 else float("nan")
    return LogLogFit(float(slope), float(intercept), se, r2, n)


@dataclass
class GateResult:
    points: list[CasePoint]
    mape: float
    fit: LogLogFit
    mape_max: float
    slope_target: float
    slope_tol: float
    min_cases: int
    passed_mape: bool
    passed_slope: bool
    enough_cases: bool
    verdict: str
    route: str
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "route": self.route,
            "reasons": self.reasons,
            "metrics": {
                "n_cases": len(self.points),
                "mape": self.mape,
                "mape_max": self.mape_max,
                "passed_mape": self.passed_mape,
                "loglog_slope": self.fit.slope,
                "loglog_slope_stderr": self.fit.slope_stderr,
                "loglog_slope_ci95": list(self.fit.slope_ci95),
                "loglog_r_squared": self.fit.r_squared,
                "slope_target": self.slope_target,
                "slope_tol": self.slope_tol,
                "passed_slope": self.passed_slope,
            },
            "points": [asdict(p) for p in self.points],
        }


def evaluate_gate(points: list[CasePoint], cfg) -> GateResult:
    mape_max = float(cfg.get("gate.mape_max", 0.30))
    target = float(cfg.get("gate.slope_target", DELTA_T_EXPONENT))
    tol = float(cfg.get("gate.slope_tol", 0.05))
    min_cases = int(cfg.get("gate.min_cases", 5))

    q = np.array([p.q_coarse_W_m2 for p in points])
    truth = np.array([p.delta_t_truth_K for p in points])
    err = np.array([abs(p.rel_error) for p in points])

    mape = float(np.nanmean(err))
    fit = fit_loglog(q, truth)

    enough = len(points) >= min_cases
    passed_mape = bool(mape <= mape_max)
    passed_slope = bool(abs(fit.slope - target) <= tol)

    reasons: list[str] = []
    if not enough:
        reasons.append(
            f"케이스가 {len(points)}개로 최소 {min_cases}개에 못 미칩니다. "
            "점이 적으면 기울기가 프리팩터와 구분되지 않아 판정 자체가 무의미합니다."
        )
    reasons.append(
        f"MAPE {mape:.1%} {'≤' if passed_mape else '>'} 기준 {mape_max:.0%} "
        f"(Cooper 문헌 산포 ±30~40% 에 맞춘 기준)"
    )
    reasons.append(
        f"log-log 기울기 {fit.slope:.3f} (±{fit.slope_stderr:.3f}), "
        f"목표 {target}±{tol} → {'통과' if passed_slope else '이탈'}; R²={fit.r_squared:.3f}"
    )

    if not enough:
        verdict, route = "판정 불가", "케이스를 더 모은 뒤 재실행"
    elif passed_slope and passed_mape:
        verdict = "통과"
        route = "Cooper 경로로 ③단계(기포 명세서)로 진행"
        reasons.append("거친 q''만으로 벽 과열도를 되찾을 수 있다는 근거가 확보됐습니다.")
    elif passed_slope and not passed_mape:
        verdict = "조건부 통과 (형태 일치, 프리팩터 이탈)"
        route = "Cooper 형태는 유지하고, 10월 게이트에서 벽함수 경로와 프리팩터를 비교"
        reasons.append(
            "기울기가 맞다는 것은 BubbleML 의 비등 물리가 Cooper 와 같은 형태를 따른다는 뜻입니다. "
            "어긋난 것은 상수뿐이고, 그건 Cooper 의 알려진 산포 범위 문제입니다. "
            "단, R_p 를 만져서 맞추면 '유체별 상수 없음'이 깨지므로 튜닝하지 마세요."
        )
    else:
        verdict = "불일치 (기울기 이탈)"
        route = "벽함수(Kader) 경로를 붙여 '무엇이 정답 ΔT를 복원하나' 비교로 전환"
        reasons.append(
            "기울기가 어긋나면 상수를 아무리 만져도 못 맞춥니다 — 이건 진짜 불일치입니다. "
            "다만 이것은 중단 신호가 아니라 경로 전환 신호입니다: "
            "'Cooper 는 안 되고 벽함수는 된다' 도, '두 표준 경로가 모두 안 된다' 도 결론이 됩니다."
        )

    return GateResult(
        points=points,
        mape=mape,
        fit=fit,
        mape_max=mape_max,
        slope_target=target,
        slope_tol=tol,
        min_cases=min_cases,
        passed_mape=passed_mape,
        passed_slope=passed_slope,
        enough_cases=enough,
        verdict=verdict,
        route=route,
        reasons=reasons,
    )


def make_point(case_id, q_coarse, delta_t_truth, model: CooperModel, vapor_fraction) -> CasePoint:
    dt_pred = float(model.delta_t_from_q(np.array([q_coarse]))[0])
    return CasePoint(
        case_id=case_id,
        q_coarse_W_m2=float(q_coarse),
        delta_t_truth_K=float(delta_t_truth),
        delta_t_cooper_K=dt_pred,
        vapor_fraction=float(vapor_fraction),
        implied_roughness_um=implied_roughness_um(model, q_coarse, delta_t_truth),
    )
