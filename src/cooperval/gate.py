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

기울기 기준의 전제 — **사이트 수가 물리로 정해질 때만** 성립한다.
    실제 비등에서는 과열도가 오르면 활성 사이트가 저절로 늘고, Cooper 의 q''^0.67 은
    그 증가를 품고 있다. BubbleML 은 사이트 수를 조건별 **입력**으로 처방한다
    (Twall 80→100 °C 에 10→30개). 그러면 비등곡선 기울기는 '누가 사이트를 몇 개
    넣었나'로 정해지고, 기울기로 Cooper 를 채점하면 상관식 대신 사이트 처방을
    채점하게 된다 (정답 기포장 보고서 8쪽, 노션 '정답 기포장 문제 해결' 제안).
    그래서 케이스별 처방 사이트 수가 주어지고 조건마다 다르면 기울기는 **참고**로
    내리고(``gate.slope_role: auto``), 판정은 조건별 오차(MAPE)로만 한다.
    기울기는 계속 계산해서 사이트 처방과의 교란 분해와 함께 보고한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .cooper import CooperModel, DELTA_T_EXPONENT, implied_roughness_um

__all__ = [
    "CasePoint", "LogLogFit", "SiteConfound", "GateResult",
    "fit_loglog", "site_confound", "evaluate_gate", "SLOPE_ROLES",
]

SLOPE_ROLES = ("auto", "criterion", "reference")
# 조건별 오차가 처방 사이트 수와 이 이상 순위상관하면 "오차 추세가 사이트 처방을 따라간다"고
# 적는다. 점 5개에서 |ρ|≥0.9 는 단측 순열 p≈0.04 수준이라 고른 값이다 (판단값).
SITE_TREND_RHO = 0.9


@dataclass
class CasePoint:
    case_id: str
    q_coarse_W_m2: float
    delta_t_truth_K: float
    delta_t_cooper_K: float
    vapor_fraction: float
    rel_error: float = field(init=False)
    implied_roughness_um: float = float("nan")
    # 시뮬레이션이 조건별 입력으로 정해 준 사이트 수 (모르면 nan). 기울기 기준의 성립 여부를 가른다.
    n_sites_prescribed: float = float("nan")

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


def _ranks(values: np.ndarray) -> np.ndarray:
    """동순위는 평균 순위로 둔 순위 (Spearman 용)."""
    v = np.asarray(values, dtype=float)
    ranks = np.empty(v.size)
    ranks[np.argsort(v, kind="mergesort")] = np.arange(v.size, dtype=float)
    for val in np.unique(v):
        tie = v == val
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def _spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class SiteConfound:
    """비등곡선 기울기를 사이트 처방과 나머지로 쪼갠 진단.

    q ∝ ΔT^a 는 N ∝ ΔT^b (처방) 와 q ∝ N^c 로 거의 a ≈ b·c 가 된다. b 가 입력이므로
    a 도, 그 역수인 log-log 기울기도 사이트를 몇 개 넣었는지에 묶인다.
    """

    q_vs_delta_t: LogLogFit
    sites_vs_delta_t: LogLogFit
    q_vs_sites: LogLogFit
    q_per_site_W_m2: list[float]
    error_site_spearman: float

    def to_dict(self) -> dict:
        return {
            "q_vs_delta_t_exponent": self.q_vs_delta_t.slope,
            "q_vs_delta_t_r_squared": self.q_vs_delta_t.r_squared,
            "sites_vs_delta_t_exponent": self.sites_vs_delta_t.slope,
            "sites_vs_delta_t_r_squared": self.sites_vs_delta_t.r_squared,
            "q_vs_sites_exponent": self.q_vs_sites.slope,
            "q_vs_sites_r_squared": self.q_vs_sites.r_squared,
            "q_per_site_W_m2": self.q_per_site_W_m2,
            "error_site_spearman": self.error_site_spearman,
        }


def _sites_vary(points: list["CasePoint"]) -> bool:
    n = np.array([p.n_sites_prescribed for p in points], dtype=float)
    return bool(n.size and np.all(np.isfinite(n)) and np.all(n > 0) and np.ptp(n) > 0)


def site_confound(points: list["CasePoint"]) -> SiteConfound:
    q = np.array([p.q_coarse_W_m2 for p in points], dtype=float)
    dt = np.array([p.delta_t_truth_K for p in points], dtype=float)
    n = np.array([p.n_sites_prescribed for p in points], dtype=float)
    err = np.array([p.rel_error for p in points], dtype=float)
    return SiteConfound(
        q_vs_delta_t=fit_loglog(dt, q),
        sites_vs_delta_t=fit_loglog(dt, n),
        q_vs_sites=fit_loglog(n, q),
        q_per_site_W_m2=[float(v) for v in q / n],
        error_site_spearman=_spearman(err, n),
    )


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
    slope_role: str = "criterion"       # criterion | reference (참고만, 판정 제외)
    n_within: int = 0                   # 상대오차가 mape_max 이내인 조건 수
    confound: SiteConfound | None = None

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
                "slope_role": self.slope_role,
                "n_within_band": self.n_within,
            },
            "site_confound": self.confound.to_dict() if self.confound else None,
            "points": [asdict(p) for p in self.points],
        }


def _resolve_slope_role(cfg, points: list[CasePoint]) -> str:
    role = str(cfg.get("gate.slope_role", "auto"))
    if role not in SLOPE_ROLES:
        raise ValueError(f"gate.slope_role 은 {SLOPE_ROLES} 중 하나 — 받은 값 {role!r}")
    if role == "auto":
        return "reference" if _sites_vary(points) else "criterion"
    return role


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
    slope_role = _resolve_slope_role(cfg, points)
    confound = site_confound(points) if _sites_vary(points) else None
    n_within = int(np.sum(err <= mape_max))

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
        f"(Cooper 문헌 산포 ±30~40% 에 맞춘 기준); "
        f"조건별로 ±{mape_max:.0%} 안에 든 것은 {n_within}/{len(points)}개"
    )
    if slope_role == "criterion":
        reasons.append(
            f"log-log 기울기 {fit.slope:.3f} (±{fit.slope_stderr:.3f}), "
            f"목표 {target}±{tol} → {'통과' if passed_slope else '이탈'}; R²={fit.r_squared:.3f}"
        )
    else:
        why = (
            "사이트 수가 조건별 입력으로 처방된 자료라 기울기는 상관식이 아니라 사이트 처방을 잽니다."
            if confound is not None
            else "설정(gate.slope_role=reference)으로 기울기를 판정에서 뺐습니다."
        )
        reasons.append(
            f"log-log 기울기 {fit.slope:.3f} (±{fit.slope_stderr:.3f}), 목표 {target}±{tol} → "
            f"{'안' if passed_slope else '밖'}이지만 **참고만** 합니다(판정 제외). {why}"
        )
    if confound is not None:
        c = confound
        reasons.append(
            f"교란 분해: q ∝ ΔT^{c.q_vs_delta_t.slope:.2f} (R²={c.q_vs_delta_t.r_squared:.3f}) 는 "
            f"처방 사이트 N ∝ ΔT^{c.sites_vs_delta_t.slope:.2f} 와 "
            f"q ∝ N^{c.q_vs_sites.slope:.2f} (R²={c.q_vs_sites.r_squared:.4f}) 로 거의 설명됩니다. "
            f"Cooper 는 q ∝ ΔT^{1.0 / DELTA_T_EXPONENT:.2f} 를 가정합니다."
        )
        if np.isfinite(c.error_site_spearman) and abs(c.error_site_spearman) >= SITE_TREND_RHO:
            reasons.append(
                f"조건별 오차도 처방 사이트 수를 따라 움직입니다 (순위상관 ρ={c.error_site_spearman:+.2f}). "
                "그래서 MAPE 값 자체가 어떤 ΔT 범위를 골랐는지에 달려 있고, 이 자료만으로는 "
                "Cooper 자체의 오차와 사이트 처방의 효과를 분리할 수 없습니다."
            )

    if not enough:
        verdict, route = "판정 불가", "케이스를 더 모은 뒤 재실행"
    elif slope_role == "reference":
        if passed_mape:
            verdict = "통과 (조건별 크기 일치)"
            route = (
                "Cooper 경로로 ③단계(기포 명세서) 진행. 단 기울기(형태)는 이 자료로 검증되지 않았으므로 "
                "사이트 수가 물리로 정해지는 자료(실험 등)에서 따로 확인"
            )
            reasons.append(
                "주어진 열유속에서 Cooper 가 조건별 과열도를 문헌 산포 안에서 되찾았습니다. "
                "사이트 수는 정답 자료의 입력이므로 상관식이 맞힐 대상이 아닙니다."
            )
        else:
            verdict = "불통과 (조건별 크기 이탈)"
            route = (
                "R_p·통과 기준·q 정의를 바꿔 맞추지 말 것. 10월 게이트에서 벽함수(Kader) 경로를 "
                "같은 조건별 기준으로 붙여 비교하고, 기울기(형태)는 사이트 수가 물리로 정해지는 "
                "자료에서 따로 검증"
            )
            reasons.append(
                "기울기를 판정에서 빼고 조건별로만 봐도 Cooper 는 기준 밖입니다. 오차가 사이트 처방을 "
                "따라간다는 것은 '왜 벗어나는지'에 대한 설명이지 판정을 바꿀 근거가 아닙니다. "
                "R_p 를 만져서 맞추면 '유체별 상수 없음'이 깨지므로 튜닝하지 마세요."
            )
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
        if confound is not None:
            reasons.append(
                "주의: 사이트 수가 처방된 자료에 기울기 기준을 강제로 걸었습니다(gate.slope_role=criterion). "
                "이 판정은 사이트 처방을 채점한 것일 수 있습니다."
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
        slope_role=slope_role,
        n_within=n_within,
        confound=confound,
    )


def make_point(
    case_id, q_coarse, delta_t_truth, model: CooperModel, vapor_fraction,
    n_sites_prescribed: float = float("nan"),
) -> CasePoint:
    dt_pred = float(model.delta_t_from_q(np.array([q_coarse]))[0])
    return CasePoint(
        case_id=case_id,
        q_coarse_W_m2=float(q_coarse),
        delta_t_truth_K=float(delta_t_truth),
        delta_t_cooper_K=dt_pred,
        vapor_fraction=float(vapor_fraction),
        implied_roughness_um=implied_roughness_um(model, q_coarse, delta_t_truth),
        n_sites_prescribed=float(n_sites_prescribed),
    )
