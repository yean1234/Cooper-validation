"""산출물 — CSV / JSON / 그림 / 사람이 읽는 요약.

그림 표기는 영어로 둔다: 컨테이너에 한글 폰트가 없으면 글자가 깨지고, 어차피 논문·
발표 도표는 영어 표기가 기본이다. 서술은 markdown 요약(한국어)에 남긴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .gate import GateResult  # noqa: E402

__all__ = ["write_outputs"]

# dataviz 팔레트(검증 통과: all-pairs, light). 1=blue, 2=orange.
C_TRUTH = "#2a78d6"
C_PRED = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_boiling_curve(gate: GateResult, path: Path, dpi: int) -> None:
    """log-log 비등곡선 — 기울기가 0.33 인지가 이 그림의 전부다."""
    q = np.array([p.q_coarse_W_m2 for p in gate.points])
    truth = np.array([p.delta_t_truth_K for p in gate.points])
    pred = np.array([p.delta_t_cooper_K for p in gate.points])
    order = np.argsort(q)

    fig, ax = plt.subplots(figsize=(6.2, 4.6), facecolor=SURFACE)
    ax.plot(q[order], truth[order], "o", ms=8, color=C_TRUTH, label="Truth  $\\Delta T_{sup}$")
    ax.plot(
        q[order], pred[order], "s--", ms=7, lw=2, color=C_PRED,
        mfc="none", mew=2, label="Cooper inverse",
    )

    # 기준 기울기 가이드 — 계열이 아니라 참조선이므로 중립 회색 + 직접 라벨.
    gq = np.array([q.min(), q.max()])
    anchor = truth[order][0] / q[order][0] ** gate.slope_target
    gy = anchor * gq**gate.slope_target
    ax.plot(gq, gy, "-", lw=1.4, color=INK_MUTED, alpha=0.55)
    ax.annotate(
        f"slope {gate.slope_target}",
        xy=(gq[-1], gy[-1]), xytext=(-6, -14), textcoords="offset points",
        ha="right", color=INK_MUTED, fontsize=9,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Coarse wall heat flux  $q''$  [W/m$^2$]")
    ax.set_ylabel("Wall superheat  $\\Delta T_{sup}$  [K]")
    ax.set_title(
        f"Boiling curve — fitted slope {gate.fit.slope:.3f} "
        f"(target {gate.slope_target}±{gate.slope_tol})",
        color=INK, fontsize=11, loc="left",
    )
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in leg.get_texts():
        text.set_color(INK_MUTED)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def plot_parity(gate: GateResult, path: Path, dpi: int) -> None:
    """정답 대 예측 — 계열이 하나뿐이라 범례 없이 제목이 이름을 대신한다."""
    truth = np.array([p.delta_t_truth_K for p in gate.points])
    pred = np.array([p.delta_t_cooper_K for p in gate.points])
    lo = 0.8 * min(truth.min(), pred.min())
    hi = 1.2 * max(truth.max(), pred.max())
    line = np.array([lo, hi])
    tol = gate.mape_max

    fig, ax = plt.subplots(figsize=(5.2, 5.0), facecolor=SURFACE)
    ax.fill_between(line, line * (1 - tol), line * (1 + tol), color=GRID, alpha=0.55, lw=0)
    ax.plot(line, line, "-", lw=1.2, color=INK_MUTED, alpha=0.7)
    ax.plot(truth, pred, "o", ms=8, color=C_TRUTH)
    for p in gate.points:
        ax.annotate(
            p.case_id, xy=(p.delta_t_truth_K, p.delta_t_cooper_K),
            xytext=(7, -3), textcoords="offset points", fontsize=8, color=INK_MUTED,
        )
    label_x = lo + 0.18 * (hi - lo)
    ax.annotate(
        f"±{tol:.0%} band", xy=(label_x, label_x * (1 + tol)), xytext=(-4, 6),
        textcoords="offset points", ha="right", fontsize=9, color=INK_MUTED,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Truth  $\\Delta T_{sup}$  [K]")
    ax.set_ylabel("Cooper-inverse  $\\Delta T_{sup}$  [K]")
    ax.set_title(f"Parity — MAPE {gate.mape:.1%}", color=INK, fontsize=11, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def plot_order_audit(diagnostics: dict, path: Path, dpi: int) -> None:
    """차분 차수 수렴 점검 — |1차-2차| 대비 |2차-3차| 가 충분히 작아야 한다."""
    rows = [
        (
            diag["heater_mean_q_by_order"].get(2, diag["heater_mean_q_by_order"].get("2", 0.0)),
            case_id,
            100 * diag["rel_diff_1st_2nd"],
            100 * diag["rel_diff_2nd_3rd"],
        )
        for case_id, diag in diagnostics.items()
        if "rel_diff_1st_2nd" in diag
    ]
    if not rows:
        return
    rows.sort()                                  # 열유속 오름차순 = 경계층이 얇아지는 순
    _, cases, d12, d23 = zip(*rows)
    cases, d12, d23 = list(cases), list(d12), list(d23)

    ypos = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(cases) + 2.0), facecolor=SURFACE)
    ax.plot(d12, ypos, "o", ms=8, color=C_TRUTH, label="|1st − 2nd| / 2nd")
    ax.plot(d23, ypos, "s", ms=8, color=C_PRED, mfc="none", mew=2, label="|2nd − 3rd| / 2nd")
    for y, a, b in zip(ypos, d12, d23):
        ax.plot([b, a], [y, y], "-", lw=1.0, color=GRID, zorder=0)
    ax.set_yticks(ypos, cases, fontsize=9)
    ax.set_xlabel("Relative difference in heater-mean $q''$  [%]")
    ax.set_title("Finite-difference order audit (2nd order is the declared standard)",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlim(left=0)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in leg.get_texts():
        text.set_color(INK_MUTED)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def _summary_markdown(gate: GateResult, manifest: dict) -> str:
    lines = [
        "# Cooper 게이트 결과 (초안)",
        "",
        f"- 판정: **{gate.verdict}**",
        f"- 다음 경로: {gate.route}",
        f"- 케이스 수: {len(gate.points)} (최소 {gate.min_cases})",
        f"- MAPE: **{gate.mape:.1%}** (기준 ≤ {gate.mape_max:.0%})",
        f"- log-log 기울기: **{gate.fit.slope:.3f}** ± {gate.fit.slope_stderr:.3f} "
        f"(목표 {gate.slope_target} ± {gate.slope_tol}), R² = {gate.fit.r_squared:.3f}",
        "",
        "## 판정 근거",
        "",
    ]
    lines += [f"{i}. {r}" for i, r in enumerate(gate.reasons, 1)]
    lines += [
        "",
        "## 케이스별",
        "",
        "| case | q'' [kW/m²] | 정답 ΔT [K] | Cooper ΔT [K] | 상대오차 | 벽 증기분율 | (참고) 역산 R_p [µm] |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in gate.points:
        lines.append(
            f"| {p.case_id} | {p.q_coarse_W_m2 / 1e3:.1f} | {p.delta_t_truth_K:.2f} | "
            f"{p.delta_t_cooper_K:.2f} | {p.rel_error:+.1%} | {p.vapor_fraction:.3f} | "
            f"{p.implied_roughness_um:.3g} |"
        )
    lines += [
        "",
        "> 역산 R_p 는 **진단용**입니다. R_p = 1 µm 고정이 원칙이고, 이 값을 맞추려고 "
        "튜닝하면 '유체별 상수 없음' 주장이 깨집니다. 물리적으로 말이 되는 범위"
        "(대략 0.1~10 µm)를 벗어났는지만 보세요.",
        "",
        "## 차분 차수 수렴",
        "",
    ]
    for case_id, diag in manifest.get("wall_flux_diagnostics", {}).items():
        if "verdict" in diag:
            lines.append(
                f"- `{case_id}`: |1차−2차| {diag['rel_diff_1st_2nd']:.1%}, "
                f"|2차−3차| {diag['rel_diff_2nd_3rd']:.1%} → {diag['verdict']}"
            )
    lines += ["", "## 재현 정보", "", "```json",
              json.dumps(manifest.get("provenance", {}), indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    gate: GateResult,
    coarse_rows: list[dict],
    manifest: dict,
    make_plots: bool = True,
    dpi: int = 160,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    case_df = pd.DataFrame([
        {
            "case_id": p.case_id,
            "q_coarse_W_m2": p.q_coarse_W_m2,
            "delta_t_truth_K": p.delta_t_truth_K,
            "delta_t_cooper_K": p.delta_t_cooper_K,
            "rel_error": p.rel_error,
            "abs_rel_error": abs(p.rel_error),
            "vapor_fraction_wall": p.vapor_fraction,
            "implied_roughness_um": p.implied_roughness_um,
        }
        for p in gate.points
    ])
    case_path = out_dir / "case_summary.csv"
    case_df.to_csv(case_path, index=False)
    written["case_summary"] = str(case_path)

    if coarse_rows:
        coarse_path = out_dir / "coarse_inputs.csv"
        pd.DataFrame(coarse_rows).to_csv(coarse_path, index=False)
        written["coarse_inputs"] = str(coarse_path)

    gate_path = out_dir / "gate_result.json"
    gate_path.write_text(json.dumps(gate.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    written["gate_result"] = str(gate_path)

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
    written["run_manifest"] = str(manifest_path)

    summary_path = out_dir / "summary.md"
    summary_path.write_text(_summary_markdown(gate, manifest), encoding="utf-8")
    written["summary"] = str(summary_path)

    if make_plots:
        bc = out_dir / "boiling_curve.png"
        plot_boiling_curve(gate, bc, dpi)
        written["boiling_curve"] = str(bc)

        par = out_dir / "parity.png"
        plot_parity(gate, par, dpi)
        written["parity"] = str(par)

        audit = out_dir / "order_audit.png"
        plot_order_audit(manifest.get("wall_flux_diagnostics", {}), audit, dpi)
        if audit.exists():
            written["order_audit"] = str(audit)

    return written
