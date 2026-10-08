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
    ax.set_xlabel("Wall heat flux  $q''$  [W/m$^2$]")
    ax.set_ylabel("Wall superheat  $\\Delta T_{sup}$  [K]")
    role = (
        f"target {gate.slope_target}±{gate.slope_tol}"
        if gate.slope_role == "criterion"
        else "reference only: sites prescribed"
    )
    ax.set_title(
        f"Boiling curve — fitted slope {gate.fit.slope:.3f} ({role})",
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


def _fmt(value: float, spec: str) -> str:
    return format(value, spec) if np.isfinite(value) else "—"


def _summary_markdown(gate: GateResult, manifest: dict) -> str:
    slope_note = (
        f"(목표 {gate.slope_target} ± {gate.slope_tol})"
        if gate.slope_role == "criterion"
        else "(**참고만 — 사이트 수가 처방된 자료라 판정에서 제외**)"
    )
    lines = [
        "# Cooper 게이트 결과 (초안)",
        "",
        f"- 판정: **{gate.verdict}**",
        f"- 다음 경로: {gate.route}",
        f"- 케이스 수: {len(gate.points)} (최소 {gate.min_cases})",
        f"- MAPE: **{gate.mape:.1%}** (기준 ≤ {gate.mape_max:.0%}); "
        f"조건별 ±{gate.mape_max:.0%} 이내 {gate.n_within}/{len(gate.points)}",
        f"- log-log 기울기: **{gate.fit.slope:.3f}** ± {gate.fit.slope_stderr:.3f} "
        f"{slope_note}, R² = {gate.fit.r_squared:.3f}",
        "",
        "## 판정 근거",
        "",
    ]
    lines += [f"{i}. {r}" for i, r in enumerate(gate.reasons, 1)]

    has_sites = any(np.isfinite(p.n_sites_prescribed) for p in gate.points)
    has_vapor = any(np.isfinite(p.vapor_fraction) for p in gate.points)
    header = "| case | q'' [kW/m²] | 정답 ΔT [K] | Cooper ΔT [K] | 상대오차 |"
    rule = "|---|---|---|---|---|"
    if has_sites:
        header += " 처방 사이트 수 |"
        rule += "---|"
    if has_vapor:
        header += " 벽 증기분율 |"
        rule += "---|"
    header += " (참고) 역산 R_p [µm] |"
    rule += "---|"
    lines += ["", "## 케이스별", "", header, rule]
    for p in gate.points:
        row = (
            f"| {p.case_id} | {p.q_coarse_W_m2 / 1e3:.1f} | {p.delta_t_truth_K:.2f} | "
            f"{p.delta_t_cooper_K:.2f} | {p.rel_error:+.1%} |"
        )
        if has_sites:
            row += f" {_fmt(p.n_sites_prescribed, '.0f')} |"
        if has_vapor:
            row += f" {_fmt(p.vapor_fraction, '.3f')} |"
        row += f" {p.implied_roughness_um:.3g} |"
        lines.append(row)
    lines += [
        "",
        "> 역산 R_p 는 **진단용**입니다. R_p = 1 µm 고정이 원칙이고, 이 값을 맞추려고 "
        "튜닝하면 '유체별 상수 없음' 주장이 깨집니다. 물리적으로 말이 되는 범위"
        "(대략 0.1~10 µm)를 벗어났는지만 보세요.",
    ]

    if gate.confound is not None:
        c = gate.confound
        lines += [
            "",
            "## 사이트 처방 교란 분해",
            "",
            "| 관계 | 지수 | R² |",
            "|---|---|---|",
            f"| q ∝ ΔT^a (정답 비등곡선) | {c.q_vs_delta_t.slope:.3f} | {c.q_vs_delta_t.r_squared:.4f} |",
            f"| N ∝ ΔT^b (처방 사이트) | {c.sites_vs_delta_t.slope:.3f} | {c.sites_vs_delta_t.r_squared:.4f} |",
            f"| q ∝ N^c | {c.q_vs_sites.slope:.3f} | {c.q_vs_sites.r_squared:.4f} |",
            "",
            f"- Cooper 가 가정하는 q ∝ ΔT^{1.0 / gate.slope_target:.2f} 와 비교할 것. "
            "a ≈ b·c 이고 b 는 입력이므로 비등곡선 기울기는 사이트 처방에 묶여 있습니다.",
            "- 사이트당 q'' [W/m²]: "
            + ", ".join(f"{p.case_id} {v:.0f}" for p, v in zip(gate.points, c.q_per_site_W_m2)),
            f"- 조건별 상대오차 ↔ 처방 사이트 수 순위상관 ρ = {c.error_site_spearman:+.2f}",
        ]

    sens = manifest.get("q_definition_sensitivity") or []
    if sens:
        lines += [
            "",
            "## q 정의 민감도",
            "",
            "판정은 `판정에 씀` 표시된 한 정의로만 합니다. 나머지는 결과가 좋은 정의를 골라 쓰지 "
            "않았다는 것을 보이려고 같이 둡니다.",
            "",
            "| 정의 | 역할 | MAPE | ±기준 이내 | 판정 | 판정에 씀 |",
            "|---|---|---|---|---|---|",
        ]
        for r in sens:
            lines.append(
                f"| `{r['q_definition']}` | {r['role']} | {r['mape']:.1%} | "
                f"{r['n_within_band']}/{len(gate.points)} | {r['verdict']} | "
                f"{'✓' if r['used_for_verdict'] else ''} |"
            )

    audits = [
        (case_id, diag) for case_id, diag in manifest.get("wall_flux_diagnostics", {}).items()
        if "verdict" in diag
    ]
    if audits:
        lines += ["", "## 차분 차수 수렴", ""]
    for case_id, diag in audits:
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
            "n_sites_prescribed": p.n_sites_prescribed,
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


# ----------------------------------------------------------------------
# 실측 비등곡선 게이트 산출물
# ----------------------------------------------------------------------
C_MUTED = "#898781"


def _plain_log_ticks(ax) -> None:
    """log 축 눈금을 10^n 대신 그냥 숫자로 (6, 10, 20, 30 …)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(NullFormatter())


def _panel_grid(n: int, ncols: int = 3):
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.5 * nrows + 0.5),
                             facecolor=SURFACE, squeeze=False)
    axes = list(axes.flat)
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def plot_experiment_curves(results, q_ref: float, path: Path, dpi: int) -> None:
    """곡선마다 한 칸: 실측 점(평가 구간 안 = 채운 점, 밖 = 회색 빈 점)과 Cooper 선."""
    fig, axes = _panel_grid(len(results))
    for i, (ax, r) in enumerate(zip(axes, results)):
        c = r.curve
        inside = r.in_window
        ax.plot(c.q[~inside] / 1e3, c.dT[~inside], "o", ms=6, mfc="none", mew=1.2, color=C_MUTED,
                label="Measured (outside window)")
        ax.plot(c.q[inside] / 1e3, c.dT[inside], "o", ms=7, color=C_TRUTH, label="Measured (gate window)")
        qq = np.geomspace(c.q.min(), c.q.max(), 60)
        ax.plot(qq / 1e3, r.model.delta_t_from_q(qq), "-", lw=2, color=C_PRED, label="Cooper (Rp = 1 µm)")
        ax.axvline(q_ref / 1e3, color=INK_MUTED, lw=1.0, ls=":", zorder=0,
                   label=f"CPU operating point ({q_ref / 1e3:.0f} kW/m²)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log_ticks(ax)
        g = r.gate
        stats = (f"slope {g.fit.slope:.2f} · MAPE {g.mape:.0%}" if g else "too few points")
        ax.set_title(f"{c.curve_id}  ({c.p_Pa / 1e5:.2f} bar, {c.orientation_deg:g}°)\n{stats}",
                     color=INK, fontsize=9, loc="left")
        ax.set_xlabel("q'' [kW/m²]", fontsize=8)
        ax.set_ylabel("ΔT_sat [K]", fontsize=8)
        _style(ax)
        if i == 0:
            # 첫 칸만 선에 직접 이름을 붙이고, 나머지는 그림 전체 범례로 읽는다.
            ax.annotate("Cooper", xy=(qq[0] / 1e3, float(r.model.delta_t_from_q(qq[0]))),
                        xytext=(2, 8), textcoords="offset points", ha="left", fontsize=8, color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# 산점도 전체-쌍 검증을 통과하는 앞 세 칸(blue, orange, aqua) + 모양으로 출처를 이중 표기한다.
SOURCE_STYLES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]


def plot_experiment_parity(results, mape_max: float, path: Path, dpi: int) -> None:
    by_source: dict[str, tuple[list, list]] = {}
    for r in results:
        m, p = by_source.setdefault(r.curve.source, ([], []))
        m.extend(r.curve.dT[r.in_window])
        p.extend(r.dT_cooper[r.in_window])
    by_source = {k: v for k, v in by_source.items() if v[0]}
    if not by_source:
        return
    meas = np.concatenate([v[0] for v in by_source.values()])
    pred = np.concatenate([v[1] for v in by_source.values()])
    lo = 0.8 * min(meas.min(), pred.min())
    hi = 1.2 * max(meas.max(), pred.max())
    line = np.array([lo, hi])
    fig, ax = plt.subplots(figsize=(5.2, 5.0), facecolor=SURFACE)
    ax.fill_between(line, line * (1 - mape_max), line * (1 + mape_max), color=GRID, alpha=0.55, lw=0)
    ax.plot(line, line, "-", lw=1.2, color=INK_MUTED, alpha=0.7)
    multi = len(by_source) > 1
    for i, (source, (m, p)) in enumerate(by_source.items()):
        color, marker = SOURCE_STYLES[i % len(SOURCE_STYLES)] if multi else (C_TRUTH, "o")
        ax.plot(m, p, marker, ms=7, color=color, mec=SURFACE, mew=1.0, label=source, ls="none")
    if multi:
        if len(by_source) > len(SOURCE_STYLES):
            log_note = "colours repeat beyond three sources; marker shape still differs"
            ax.text(0.02, 0.02, log_note, transform=ax.transAxes, fontsize=7, color=INK_MUTED)
        leg = ax.legend(frameon=False, fontsize=8, loc="upper left", title="Source", title_fontsize=8)
        for text in leg.get_texts():
            text.set_color(INK_MUTED)
    ax.annotate(f"±{mape_max:.0%} band", xy=(hi * 0.9, hi * 0.9 * (1 - mape_max)), xytext=(0, -12),
                textcoords="offset points", ha="right", fontsize=9, color=INK_MUTED)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured ΔT_sat [K]")
    ax.set_ylabel("Cooper ΔT_sat [K]")
    ax.set_title("Parity — points inside the gate window", color=INK, fontsize=11, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def plot_pressure_trends(trends, q_ref: float, path: Path, dpi: int) -> None:
    if not trends:
        return
    fig, axes = _panel_grid(len(trends), ncols=2)
    for ax, t in zip(axes, trends):
        pr = np.array(t.reduced_pressure)
        ax.plot(pr, t.dT_meas, "o-", ms=7, lw=2, color=C_TRUTH,
                label=rf"Measured  ($\propto p_r^{{{t.fit_meas.slope:.2f}}}$ ± {t.fit_meas.slope_stderr:.2f}, "
                      f"{len(t.curve_ids)} pressures)")
        ax.plot(pr, t.dT_cooper, "s--", ms=6, lw=2, mfc="none", mew=1.5, color=C_PRED,
                label=rf"Cooper  ($\propto p_r^{{{t.fit_cooper.slope:.2f}}}$)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log_ticks(ax)
        ax.set_xlabel("Reduced pressure p_r")
        ax.set_ylabel(f"ΔT_sat at {q_ref / 1e3:.0f} kW/m² [K]")
        ax.set_title(t.group, color=INK, fontsize=9, loc="left")
        leg = ax.legend(frameon=False, fontsize=9)
        for text in leg.get_texts():
            text.set_color(INK_MUTED)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def _f(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) and np.isfinite(v) else "—"


def _experiment_summary(run, cfg, manifest: dict) -> str:
    rows = [r.to_row() for r in run.results]
    counts = run.verdict_counts
    q_min, q_max_frac = run.window
    lines = [
        "# 실측 비등곡선 Cooper 게이트",
        "",
        f"- 입력: {', '.join('`' + p + '`' for p in manifest['inputs'])}",
        f"- 평가 구간: 상승 측정, q ≥ {q_min / 1e3:.0f} kW/m², q ≤ {q_max_frac:.0%} × 곡선 최대 q (판단값, 아래 민감도 표)",
        f"- Cooper: R_p = {cfg.get('cooper.roughness_um')} µm 고정, 곡선마다 그 압력의 p_r",
        f"- 기준: 곡선마다 MAPE ≤ {cfg.get('gate.mape_max'):.0%}, log-log 기울기 {cfg.get('gate.slope_target')} ± "
        f"{cfg.get('gate.slope_tol')}, 구간 안 점 {cfg.get('gate.min_cases')}개 이상. "
        "실측 곡선은 사이트 수가 물리로 정해지므로 기울기도 판정 기준이다.",
        "",
        "## 한눈에",
        "",
        f"- 곡선 {len(run.results)}개 — 통과 {counts.get('통과', 0)} · 조건부 통과 {counts.get('조건부', 0)} · "
        f"불일치 {counts.get('불일치', 0)} · 판정 불가 {counts.get('판정 불가', 0)}",
        f"- 구간 안 전체 점 평균 오차 {_f(run.pooled_mape, '.1%')}, 평균 부호 오차 {_f(run.pooled_bias, '+.1%')} "
        "(+면 Cooper 가 과열도를 높게 냄)",
        f"- {run.q_ref / 1e3:.0f} kW/m² 에서 실측 ΔT / Cooper ΔT: "
        + ", ".join(f"{r.curve.curve_id} {_f(r.dT_meas_at_ref, '.1f')}/{_f(r.dT_cooper_at_ref, '.1f')} K"
                    for r in run.results),
        "",
        "## 곡선별",
        "",
        "| 출처 | 곡선 | 압력 [bar] | p_r | 방향 | 점(구간/전체) | MAPE | 부호 오차 | 기울기 | 판정 | "
        f"ΔT@{run.q_ref / 1e3:.0f}k 실측 | Cooper | 역산 R_p [µm] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in rows:
        lines.append(
            f"| {d['source']} | {d['curve_id']} | {d['p_Pa'] / 1e5:.2f} | {d['reduced_pressure']:.4f} | "
            f"{_f(d['orientation_deg'], 'g')}° | {d['n_window']}/{d['n_points']} | {_f(d['mape'], '.0%')} | "
            f"{_f(d['bias'], '+.0%')} | {_f(d['slope'], '.3f')} ± {_f(d['slope_stderr'], '.3f')} | {d['verdict']} | "
            f"{_f(d['dT_meas_at_ref_K'], '.2f')} | {_f(d['dT_cooper_at_ref_K'], '.2f')} | "
            f"{_f(d['implied_rp_at_ref_um'], '.3g')} |"
        )
    lines += [
        "",
        "> 역산 R_p 는 진단용이다. 이 점을 맞추려면 R_p 가 얼마여야 했는지를 보여 줄 뿐이고, "
        "R_p 는 1 µm 로 고정한다.",
    ]
    if run.trends:
        lines += [
            "",
            "## 압력 항 (같은 표면, 압력만 다른 곡선들)",
            "",
            f"{run.q_ref / 1e3:.0f} kW/m² 에서 ΔT ∝ p_r^m 로 맞춘 지수. 프리팩터는 ΔT 를 평행이동만 시키므로 "
            "m 은 상수로 맞출 수 없는 형태 정보다.",
            "",
            "| 묶음 | 곡선 수 | 실측 m | Cooper m | 실측 R² |",
            "|---|---|---|---|---|",
        ]
        for t in run.trends:
            lines.append(
                f"| {t.group} | {len(t.curve_ids)} | {t.fit_meas.slope:.3f} ± {_f(t.fit_meas.slope_stderr, '.3f')} | "
                f"{t.fit_cooper.slope:.3f} | {t.fit_meas.r_squared:.3f} |"
            )
    if run.sensitivity:
        lines += [
            "",
            "## 평가 구간 민감도",
            "",
            "| q 하한 [kW/m²] | 상한 (× 최대 q) | 판정된 곡선 | 통과 | 조건부 | 불일치 | MAPE 중앙값 | 기울기 중앙값 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in run.sensitivity:
            lines.append(
                f"| {s['q_min_W_m2'] / 1e3:.0f} | {s['q_max_frac']:.2f} | {s['n_judged']}/{s['n_curves']} | "
                f"{s['n_pass']} | {s['n_conditional']} | {s['n_mismatch']} | {_f(s['median_mape'], '.0%')} | "
                f"{_f(s['median_slope'], '.3f')} |"
            )
    if run.sensitivity and run.sensitivity[0].get("per_curve"):
        short = {"통과": "통과", "판정 불가": "—"}
        head = " | ".join(f"{s['q_min_W_m2'] / 1e3:.0f}k–{s['q_max_frac']:.1f}" for s in run.sensitivity)
        lines += [
            "",
            "### 곡선별 — 구간을 바꿔도 판정이 같은가",
            "",
            "칸: 기울기 / 부호 오차 / 판정 (조건부 = 형태 일치·크기 이탈). 기울기 판정이 구간마다 뒤집히는 곡선은 "
            "한 개의 거듭제곱(Cooper 형태)으로 설명되지 않고 휘어 있다는 뜻이다. 부호 오차의 방향과 크기가 "
            "구간과 무관하면 그쪽이 견고한 결론이다.",
            "",
            f"| 곡선 | {head} |",
            "|---|" + "---|" * len(run.sensitivity),
        ]
        for r in run.results:
            cells = []
            for s in run.sensitivity:
                d = s["per_curve"].get(r.curve.key, {})
                v = d.get("verdict", "—")
                v = short.get(v, "조건부" if v.startswith("조건부") else "불일치" if v.startswith("불일치") else v)
                cells.append(f"{_f(d.get('slope', float('nan')), '.2f')} / {_f(d.get('bias', float('nan')), '+.0%')} / {v}")
            lines.append(f"| {r.curve.curve_id} | " + " | ".join(cells) + " |")
    lines += ["", "## 재현 정보", "", "```json",
              json.dumps(manifest.get("provenance", {}), indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def write_experiment_outputs(out_dir: Path, run, cfg, manifest: dict,
                             make_plots: bool = True, dpi: int = 160) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    curves_path = out_dir / "curves.csv"
    pd.DataFrame([r.to_row() for r in run.results]).to_csv(curves_path, index=False)
    written["curves"] = str(curves_path)

    pts = []
    for r in run.results:
        c = r.curve
        for i in range(c.q.size):
            pts.append({
                "source": c.source, "curve_id": c.curve_id, "p_Pa": c.p_Pa, "branch": c.branch[i],
                "q_W_m2": c.q[i], "dT_meas_K": c.dT[i], "dT_cooper_K": r.dT_cooper[i],
                "rel_error": (r.dT_cooper[i] - c.dT[i]) / c.dT[i], "in_window": bool(r.in_window[i]),
            })
    points_path = out_dir / "points.csv"
    pd.DataFrame(pts).to_csv(points_path, index=False)
    written["points"] = str(points_path)

    result_path = out_dir / "experiment_result.json"
    result_path.write_text(json.dumps({
        "verdict_counts": run.verdict_counts,
        "pooled_mape": run.pooled_mape,
        "pooled_bias": run.pooled_bias,
        "q_ref_W_m2": run.q_ref,
        "window": {"q_min_W_m2": run.window[0], "q_max_frac": run.window[1]},
        "curves": [{**r.to_row(), "gate": r.gate.to_dict() if r.gate else None} for r in run.results],
        "pressure_trends": [t.to_dict() for t in run.trends],
        "window_sensitivity": run.sensitivity,
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    written["experiment_result"] = str(result_path)

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    written["run_manifest"] = str(manifest_path)

    summary_path = out_dir / "summary.md"
    summary_path.write_text(_experiment_summary(run, cfg, manifest), encoding="utf-8")
    written["summary"] = str(summary_path)

    if make_plots:
        p = out_dir / "boiling_curves.png"
        plot_experiment_curves(run.results, run.q_ref, p, dpi)
        written["boiling_curves"] = str(p)
        p = out_dir / "parity.png"
        plot_experiment_parity(run.results, float(cfg.get("gate.mape_max", 0.30)), p, dpi)
        if p.exists():
            written["parity"] = str(p)
        p = out_dir / "pressure_trend.png"
        plot_pressure_trends(run.trends, run.q_ref, p, dpi)
        if p.exists():
            written["pressure_trend"] = str(p)
    return written
