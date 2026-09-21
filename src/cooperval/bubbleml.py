"""BubbleML HDF5 로더 + 무차원 → 유차원 환산.

BubbleML 은 Flash-X 비압축성 다상 솔버 결과를 모은 데이터셋이고, 필드는 보통
``temperature``, ``dfun``(level-set φ), ``velx``, ``vely``, ``pressure``, ``x``, ``y``
가 ``(n_t, n_y, n_x)`` 모양으로 들어 있다.

**여기가 이 파이프라인에서 가장 가정이 많은 부분이다.** 값들이 무차원으로 저장돼
있기 때문에, 온도·길이 스케일을 무엇으로 되돌리느냐에 따라 q'' 가 통째로 바뀐다.
그래서 이 모듈은
  1) 파일에 들어 있는 runtime params 를 먼저 읽고,
  2) 없으면 설정(case_table)을 쓰고,
  3) 그것도 없으면 파일명에서 Twall 을 긁고,
  4) 어느 경로로 정해졌는지를 ``provenance`` 에 남긴다.
추측이 섞였으면 경고를 띄운다. 조용히 넘어가지 않는다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from .config import Config
from .fluids import Fluid

log = logging.getLogger(__name__)

__all__ = ["CaseMeta", "CaseData", "discover_cases", "load_case"]

KELVIN_OFFSET = 273.15


@dataclass
class CaseMeta:
    case_id: str
    path: str
    T_wall_K: float
    T_sat_K: float
    T_bulk_K: float
    provenance: dict = field(default_factory=dict)

    @property
    def delta_t_sup_truth(self) -> float:
        """정답 벽 과열도 ΔT_sup = T_wall - T_sat.

        BubbleML 히터는 Dirichlet 경계라 벽 온도가 입력값으로 정확히 주어진다.
        그래서 이 값이 게이트의 '정답'이 된다 — 우리가 추정한 수가 아니다.
        """
        return self.T_wall_K - self.T_sat_K

    @property
    def delta_t_sub(self) -> float:
        """서브쿨 ΔT_sub = T_sat - T_bulk (포화 조건이면 0)."""
        return self.T_sat_K - self.T_bulk_K

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "path": self.path,
            "T_wall_K": self.T_wall_K,
            "T_sat_K": self.T_sat_K,
            "T_bulk_K": self.T_bulk_K,
            "delta_t_sup_truth_K": self.delta_t_sup_truth,
            "delta_t_sub_K": self.delta_t_sub,
            "provenance": self.provenance,
        }


@dataclass
class CaseData:
    meta: CaseMeta
    temperature: np.ndarray   # (n_t, n_y, n_x) [K]
    dfun: np.ndarray          # (n_t, n_y, n_x) level-set, 길이 단위는 무관(부호만 씀)
    x: np.ndarray             # (n_x,) [m]
    y: np.ndarray             # (n_y,) [m]
    dx: float
    dy: float
    runtime_params: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.temperature.shape[0]


# ----------------------------------------------------------------------
# 케이스 탐색
# ----------------------------------------------------------------------
def discover_cases(cfg: Config) -> list[Path]:
    root = Path(cfg.get("data.root"))
    explicit = cfg.get("data.cases")
    if explicit:
        paths = [root / c if not Path(c).is_absolute() else Path(c) for c in explicit]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"설정에 적힌 케이스 파일이 없습니다: {missing}")
        return paths
    paths = sorted(root.glob(cfg.get("data.glob")))
    if not paths:
        raise FileNotFoundError(
            f"{root}/{cfg.get('data.glob')} 에 해당하는 파일이 없습니다. "
            "scripts/make_synthetic.py 로 합성 데이터를 먼저 만들어 보세요."
        )
    return paths


# ----------------------------------------------------------------------
# runtime params
# ----------------------------------------------------------------------
def _read_runtime_params(h5: h5py.File) -> dict:
    """Flash-X 가 남긴 runtime params 를 최대한 긁어 온다 (형식이 파일마다 다름)."""
    params: dict = {}
    for key in ("real-runtime-params", "int-runtime-params", "string-runtime-params",
                "real runtime parameters", "integer runtime parameters"):
        if key not in h5:
            continue
        try:
            for row in h5[key][()]:
                try:
                    name, value = row[0], row[1]
                except (IndexError, TypeError):
                    continue
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "ignore")
                if isinstance(value, bytes):
                    value = value.decode("utf-8", "ignore")
                params[str(name).strip()] = value
        except Exception as exc:  # 형식이 예상과 다르면 조용히 건너뛴다
            log.debug("runtime params %s 파싱 실패: %s", key, exc)
    return params


def _coord_axis(arr: np.ndarray, axis: str) -> np.ndarray:
    """(n_t, n_y, n_x) 또는 (n_y, n_x) 또는 (n,) 좌표 배열을 1D 로 줄인다."""
    a = np.asarray(arr)
    while a.ndim > 2:
        a = a[0]
    if a.ndim == 2:
        a = a[0, :] if axis == "x" else a[:, 0]
    return np.asarray(a, dtype=float).ravel()


# ----------------------------------------------------------------------
# 온도 스케일 복원
# ----------------------------------------------------------------------
def _infer_temperature_mode(theta: np.ndarray) -> str:
    lo, hi = float(np.nanmin(theta)), float(np.nanmax(theta))
    if hi > 200.0:
        return "kelvin"
    if hi > 40.0:
        return "celsius"
    return "nondimensional"


def _resolve_case_temperatures(
    case_id: str, path: Path, cfg: Config, fluid: Fluid, runtime: dict
) -> CaseMeta:
    """T_wall / T_sat / T_bulk 를 정하고 출처를 기록한다."""
    prov: dict = {}
    table = (cfg.get("nondim.case_table") or {}).get(case_id, {}) or {}

    def _pick(name_c: str, runtime_keys: tuple[str, ...], fallback: float | None):
        if name_c in table:
            prov[name_c] = "config:case_table"
            return float(table[name_c]) + KELVIN_OFFSET
        for rk in runtime_keys:
            if rk in runtime:
                try:
                    prov[name_c] = f"hdf5:{rk}"
                    return float(runtime[rk])
                except (TypeError, ValueError):
                    continue
        if fallback is not None:
            prov[name_c] = "fallback"
            return fallback
        return None

    T_sat = _pick("T_sat_C", ("tsat", "sim_tsat", "t_sat"), fluid.T_sat_K)
    T_wall = _pick("T_wall_C", ("twall_high", "twall", "sim_twall"), None)

    if T_wall is None:
        pattern = cfg.get("nondim.filename_twall_regex")
        m = re.search(pattern, path.stem)
        if m:
            T_wall = float(m.group(1)) + KELVIN_OFFSET
            prov["T_wall_C"] = f"filename:{m.group(0)}"
        else:
            raise ValueError(
                f"[{case_id}] 벽 온도를 알아낼 수 없습니다. "
                "configs 의 nondim.case_table 에 T_wall_C 를 적어주세요. "
                "이 값이 게이트의 '정답 ΔT' 이므로 추측하면 안 됩니다."
            )

    T_bulk = _pick("T_bulk_C", ("tbulk", "sim_tbulk", "t_bulk"), T_sat)

    if T_wall <= T_sat:
        raise ValueError(
            f"[{case_id}] T_wall({T_wall:.2f} K) <= T_sat({T_sat:.2f} K) — "
            "과열도가 0 이하입니다. 단위(섭씨/켈빈)를 확인하세요."
        )

    meta = CaseMeta(
        case_id=case_id,
        path=str(path),
        T_wall_K=T_wall,
        T_sat_K=T_sat,
        T_bulk_K=T_bulk,
        provenance=prov,
    )
    guessed = [k for k, v in prov.items() if v in ("fallback",) or v.startswith("filename")]
    if guessed:
        log.warning(
            "[%s] 온도 기준 중 %s 는 파일에서 읽은 값이 아니라 추정입니다 (%s). "
            "Flash-X par 파일로 확인하세요.",
            case_id, guessed, {k: prov[k] for k in guessed},
        )
    return meta


def _dimensionalize_temperature(theta: np.ndarray, meta: CaseMeta, cfg: Config) -> np.ndarray:
    mode = cfg.get("nondim.temperature_mode", "auto")
    if mode == "auto":
        mode = _infer_temperature_mode(theta)
        log.info("[%s] 온도 필드 모드 자동판정: %s", meta.case_id, mode)
    if mode == "kelvin":
        return theta.astype(float)
    if mode == "celsius":
        return theta.astype(float) + KELVIN_OFFSET
    if mode != "nondimensional":
        raise ValueError(f"알 수 없는 temperature_mode {mode!r}")

    convention = cfg.get("nondim.temperature_convention")
    if convention == "theta_bulk_wall":
        # theta = (T - T_bulk) / (T_wall - T_bulk)
        return meta.T_bulk_K + theta.astype(float) * (meta.T_wall_K - meta.T_bulk_K)
    if convention == "theta_sat_wall":
        # theta = (T - T_sat) / (T_wall - T_sat)
        return meta.T_sat_K + theta.astype(float) * (meta.T_wall_K - meta.T_sat_K)
    raise ValueError(f"알 수 없는 temperature_convention {convention!r}")


def _dimensionalize_length(coord: np.ndarray, cfg: Config, fluid: Fluid, case_id: str):
    mode = cfg.get("nondim.length_mode", "auto")
    span = float(np.nanmax(coord) - np.nanmin(coord))
    scale = cfg.get("nondim.length_scale_m") or fluid.capillary_length
    if mode == "auto":
        # 도메인 폭이 1 이상이면 무차원(모세관 길이 단위), 0.1 m 미만이면 이미 미터.
        mode = "nondimensional" if span > 0.5 else "meters"
        log.info("[%s] 길이 모드 자동판정: %s (도메인 폭 %.4g)", case_id, mode, span)
    if mode == "meters":
        return coord.astype(float), 1.0
    if mode == "nondimensional":
        return coord.astype(float) * scale, scale
    raise ValueError(f"알 수 없는 length_mode {mode!r}")


# ----------------------------------------------------------------------
def load_case(path: str | Path, cfg: Config, fluid: Fluid) -> CaseData:
    path = Path(path)
    case_id = path.stem
    fields = cfg.get("data.fields")

    with h5py.File(path, "r") as h5:
        runtime = _read_runtime_params(h5)
        for logical, name in (("temperature", fields["temperature"]), ("dfun", fields["dfun"])):
            if name not in h5:
                raise KeyError(
                    f"[{case_id}] 필드 {name!r}({logical}) 이 없습니다. "
                    f"파일에 있는 키: {sorted(h5.keys())[:20]}"
                )
        theta = np.asarray(h5[fields["temperature"]][()], dtype=float)
        phi = np.asarray(h5[fields["dfun"]][()], dtype=float)
        x_raw = _coord_axis(h5[fields["x"]][()], "x") if fields["x"] in h5 else None
        y_raw = _coord_axis(h5[fields["y"]][()], "y") if fields["y"] in h5 else None

    if theta.ndim != 3:
        raise ValueError(f"[{case_id}] 온도 필드가 (n_t, n_y, n_x) 가 아닙니다: {theta.shape}")
    n_t, n_y, n_x = theta.shape

    meta = _resolve_case_temperatures(case_id, path, cfg, fluid, runtime)
    temperature = _dimensionalize_temperature(theta, meta, cfg)

    # 시간 프레임 자르기: 초기 과도 구간은 정상상태가 아니라서 버린다.
    skip = float(cfg.get("data.skip_initial_fraction") or 0.0)
    start = int(round(n_t * skip))
    stride = max(1, int(cfg.get("data.time_stride") or 1))
    sl = slice(start, None, stride)
    temperature = temperature[sl]
    phi = phi[sl]
    max_frames = cfg.get("data.max_frames")
    if max_frames:
        temperature = temperature[: int(max_frames)]
        phi = phi[: int(max_frames)]
    if temperature.shape[0] == 0:
        raise ValueError(f"[{case_id}] 시간 프레임을 전부 잘라냈습니다 — skip_initial_fraction 확인")

    # 좌표
    if x_raw is None or x_raw.size != n_x:
        x_raw = np.arange(n_x, dtype=float)
    if y_raw is None or y_raw.size != n_y:
        y_raw = np.arange(n_y, dtype=float)
    x_m, _ = _dimensionalize_length(x_raw, cfg, fluid, case_id)
    y_m, length_scale = _dimensionalize_length(y_raw, cfg, fluid, case_id)

    dx = float(cfg.get("geometry.dx_m") or np.mean(np.diff(x_m)))
    dy = float(cfg.get("geometry.dy_m") or np.mean(np.diff(y_m)))
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dy) or dy <= 0:
        raise ValueError(f"[{case_id}] 격자 간격이 이상합니다: dx={dx}, dy={dy}")

    meta.provenance.update(
        {
            "length_scale_m": length_scale,
            "dx_m": dx,
            "dy_m": dy,
            "n_frames_used": int(temperature.shape[0]),
            "n_frames_total": int(n_t),
            "runtime_params_found": bool(runtime),
        }
    )

    return CaseData(
        meta=meta,
        temperature=temperature,
        dfun=phi,
        x=x_m,
        y=y_m,
        dx=dx,
        dy=dy,
        runtime_params=runtime,
    )
