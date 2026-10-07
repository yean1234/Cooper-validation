"""설정 로딩 — YAML 을 기본값 위에 깊은 병합(deep-merge)한다.

설계 원칙: **추측한 값은 전부 설정으로 밖에 꺼내 놓는다.** 코드 안에 숨은 상수를
두지 않아야 나중에 "이 수는 어디서 나왔나"에 답할 수 있다.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "load_config", "DEFAULTS"]

DEFAULTS: dict[str, Any] = {
    "run": {
        "out_dir": "outputs",
        "run_id": None,          # None → 타임스탬프
        "label": "cooper-gate-draft",
    },
    "data": {
        "root": "data/bubbleml",
        "glob": "*.hdf5",
        "cases": None,           # None → glob 결과 전부
        "fields": {
            "temperature": "temperature",
            "dfun": "dfun",
            "x": "x",
            "y": "y",
            "velx": "velx",
            "vely": "vely",
            "pressure": "pressure",
        },
        "skip_initial_fraction": 0.3,   # 초기 과도(transient) 버리기
        "time_stride": 1,
        "max_frames": None,
    },
    "geometry": {
        "wall": "bottom",               # bottom | top
        "dx_m": None,                   # None → 좌표 배열에서 추정
        "dy_m": None,
        "heater_x_range_m": None,       # None → 벽 전체를 히터로 본다
    },
    "nondim": {
        "temperature_mode": "auto",     # auto | nondimensional | kelvin | celsius
        "temperature_convention": "theta_bulk_wall",  # theta = (T-T_bulk)/(T_wall-T_bulk)
        "length_mode": "auto",          # auto | nondimensional | meters
        "length_scale_m": None,         # None → 유체의 모세관 길이 사용
        "case_table": {},               # case_id: {T_wall_C, T_sat_C, T_bulk_C}
        "filename_twall_regex": r"[Tt]wall[-_]?(\d+(?:\.\d+)?)",
    },
    "phase": {
        "phi_positive_in": "liquid",    # dfun 부호 규약
        "eps_cells": 1.5,               # smoothed Heaviside 폭 = eps_cells * dy
        "mixing_rule": "arithmetic",    # arithmetic | harmonic
    },
    "wall_flux": {
        "order": 2,                     # 기준 차분 차수 (9/20 결론: 2차로 못 박는다)
        "audit_orders": [1, 2, 3],      # 수렴 확인용
        "include_mixed_cells": True,    # False 면 9/20 이전의 (편향된) 방식 재현
    },
    "coarsen": {
        "dx_m": 0.002,                  # 거친 격자 2 mm
        "aggregate": "heater_mean",     # heater_mean | per_coarse_cell
    },
    "cooper": {
        "fluid": "FC-72",
        "roughness_um": 1.0,            # 고정. 튜닝 금지.
        "allow_tuning": False,
        "fluid_overrides": None,        # configs/fluids.yaml 경로
    },
    "gate": {
        "mape_max": 0.30,               # Cooper 문헌 산포 ±30~40% 에 맞춘 값
        "slope_target": 0.33,
        "slope_tol": 0.05,
        "min_cases": 5,
        # 기울기 기준의 역할. auto → 케이스별 처방 사이트 수가 주어지고 조건마다 다르면
        # 'reference'(참고만, 판정 제외), 아니면 'criterion'. 사이트 수가 입력인 자료에서
        # 기울기는 상관식이 아니라 사이트 처방을 잰다 (정답 기포장 보고서 8쪽).
        "slope_role": "auto",           # auto | criterion | reference
    },
    "ground_truth": {
        "csv": None,                    # 정답 기포장 인계 표. 주면 HDF5 대신 이 표의 q'' 를 쓴다
        "q_definition": "arith",        # arith(기본) | harm(동등 후보) | kl | legacy (참고)
    },
    "wall_function": {
        "enabled": False,               # 10월 정식 게이트용. 9/21 일정에서는 끈다.
        "model": "kader",
    },
    "report": {
        "make_plots": True,
        "dpi": 160,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Config:
    raw: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS))
    source_path: str | None = None

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """``cfg.get('gate.mape_max')`` 형태의 접근."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return copy.deepcopy(self.raw)

    def dumps(self) -> str:
        return json.dumps(self.raw, indent=2, ensure_ascii=False, default=str)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    raw = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config(raw=raw, source_path=str(path) if path else None)
