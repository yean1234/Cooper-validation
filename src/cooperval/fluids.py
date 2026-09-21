"""유체 물성.

주의 — 여기 숫자는 **문헌 대표값**이고 BubbleML 시뮬레이션이 실제로 쓴 값이 아닐 수
있다. Flash-X par 파일 / HDF5 의 ``real-runtime-params`` 를 확인해서 덮어쓰는 것이
원칙이다 (``configs/fluids.yaml`` 에서 override 가능).
어떤 값을 실제로 썼는지는 run manifest 에 그대로 기록된다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import yaml

__all__ = ["Fluid", "FLUID_LIBRARY", "get_fluid", "load_fluid_library"]

P_ATM = 101325.0  # Pa


@dataclass(frozen=True)
class Fluid:
    name: str
    molar_mass_g_mol: float     # M  [g/mol]  — Cooper 식은 g/mol 을 요구
    p_critical_Pa: float        # p_c [Pa]
    T_sat_K: float              # 작동 압력에서의 포화온도
    k_liquid: float             # k_l [W/m/K]
    k_vapor: float              # k_v [W/m/K]
    rho_liquid: float           # [kg/m^3]
    rho_vapor: float            # [kg/m^3]
    cp_liquid: float            # [J/kg/K]
    mu_liquid: float            # [Pa s]
    h_fg: float                 # 잠열 [J/kg]
    sigma: float                # 표면장력 [N/m]
    p_operating_Pa: float = P_ATM
    source: str = "문헌 대표값 (미검증)"

    @property
    def reduced_pressure(self) -> float:
        """p_r = p / p_c — Cooper 식의 핵심 입력."""
        return self.p_operating_Pa / self.p_critical_Pa

    @property
    def prandtl_liquid(self) -> float:
        return self.mu_liquid * self.cp_liquid / self.k_liquid

    @property
    def capillary_length(self) -> float:
        """l_0 = sqrt(sigma / (g (rho_l - rho_v))) — BubbleML 의 기준 길이 후보."""
        g = 9.81
        return (self.sigma / (g * (self.rho_liquid - self.rho_vapor))) ** 0.5

    def replace(self, **kwargs) -> "Fluid":
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["reduced_pressure"] = self.reduced_pressure
        d["capillary_length_m"] = self.capillary_length
        return d


# BubbleML pool boiling 의 작동유체. 값은 상온~포화 근방 대표값.
FC72 = Fluid(
    name="FC-72",
    molar_mass_g_mol=338.04,
    p_critical_Pa=1.83e6,
    T_sat_K=329.75,      # 56.6 C @ 1 atm
    k_liquid=0.0545,
    k_vapor=0.0130,
    rho_liquid=1602.0,
    rho_vapor=13.4,
    cp_liquid=1100.0,
    mu_liquid=4.40e-4,
    h_fg=88000.0,
    sigma=0.0084,
    source="3M FC-72 product data / 비등 문헌 대표값 (BubbleML par 파일로 교차확인 필요)",
)

# 물 — Cooper 식 구현의 위생 검사(sanity check)용. 1 atm, 100 kW/m^2 에서
# ΔT ~ 10 K 근처가 나와야 한다.
WATER = Fluid(
    name="water",
    molar_mass_g_mol=18.015,
    p_critical_Pa=22.064e6,
    T_sat_K=373.15,
    k_liquid=0.679,
    k_vapor=0.0248,
    rho_liquid=958.4,
    rho_vapor=0.5978,
    cp_liquid=4217.0,
    mu_liquid=2.82e-4,
    h_fg=2.257e6,
    sigma=0.0589,
    source="IAPWS 근사 대표값",
)

FLUID_LIBRARY: dict[str, Fluid] = {f.name.lower(): f for f in (FC72, WATER)}


def load_fluid_library(path) -> None:
    """YAML 로 물성을 덮어쓰거나 새 유체를 등록한다."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    for name, fields in (data.get("fluids") or {}).items():
        key = name.lower()
        if key in FLUID_LIBRARY:
            FLUID_LIBRARY[key] = FLUID_LIBRARY[key].replace(**fields)
        else:
            FLUID_LIBRARY[key] = Fluid(name=name, **fields)


def get_fluid(name: str) -> Fluid:
    try:
        return FLUID_LIBRARY[name.lower()]
    except KeyError:
        raise KeyError(
            f"등록되지 않은 유체 {name!r}. 등록된 것: {sorted(FLUID_LIBRARY)}"
        ) from None
