import numpy as np
import pytest

from cooperval.heaviside import liquid_fraction, mix_conductivity, smoothed_heaviside


def test_heaviside_is_bounded_and_monotonic():
    phi = np.linspace(-3.0, 3.0, 201)
    h = smoothed_heaviside(phi, eps=1.0)
    assert h.min() >= 0.0 and h.max() <= 1.0
    assert np.all(np.diff(h) >= -1e-12)
    assert h[0] == 0.0 and h[-1] == 1.0
    assert smoothed_heaviside(np.array([0.0]), 1.0)[0] == pytest.approx(0.5)


def test_sign_convention_flips():
    phi = np.array([-1.0, 1.0])
    assert liquid_fraction(phi, 0.1, "liquid") == pytest.approx([0.0, 1.0])
    assert liquid_fraction(phi, 0.1, "vapor") == pytest.approx([1.0, 0.0])
    with pytest.raises(ValueError):
        liquid_fraction(phi, 0.1, "plasma")


def test_mixing_rules_bracket_the_pure_phases():
    alpha = np.array([0.0, 0.5, 1.0])
    for rule in ("arithmetic", "harmonic"):
        k = mix_conductivity(alpha, 0.0545, 0.0130, rule)
        assert k[0] == pytest.approx(0.0130)
        assert k[2] == pytest.approx(0.0545)
        assert 0.0130 < k[1] < 0.0545
    # 직렬 전도인 harmonic 은 산술평균보다 항상 작다
    a = mix_conductivity(np.array([0.5]), 0.0545, 0.0130, "arithmetic")[0]
    h = mix_conductivity(np.array([0.5]), 0.0545, 0.0130, "harmonic")[0]
    assert h < a
