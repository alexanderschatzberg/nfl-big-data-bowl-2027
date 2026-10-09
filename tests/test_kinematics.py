import numpy as np
import pytest

from bdb27.kinematics import attempt_features, sprint_fit

DT = 0.1


def _trace(speed, heading):
    """Integrate a speed/heading profile into (t, x, y, s)."""
    t = np.arange(len(speed)) * DT
    x = np.cumsum(speed * np.cos(heading)) * DT
    y = np.cumsum(speed * np.sin(heading)) * DT
    return t, x, y, speed


def test_too_short_returns_none():
    z = np.zeros(5)
    assert attempt_features(z, z, z, z) is None


def test_straight_sprint_recovers_exponential_parameters():
    t = np.arange(50) * DT
    s = 10.0 * (1 - np.exp(-(t + 0.2) / 1.2))
    fit = sprint_fit(t, s)
    assert fit["f40_vmax"] == pytest.approx(10.0, rel=0.03)
    assert fit["f40_tau"] == pytest.approx(1.2, rel=0.05)
    f = attempt_features(*_trace(s, np.zeros(50)), sprint=True)
    assert f["path_eff"] == pytest.approx(1.0, abs=1e-6)
    assert np.isnan(f["brk_dec"])  # no speed valley on a straight sprint
    assert f["cod_n"] == 0


def test_stop_and_go_break_is_measured():
    # 0 -> 8 yd/s over 2 s, brake to 0 at 8 yd/s^2, turn 180 degrees, re-accelerate at 4 yd/s^2.
    up = np.linspace(0, 8, 21)
    down = np.linspace(8, 0, 11)[1:]
    out = np.linspace(0, 6, 16)[1:]
    speed = np.concatenate([up, down, out])
    heading = np.concatenate([np.zeros(len(up) + len(down)), np.full(len(out), np.pi)])
    f = attempt_features(*_trace(speed, heading))
    assert f["brk_v_in"] == pytest.approx(8.0, abs=0.3)
    assert f["brk_dec"] == pytest.approx(8.0, rel=0.15)
    assert f["brk_stop_eff"] == pytest.approx(8.0, rel=0.2)
    assert f["brk_reacc_05"] == pytest.approx(2.0, abs=0.4)
    assert f["brk_angle"] == pytest.approx(180, abs=5)
    assert f["cod_n"] == 1
    assert f["accel_2to5"] == pytest.approx(4.0, rel=0.1)


def test_circle_reports_centripetal_acceleration():
    n, v, radius = 80, 6.0, 5.0
    heading = np.arange(n) * DT * v / radius
    f = attempt_features(*_trace(np.full(n, v), heading))
    assert f["curve_speed"] == pytest.approx(v, rel=0.05)
    assert f["lat_acc"] == pytest.approx(v**2 / radius, rel=0.1)
