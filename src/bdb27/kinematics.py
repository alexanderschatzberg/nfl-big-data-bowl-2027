"""Per-attempt movement features from a 10 Hz (t, x, y, s) trace.

Units follow the source data: yards, seconds, yards/second. The provided `a`
column is an unsigned magnitude and `dis` is unreliable in the Combine file, so
signed acceleration and path length are derived here from `s`, `x` and `y`.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

MIN_FRAMES = 15
VALLEY_PROMINENCE = 1.0  # yd/s; a speed dip this deep counts as a change of direction
BREAK_MIN_DEPTH = 1.5  # yd/s; the deepest dip must be at least this to be "the break"
TURN_RATE_MIN = np.deg2rad(30)  # rad/s; frames turning faster than this are "in a curve"

FEATURES = [
    "duration", "path_len", "path_eff", "vmax", "acc_max", "dec_max",
    "accel_1to3", "accel_2to5",
    "brk_v_in", "brk_v_min", "brk_dec", "brk_stop_eff", "brk_reacc_05", "brk_reacc_10",
    "brk_time_under", "brk_angle",
    "cod_n", "cod_rate", "cod_reacc", "cod_peak_speed",
    "curve_speed", "lat_acc",
    "f40_vmax", "f40_tau", "f40_a0",
]


def _smooth(v: np.ndarray) -> np.ndarray:
    return savgol_filter(v, 5, 2) if len(v) >= 5 else v


def _first_crossing(t: np.ndarray, s: np.ndarray, level: float) -> float:
    """Time of the first upward crossing of `level`, linearly interpolated."""
    above = s >= level
    idx = np.flatnonzero(~above[:-1] & above[1:])
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    frac = (level - s[i]) / (s[i + 1] - s[i])
    return t[i] + frac * (t[i + 1] - t[i])


def _ramp_accel(t: np.ndarray, s: np.ndarray, lo: float, hi: float) -> float:
    """Mean acceleration (yd/s^2) between first reaching `lo` and then `hi`.

    Clips start at arbitrary points in the first step, so this is anchored on
    speed levels rather than on the first frame.
    """
    t_lo = _first_crossing(t, s, lo)
    if np.isnan(t_lo):
        return np.nan
    later = t >= t_lo
    t_hi = _first_crossing(t[later], s[later], hi) if later.sum() > 1 else np.nan
    if np.isnan(t_hi) or t_hi <= t_lo:
        return np.nan
    return (hi - lo) / (t_hi - t_lo)


def _mono_exp(t, vmax, tau, t0):
    return vmax * (1.0 - np.exp(-np.clip(t - t0, 0, None) / tau))


def sprint_fit(t: np.ndarray, s: np.ndarray) -> dict:
    """Fit v(t) = vmax * (1 - exp(-(t - t0) / tau)) to a straight-line sprint."""
    out = {"f40_vmax": np.nan, "f40_tau": np.nan, "f40_a0": np.nan}
    end = int(np.argmax(s)) + 1  # ignore deceleration after the finish
    if end < MIN_FRAMES:
        return out
    try:
        (vmax, tau, _), _ = curve_fit(
            _mono_exp, t[:end], s[:end], p0=[s.max() * 1.05, 1.0, t[0] - 0.3],
            bounds=([s.max() * 0.9, 0.2, t[0] - 3.0], [s.max() * 1.5, 3.0, t[0] + 1.0]),
        )
    except (RuntimeError, ValueError):
        return out
    return {"f40_vmax": vmax, "f40_tau": tau, "f40_a0": vmax / tau}


def attempt_features(t, x, y, s, sprint: bool = False) -> dict | None:
    t, x, y, s = (np.asarray(v, dtype=float) for v in (t, x, y, s))
    if len(t) < MIN_FRAMES:
        return None
    t = t - t[0]
    x, y, s = _smooth(x), _smooth(y), np.clip(_smooth(s), 0, None)
    acc = np.gradient(s, t)
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    step = np.hypot(np.diff(x), np.diff(y))
    path_len = step.sum()

    f = dict.fromkeys(FEATURES, np.nan)
    f["duration"] = t[-1]
    f["path_len"] = path_len
    f["path_eff"] = np.hypot(x[-1] - x[0], y[-1] - y[0]) / path_len if path_len > 0 else np.nan
    f["vmax"] = s.max()
    f["acc_max"] = acc.max()
    f["dec_max"] = -acc.min()
    f["accel_1to3"] = _ramp_accel(t, s, 1.0, 3.0)
    f["accel_2to5"] = _ramp_accel(t, s, 2.0, 5.0)

    # The break: the deepest interior speed valley, measured against the lower
    # of the speed peaks on either side of it.
    run_max = np.maximum.accumulate(s)
    rev_max = np.maximum.accumulate(s[::-1])[::-1]
    depth = np.minimum(run_max, rev_max) - s
    ib = int(np.argmax(depth))
    if depth[ib] >= BREAK_MIN_DEPTH:
        ipk = int(np.argmax(s[: ib + 1]))
        v_in = s[ipk]
        stop_dist = step[ipk:ib].sum()
        f["brk_v_in"] = v_in
        f["brk_v_min"] = s[ib]
        f["brk_dec"] = -acc[ipk : ib + 1].min()
        # Constant-deceleration equivalent: v^2 = 2 * a * d.
        f["brk_stop_eff"] = (v_in**2 - s[ib] ** 2) / (2 * stop_dist) if stop_dist > 0 else np.nan
        for name, lag in (("brk_reacc_05", 5), ("brk_reacc_10", 10)):
            if ib + lag < len(s):
                f[name] = s[ib + lag] - s[ib]
        lo, hi = ib, ib
        while lo > 0 and s[lo - 1] < 0.5 * v_in:
            lo -= 1
        while hi < len(s) - 1 and s[hi + 1] < 0.5 * v_in:
            hi += 1
        f["brk_time_under"] = t[hi] - t[lo]
        if ib - 8 >= 0 and ib + 8 < len(s):
            a_in = np.arctan2(y[ib - 3] - y[ib - 8], x[ib - 3] - x[ib - 8])
            a_out = np.arctan2(y[ib + 8] - y[ib + 3], x[ib + 8] - x[ib + 3])
            f["brk_angle"] = np.degrees(np.abs(np.angle(np.exp(1j * (a_out - a_in)))))

    # Repeated changes of direction (mirror, wave, bag drills).
    valleys, _ = find_peaks(-s, prominence=VALLEY_PROMINENCE)
    f["cod_n"] = len(valleys)
    f["cod_rate"] = len(valleys) / t[-1] if t[-1] > 0 else np.nan
    if len(valleys):
        gains = [s[v + 5] - s[v] for v in valleys if v + 5 < len(s)]
        f["cod_reacc"] = np.mean(gains) if gains else np.nan
        peaks, _ = find_peaks(s, prominence=VALLEY_PROMINENCE)
        f["cod_peak_speed"] = s[peaks].mean() if len(peaks) else np.nan

    # Speed carried through curves. Heading is only defined when moving.
    moving = s > 2.0
    if moving.sum() >= 5:
        heading = np.unwrap(np.arctan2(vy, vx))
        omega = np.abs(np.gradient(heading, t))
        turning = moving & (omega > TURN_RATE_MIN)
        if turning.sum() >= 3:
            f["curve_speed"] = s[turning].mean()
        f["lat_acc"] = np.percentile((s * omega)[moving], 95)

    if sprint:
        f.update(sprint_fit(t, s))
    return f
