#!/usr/bin/env python3
"""
motec_distance.py — computes the distance covered in MoTeC .ld telemetry logs.

MoTeC's "Corr Dist" does not exist inside the .ld file (i2 computes it with maths
channels after loading), so we compute it ourselves: integrate speed over time
using the trapezoid rule.

Speed sources, in order of trust:
  GPS (what MoTeC does)  ->  front wheels (most trustworthy, undriven)
  ->  rear wheels for corroboration only (driven wheels over-read, and they
      spin on a dyno while the car stands still).

Agreement with MoTeC: median ~0%, worst case ~2%.
The file is memory-mapped (mmap); maths run on numpy (a slower fallback exists).

Channel descriptor record (124 B): see the _C_* offsets below.
  value = raw * mul / scale * 10^(-dec) + shift

Channel names are NOT hardcoded — they are read from a config file
(channels.json by default, template in channels.example.json). To see the
channel names in your own log:  python motec_distance.py file.ld --list-channels

Usage: python motec_distance.py <path> [-r] [--unit m] [-o results.csv] [-j 8]
"""

import argparse
import csv
import json
import mmap
import struct
import sys
from pathlib import Path
from typing import Optional

# numpy = fast (a 2 GB log in ms instead of ~1.3 s). Works without it, just slower.
try:
    import numpy as _np
    _HAVE_NP = True
    _trapz = getattr(_np, "trapezoid", None) or _np.trapz   # numpy 2.0: trapz -> trapezoid
except ImportError:                       # pragma: no cover
    _np = None
    _HAVE_NP = False
    _trapz = None


# ── Units ───────────────────────────────────────────────────────

KM_TO_MI = 0.621371

def to_display(km: float, unit: str) -> tuple[float, str]:
    """km -> chosen display unit: (value, label)."""
    if unit == "miles": return km * KM_TO_MI, "mi"
    if unit == "m":     return km * 1000, "m"
    return km, "km"


# ── Channel name configuration ──────────────────────────────────
# Every team names its channels differently, so the names live in a JSON file
# next to the script rather than in the code. The values below are PLACEHOLDERS —
# put your own into channels.json (see channels.example.json).

CONFIG_FILE = "channels.json"
EXAMPLE_FILE = "channels.example.json"

# Wheel position codes: FL/FR = front left/right, RL/RR = rear left/right.
FRONT_POSITIONS = ("FL", "FR")
REAR_POSITIONS = ("RL", "RR")
POSITIONS = FRONT_POSITIONS + REAR_POSITIONS

DEFAULT_CONFIG = {
    # Fragments of wheel-speed channel names (matched as "contains",
    # case-insensitive). List order = priority.
    "wheel_channels": {
        "FL": ["<front_left_wheel_speed>"],
        "FR": ["<front_right_wheel_speed>"],
        "RL": ["<rear_left_wheel_speed>"],
        "RR": ["<rear_right_wheel_speed>"],
    },
    "gps": {
        # A GPS/IMU reporting the velocity vector as components: matched by name
        # SUFFIX, so whatever precedes it is the receiver prefix — which is how
        # two receivers in one log get told apart automatically.
        "axis_x_suffixes": ["velx"],
        "axis_y_suffixes": ["vely"],
        # Fallback: a scalar GPS speed channel — the name must contain ALL of
        # "scalar_requires" and NONE of "scalar_excludes".
        "scalar_requires": ["gps", "speed"],
        "scalar_excludes": ["command", "setpoint", "target", "limit",
                            "max", "min", "error"],
    },
    # Last resort when there are neither front wheels nor GPS.
    # DELIBERATELY no rear wheels — we never compute distance from driven wheels.
    "fallback_channels": [
        "ground speed",
        "gps speed",
        "vehicle speed",
        "<front_left_wheel_speed>",
        "<front_right_wheel_speed>",
        "<ecu_speed>",
    ],
}


def load_config(path: Optional[str] = None) -> dict:
    """Load channel names: explicit path -> channels.json -> channels.example.json.

    Missing sections fall back to the defaults, so your file only needs to carry
    what actually differs.
    """
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        beside = Path(__file__).resolve().parent
        for name in (CONFIG_FILE, EXAMPLE_FILE):
            candidates.append(Path.cwd() / name)
            candidates.append(beside / name)

    loaded: dict = {}
    for p in candidates:
        if not p.is_file():
            continue
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"Cannot read channel config {p}: {e}")
        break
    else:
        if path:                          # explicit path given, but no such file
            raise SystemExit(f"Channel config not found: {path}")

    cfg = {
        "wheel_channels": dict(DEFAULT_CONFIG["wheel_channels"]),
        "gps": dict(DEFAULT_CONFIG["gps"]),
        "fallback_channels": list(DEFAULT_CONFIG["fallback_channels"]),
    }
    cfg["wheel_channels"].update(loaded.get("wheel_channels", {}))
    cfg["gps"].update(loaded.get("gps", {}))
    if "fallback_channels" in loaded:
        cfg["fallback_channels"] = list(loaded["fallback_channels"])
    return cfg


def _wheel_patterns(cfg: dict, pos: str) -> list[str]:
    """Lower-cased name fragments to look for at a given wheel position."""
    return [w.lower() for w in cfg["wheel_channels"].get(pos, [])]


# ── Binary .ld parser ───────────────────────────────────────────
# Field offsets inside a channel descriptor record.
_C_NEXT   = 0x04   # pointer to the next channel (linked list)
_C_DATA   = 0x08   # sample address
_C_N      = 0x0C   # u32  sample count
_C_DSIZE  = 0x14   # u16  bytes per sample
_C_RATE   = 0x16   # u16  Hz
_C_SHIFT  = 0x18   # i16
_C_MUL    = 0x1A   # i16
_C_SCALE  = 0x1C   # u16
_C_DEC    = 0x1E   # i16  (10^-dec)
_C_NAME   = 0x20   # 32 B name
_C_UNIT   = 0x48   # 12 B unit
_C_SIZE   = 0x7C   # record length

# Read a string / numbers at an offset in the buffer.
def _rstr(d, o, n): return d[o:o+n].rstrip(b"\x00").decode("ascii","replace").strip()
def _ru16(d, o):    return struct.unpack_from("<H", d, o)[0]
def _ri16(d, o):    return struct.unpack_from("<h", d, o)[0]
def _ru32(d, o):    return struct.unpack_from("<I", d, o)[0]


def _read_channels(data) -> list[dict]:
    """Channel descriptors from an .ld buffer (linked list) -> list of dicts."""
    addr = _ru32(data, 0x08)                       # pointer to 1st channel descriptor
    if addr == 0 or addr >= len(data):
        return []

    channels: list[dict] = []
    visited: set[int] = set()                      # guards against a cyclic list
    while addr and addr not in visited and addr + _C_SIZE <= len(data):
        visited.add(addr)
        mul   = _ri16(data, addr + _C_MUL)   or 1
        scale = _ru16(data, addr + _C_SCALE) or 1
        channels.append({
            "name":      _rstr(data, addr + _C_NAME, 32),
            "unit":      _rstr(data, addr + _C_UNIT, 12),
            "n":         _ru32(data, addr + _C_N),
            "dsize":     _ru16(data, addr + _C_DSIZE),
            "rate":      _ru16(data, addr + _C_RATE),
            "shift":     _ri16(data, addr + _C_SHIFT),
            "mul":       mul,
            "scale":     scale,
            "dec":       _ri16(data, addr + _C_DEC),
            "data_addr": _ru32(data, addr + _C_DATA),
        })
        addr = _ru32(data, addr + _C_NEXT)
    return channels


def _read_samples(data, ch: dict):
    """Raw channel samples -> physical values (numpy array or list)."""
    n, base, size = ch["n"], ch["data_addr"], ch["dsize"]
    mul, scale, shift, dec = ch["mul"], ch["scale"], ch["shift"], ch["dec"]
    if size not in (2, 4):                       # only 2/4-byte samples
        return _np.empty(0) if _HAVE_NP else []
    end = base + n * size
    if n == 0 or end > len(data):                # empty, or runs past the file
        return _np.empty(0) if _HAVE_NP else []
    factor = (mul / scale) * (10.0 ** (-dec))    # scaling from the descriptor
    if _HAVE_NP:
        dtype = "<i4" if size == 4 else "<i2"
        raw = _np.frombuffer(data, dtype=dtype, count=n, offset=base)
        return raw.astype(_np.float64) * factor + shift
    fmt = "<i" if size == 4 else "<h"
    return [
        struct.unpack_from(fmt, data, base + i*size)[0] * factor + shift
        for i in range(n)
    ]


def _has_signal(samples) -> bool:
    """Whether the channel has any non-zero sample (i.e. is actually populated)."""
    if _HAVE_NP and isinstance(samples, _np.ndarray):
        return bool(samples.any())
    return any(samples)


# ── Thresholds and heuristics ───────────────────────────────────

# Per-sample speed ceiling (km/h). Above this = sensor fault (0xFFFF). 119 = as MoTeC.
MAX_SPEED_KMH = 119.0

# Front wheels differing by > 10% (and > MIN km) = one sensor died -> take the higher.
WHEEL_DISAGREE_FRAC = 0.10
WHEEL_DISAGREE_MIN_KM = 0.02

# GPS deadband (m/s): suppresses standstill noise. MoTeC has none, so 0.0
# (0.3 under-read the result by ~30%).
GPS_DEADBAND_MS = 0.0
GPS_MAX_MS      = MAX_SPEED_KMH / 3.6   # the same ceiling in m/s
GPS_AGREE_FRAC  = 0.10                  # GPS "confirms" when the gap is <= 10%

# Dead wheel: when GPS shows movement (>= MIN) but the wheel has < 50% of the GPS
# distance, we call it dead and take GPS. The 0.5 threshold is loose on purpose,
# so healthy wheels are never touched.
GPS_HEALTHY_MIN_KM      = 0.05   # 50 m — below that GPS is too small to arbitrate
WHEEL_TRUST_VS_GPS_FRAC = 0.50

# GPS primary (as MoTeC): when GPS is > 10% above the wheel, the wheel lost
# distance -> take GPS. Exception: GPS is noisy when its peak > 1.8x the wheel peak.
GPS_PRIMARY_MARGIN     = 1.10
GPS_CORRUPT_PEAK_RATIO = 1.80

# Two GPS receivers: in agreement (gap <= 30%) -> average them (noise cancels);
# in disagreement -> the one whose peak sits closer to the wheels (noisy one out).
GPS_UNIT_AGREE = 0.30

# Rear wheels only corroborate (driven wheels over-read under wheelspin). The band
# is asymmetric: wider upward (wheelspin), narrow downward (brake lock-up).
REAR_CORROB_LO = 0.05
REAR_CORROB_HI = 0.20

# Status prefixes — also used to classify results in the summary, so they live in
# one place instead of being literals scattered across several functions.
ST_OK = "ok"
ST_RECOVERED = "ok (recovered)"
ST_GPS_PRIMARY = "ok (GPS primary)"
ST_CHECK = "CHECK"
ST_ERROR = "error"


def _integrate_speed(samples, rate_hz: int, unit: str) -> float:
    """Integrate speed (trapezoid) -> km. Out-of-range samples: hold the last good one."""
    if len(samples) < 2:
        return 0.0
    dt = 1.0 / (rate_hz if rate_hz > 0 else 100)
    in_ms = "m/s" in unit.lower()                # some channels are in m/s

    if _HAVE_NP and isinstance(samples, _np.ndarray):
        s = samples * 3.6 if in_ms else samples
        good = (s >= 0.0) & (s <= MAX_SPEED_KMH)
        # forward-fill with the last good value (vectorised form of the loop below)
        last = _np.where(good, _np.arange(s.size), -1)
        _np.maximum.accumulate(last, out=last)
        clean = _np.zeros(s.size)
        seen = last >= 0
        clean[seen] = s[last[seen]]
        return float(_trapz(clean, dx=dt) / 3600.0)   # km/h·s -> km

    samples = [x * 3.6 for x in samples] if in_ms else samples
    clean = []
    last = 0.0
    for x in samples:
        if 0.0 <= x <= MAX_SPEED_KMH:            # good sample -> remember it
            last = x
        clean.append(last)
    return sum(
        (clean[i] + clean[i+1]) / 2.0 * (dt / 3600.0)
        for i in range(len(clean) - 1)
    )


def _wheel_distance(data, channels: list[dict], pos: str, cfg: dict) -> Optional[float]:
    """Distance (km) from the wheel-speed channel at `pos`, or None if missing/empty."""
    for pattern in _wheel_patterns(cfg, pos):
        for ch in channels:
            if pattern in ch["name"].lower():
                samples = _read_samples(data, ch)
                if len(samples) >= 2 and _has_signal(samples):
                    return _integrate_speed(samples, ch["rate"], ch["unit"])
    return None


def _wheel_speed_kmh(data, channels: list[dict], pos: str, cfg: dict):
    """(speed_km/h, rate) for the wheel at `pos`, or None."""
    for pattern in _wheel_patterns(cfg, pos):
        for ch in channels:
            if pattern in ch["name"].lower():
                s = _read_samples(data, ch)
                if len(s) >= 2 and _has_signal(s):
                    kmh = s * 3.6 if "m/s" in ch["unit"].lower() else s
                    return kmh, (ch["rate"] or 100)
    return None


def _peak_kmh(samples) -> float:
    """Peak speed (99.5th percentile) — immune to a single glitch."""
    if _HAVE_NP and isinstance(samples, _np.ndarray):
        return float(_np.percentile(_np.clip(samples, 0, MAX_SPEED_KMH), 99.5))
    vals = sorted(x for x in samples if 0 <= x <= MAX_SPEED_KMH)
    return vals[int(0.995 * (len(vals) - 1))] if vals else 0.0


def _wheel_peak_kmh(data, channels: list[dict], cfg: dict) -> float:
    """Highest speed peak across the four wheels (km/h)."""
    wp = 0.0
    for pos in POSITIONS:
        w = _wheel_speed_kmh(data, channels, pos, cfg)
        if w is not None:
            wp = max(wp, _peak_kmh(w[0]))
    return wp


def _component_gps_units(data, channels: list[dict], cfg: dict) -> list[dict]:
    """Receivers reporting velocity components: {sp(km/h), rate, dist(km), peak, n}.

    Grouped by name prefix — channels <prefix>velX / <prefix>velY form one
    receiver, so two receivers in a log identify themselves.
    """
    x_suffixes = [s.lower() for s in cfg["gps"].get("axis_x_suffixes", [])]
    y_suffixes = [s.lower() for s in cfg["gps"].get("axis_y_suffixes", [])]

    groups: dict[str, dict[str, dict]] = {}
    for ch in channels:
        nm = ch["name"].lower()
        for axis, suffixes in (("x", x_suffixes), ("y", y_suffixes)):
            for suf in suffixes:
                if nm.endswith(suf):
                    prefix = ch["name"][:-len(suf)]
                    groups.setdefault(prefix, {})[axis] = ch
                    break

    units: list[dict] = []
    for axes in groups.values():
        if "x" not in axes or "y" not in axes:       # both axes required
            continue
        vx = _read_samples(data, axes["x"])
        vy = _read_samples(data, axes["y"])
        n = min(len(vx), len(vy))
        if n < 2 or not (_has_signal(vx) or _has_signal(vy)):
            continue
        rate = axes["x"]["rate"] or 100
        if _HAVE_NP and isinstance(vx, _np.ndarray):
            sp = _np.sqrt(vx[:n] * vx[:n] + vy[:n] * vy[:n]) * 3.6   # horizontal speed km/h
            keep = _np.where((sp >= GPS_DEADBAND_MS * 3.6) & (sp <= MAX_SPEED_KMH), sp, 0.0)
            dist = float(_trapz(keep, dx=1.0 / rate) / 3600.0)
            peak = _peak_kmh(sp)
        else:
            dt = 1.0 / rate
            sp = [((vx[i] * vx[i] + vy[i] * vy[i]) ** 0.5) * 3.6 for i in range(n)]
            keep = [s if GPS_DEADBAND_MS * 3.6 <= s <= MAX_SPEED_KMH else 0.0 for s in sp]
            dist = sum((keep[i] + keep[i+1]) / 2.0 * (dt / 3600.0) for i in range(n - 1))
            peak = _peak_kmh(sp)
        units.append({"sp": sp, "rate": rate, "dist": dist, "peak": peak, "n": n})
    return units


def _gps_fused(data, channels: list[dict], cfg: dict):
    """Fuse GPS receivers -> (dist_km, speed, rate).

    One receiver -> that one; two in agreement -> their average; two in
    disagreement -> the one whose peak sits closer to the wheels (noisy one out).
    """
    units = _component_gps_units(data, channels, cfg)
    if not units:
        return None
    if len(units) == 1:
        u = units[0]
        return u["dist"], u["sp"], u["rate"]
    units.sort(key=lambda u: -u["n"])            # the two best-sampled ones
    a, b = units[0], units[1]
    hi = max(a["dist"], b["dist"])
    if hi > 0 and abs(a["dist"] - b["dist"]) / hi <= GPS_UNIT_AGREE:
        rep = a if a["peak"] <= b["peak"] else b  # cleaner trace for the tests below
        return (a["dist"] + b["dist"]) / 2.0, rep["sp"], rep["rate"]
    wp = _wheel_peak_kmh(data, channels, cfg)    # disagreement -> closer to wheel peak
    pick = a if abs(a["peak"] - wp) <= abs(b["peak"] - wp) else b
    return pick["dist"], pick["sp"], pick["rate"]


def _gps_distance(data, channels: list[dict], cfg: dict) -> Optional[float]:
    """Distance (km) from GPS: velX/velY components (fused) or a scalar speed channel."""
    fused = _gps_fused(data, channels, cfg)
    if fused is not None:
        return fused[0]
    # fallback: a scalar GPS speed channel
    requires = [s.lower() for s in cfg["gps"].get("scalar_requires", [])]
    excludes = [s.lower() for s in cfg["gps"].get("scalar_excludes", [])]
    if not requires:
        return None
    for ch in channels:
        nm = ch["name"].lower()
        if all(x in nm for x in requires) and not any(x in nm for x in excludes):
            samples = _read_samples(data, ch)
            if len(samples) >= 2 and _has_signal(samples):
                return _integrate_speed(samples, ch["rate"], ch["unit"])
    return None


def _gps_speed_kmh(data, channels: list[dict], cfg: dict):
    """Speed of the chosen GPS (sample by sample) — for the dropout/corruption tests."""
    if not _HAVE_NP:
        return None
    fused = _gps_fused(data, channels, cfg)
    if fused is None:
        return None
    return fused[1], fused[2]


def _gps_velocity_suspect(gps_comp, data, channels: list[dict], cfg: dict) -> bool:
    """GPS is noisy when its peak >> the wheel peak (fallback for a single receiver)."""
    if not _HAVE_NP or gps_comp is None:
        return False
    gps_peak = _peak_kmh(gps_comp[0])
    wheel_peak = _wheel_peak_kmh(data, channels, cfg)
    return wheel_peak > 5.0 and gps_peak > wheel_peak * GPS_CORRUPT_PEAK_RATIO


def _rear_confirms(chosen: float, rears: dict) -> bool:
    """Whether a rear wheel corroborates the value (inside the asymmetric slip band)."""
    if not rears or chosen <= 0:
        return False
    lo, hi = chosen * (1 - REAR_CORROB_LO), chosen * (1 + REAR_CORROB_HI)
    return any(lo <= r <= hi for r in rears.values())


# ── Source selection ────────────────────────────────────────────

def parse_ld(path: Path, cfg: Optional[dict] = None) -> tuple[Optional[float], str]:
    """(distance_km, status). Status starts with 'ok', 'CHECK' or 'error'.

    The file is memory-mapped (mmap) — we read only the channels we need instead
    of pulling the whole log into RAM.
    """
    if cfg is None:
        cfg = load_config()
    try:
        fh = open(path, "rb")
    except OSError:
        return None, f"{ST_ERROR}: unreadable"
    try:
        try:
            data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            return None, f"{ST_ERROR}: not a .ld file"   # empty / not mappable
        try:
            return _distance_from_buffer(data, cfg)
        finally:
            data.close()
    finally:
        fh.close()


def _distance_from_buffer(data, cfg: dict) -> tuple[Optional[float], str]:
    """Compute (distance_km, status) from an .ld buffer."""
    if len(data) < 0x100 or data[0] != 0x40:      # an .ld header starts with 0x40
        return None, f"{ST_ERROR}: not a .ld file"

    channels = _read_channels(data)
    if not channels:
        return None, f"{ST_ERROR}: no channels"

    # Compute every source separately, then pick (logic below). In short: a healthy
    # front wheel is the answer; GPS takes over when a wheel died or lost distance;
    # rear wheels only vote.
    fl = _wheel_distance(data, channels, "FL", cfg)
    fr = _wheel_distance(data, channels, "FR", cfg)
    fronts = {k: v for k, v in (("FL", fl), ("FR", fr)) if v is not None}
    rears = {k: v for k, v in ((p, _wheel_distance(data, channels, p, cfg))
                               for p in REAR_POSITIONS)
             if v is not None}
    gps = _gps_distance(data, channels, cfg)

    # Dead wheel: GPS showed movement but the wheel has < 50% of its distance ->
    # drop the wheel (GPS is taken below). Healthy wheels sit above, untouched.
    dead_fronts: list[str] = []
    gps_healthy = gps is not None and gps >= GPS_HEALTHY_MIN_KM
    if gps_healthy:
        floor = gps * WHEEL_TRUST_VS_GPS_FRAC
        for label in FRONT_POSITIONS:
            if label in fronts and fronts[label] < floor:
                dead_fronts.append(label)
                del fronts[label]
        for label in REAR_POSITIONS:
            if label in rears and rears[label] < floor:
                del rears[label]

    # GPS primary (as MoTeC): when healthy component GPS is > 10% above the wheel,
    # the wheel lost distance -> take GPS. Unless GPS is noisy -> keep the wheel and flag.
    gps_comp = _gps_speed_kmh(data, channels, cfg) if gps_healthy else None
    if gps_comp is not None and fronts:
        pos = "FL" if "FL" in fronts else "FR"
        if gps > fronts[pos] * GPS_PRIMARY_MARGIN:
            if _gps_velocity_suspect(gps_comp, data, channels, cfg):
                return fronts[pos], (f"{ST_CHECK}: GPS velocity spiky (peak >> wheel), "
                                     f"used {pos}={fronts[pos]*1000:.0f}m, "
                                     f"GPS={gps*1000:.0f}m")
            return gps, (f"{ST_GPS_PRIMARY}: {pos} under-recorded, "
                         f"GPS used ({gps*1000:.0f}m)")

    def _agrees(a: float, b: float) -> bool:
        hi = max(abs(a), abs(b))                  # agreement = gap <= 10%
        return hi == 0 or abs(a - b) / hi <= GPS_AGREE_FRAC

    def _witnesses(value: float) -> list[str]:
        w = []                                    # who corroborates: GPS and/or rears
        if gps is not None and _agrees(gps, value):
            w.append(f"GPS={gps*1000:.0f}m")
        if _rear_confirms(value, rears):
            w.append("rears")
        return w

    if len(fronts) == 2:
        lo, hi = min(fl, fr), max(fl, fr)
        disagree = (hi > 0 and (hi - lo) > WHEEL_DISAGREE_MIN_KM
                    and (hi - lo) / hi > WHEEL_DISAGREE_FRAC)
        if not disagree:
            return fl, f"{ST_OK} (FL)"
        # A real disagreement: one sensor died (under-reads) -> the working one is higher.
        pct = (hi - lo) / hi * 100
        detail = f"FL={fl*1000:.0f} FR={fr*1000:.0f} m differ {pct:.0f}%"
        w = _witnesses(hi)
        if w:
            return hi, (f"{ST_RECOVERED}: {detail}, used higher wheel, "
                        f"confirmed by {', '.join(w)}")
        gps_note = f", GPS={gps*1000:.0f}m disagrees" if gps is not None else ", no GPS"
        return hi, f"{ST_CHECK}: {detail}{gps_note}, used higher wheel, no confirmation"

    if len(fronts) == 1:
        pos, val = next(iter(fronts.items()))
        w = _witnesses(val)
        if w:
            return val, (f"{ST_RECOVERED}: only front {pos}={val*1000:.0f}m, "
                         f"confirmed by {', '.join(w)}")
        return val, f"{ST_CHECK}: only one front sensor ({pos}), no confirmation"

    # No front wheels -> GPS, if present.
    if gps is not None:
        if dead_fronts:
            return gps, (f"{ST_RECOVERED}: front {'+'.join(dead_fronts)} dead vs "
                         f"GPS, used GPS ({gps*1000:.0f}m)")
        return gps, f"{ST_OK} (GPS, no front wheel data)"
    if rears:
        # Rear (driven) wheels only — possibly a dyno run (they spin, the car stands).
        # We do not compute from them; the file is excluded from the total (km=None)
        # but still listed.
        return None, (f"{ST_ERROR}: rear-only (no front/GPS) -> excluded, possible dyno")

    # No wheels and no GPS -> the generic speed channels from the config.
    for keyword in cfg["fallback_channels"]:
        k = keyword.lower()
        for ch in channels:
            if k not in ch["name"].lower():
                continue
            samples = _read_samples(data, ch)
            if len(samples) < 2 or not _has_signal(samples):
                continue
            return _integrate_speed(samples, ch["rate"], ch["unit"]), f"{ST_OK} ({ch['name']})"

    return None, f"{ST_ERROR}: no speed channel"


def list_channels(path: Path) -> list[dict]:
    """Every channel in an .ld file — to look up your own names before configuring."""
    with open(path, "rb") as fh:
        data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(data) < 0x100 or data[0] != 0x40:
                return []
            return _read_channels(data)
        finally:
            data.close()


# ── File discovery ──────────────────────────────────────────────

def discover(paths: list[str], recursive: bool) -> list[Path]:
    """.ld files from paths, folders and globs (sorted, de-duplicated)."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():                                   # folder
            found.extend(p.glob("**/*.ld" if recursive else "*.ld"))
        elif "*" in str(p) or "?" in str(p):             # glob, e.g. logs/*.ld
            parent = p.parent if str(p.parent) != "." else Path(".")
            found.extend(f for f in parent.glob(p.name) if f.suffix.lower() == ".ld")
        elif p.is_file() and p.suffix.lower() == ".ld":  # single file
            found.append(p)
    return sorted(set(found))


# ── Command-line interface ──────────────────────────────────────

def _worker(path: Path, cfg: dict) -> tuple[Path, Optional[float], str]:
    """Process-pool wrapper: never raises."""
    try:
        km, status = parse_ld(path, cfg)
    except Exception:
        km, status = None, f"{ST_ERROR}: exception"
    return path, km, status


def _print_channels(files: list[Path]) -> None:
    """--list-channels mode: names, units and rates of the channels in each file."""
    for f in files:
        print(f"\n{f}")
        channels = list_channels(f)
        if not channels:
            print("  (could not read channels)")
            continue
        width = max(len(c["name"]) for c in channels)
        for c in channels:
            print(f"  {c['name']:<{width}}  {c['unit']:<8} {c['rate']:>4} Hz  "
                  f"{c['n']:>9} samples")


def main() -> None:
    ap = argparse.ArgumentParser(description="Distance from MoTeC .ld files.")
    ap.add_argument("paths", nargs="+", help="File(s), folder(s), or globs")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="search folders recursively")
    ap.add_argument("--unit", choices=["km", "miles", "m"], default="km",
                    help="output unit (default km)")
    ap.add_argument("-o", "--output", metavar="FILE", help="write results to a CSV file")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="no table, just a progress counter and the summary")
    ap.add_argument("-c", "--channels", metavar="FILE",
                    help=f"JSON file with channel names (default {CONFIG_FILE}, "
                         f"falling back to {EXAMPLE_FILE})")
    ap.add_argument("--list-channels", action="store_true",
                    help="print the channels found in the files and exit "
                         "(handy when filling in the config)")
    ap.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                    help="parallel worker processes (default 1; try 4-8 for "
                         "large batches). I/O-bound past your disk's throughput.")
    args = ap.parse_args()

    files = discover(args.paths, args.recursive)
    if not files:
        print("No .ld files found.", file=sys.stderr)
        sys.exit(1)

    if args.list_channels:
        _print_channels(files)
        return

    cfg = load_config(args.channels)

    results: list[tuple[Path, Optional[float], str]] = []
    col_w = max(len(str(f)) for f in files) + 2

    if not args.quiet:
        print(f"\n{'File':<{col_w}}  {'Distance':>12}  Status")
        print("-" * (col_w + 26))

    def emit(f: Path, km: Optional[float], status: str) -> None:
        # store the result and print a row (or a progress counter when quiet)
        results.append((f, km, status))
        done = len(results)
        if args.quiet:
            if done % 50 == 0 or done == len(files):
                print(f"\r  {done}/{len(files)} files...", end="", file=sys.stderr, flush=True)
            return
        if km is None:
            print(f"{str(f):<{col_w}}  {'-':>12}  {status}")
        else:
            val, lab = to_display(km, args.unit)
            print(f"{str(f):<{col_w}}  {val:>10.3f} {lab}  {status}")

    if args.jobs and args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        from itertools import repeat
        # map() preserves input order -> reproducible rows and CSV
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for f, km, status in ex.map(_worker, files, repeat(cfg), chunksize=4):
                emit(f, km, status)
    else:
        for f in files:
            _, km, status = _worker(f, cfg)
            emit(f, km, status)
    if args.quiet:
        print(file=sys.stderr)

    # summary: how many parsed, how many GPS recovered, how many need a look
    valid = [(f, km) for f, km, _ in results if km is not None]
    recovered = [(f, s) for f, km, s in results if km is not None
                 and (s.startswith(ST_RECOVERED) or s.startswith(ST_GPS_PRIMARY))]
    flagged = [(f, s) for f, km, s in results
               if km is not None and s.startswith(ST_CHECK)]
    total_km = sum(km for _, km in valid)
    tv, lab = to_display(total_km, args.unit)

    print(f"\nFiles analyzed      : {len(files)}")
    print(f"Parsed OK           : {len(valid)}")
    print(f"Failed              : {len(files) - len(valid)}")
    print(f"Auto-recovered      : {len(recovered)}  (dead/disagreeing wheel, "
          f"recovered via GPS/rears)")
    print(f"Needs review        : {len(flagged)}  (no independent confirmation)")
    print(f"Total distance      : {tv:.3f} {lab}\n")

    if flagged and not args.quiet:
        print("Sessions to verify manually:")
        for f, s in flagged:
            print(f"  {f.name}: {s}")
        print()

    if args.output:                                  # optional CSV export
        out = Path(args.output)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", f"distance_{lab}", "status", "review"])
            for f, km, status in results:
                review = "REVIEW" if status.startswith(ST_CHECK) else ""
                if km is None:
                    w.writerow([str(f), "", status, review])
                else:
                    v, _ = to_display(km, args.unit)
                    w.writerow([str(f), f"{v:.4f}", status, review])
            w.writerow([])
            w.writerow(["TOTAL", f"{tv:.4f}",
                        f"{len(valid)}/{len(files)} parsed",
                        f"{len(flagged)} to review"])
        print(f"Results saved -> {out}")


if __name__ == "__main__":
    main()
