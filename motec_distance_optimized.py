#!/usr/bin/env python3
""" pisane ze wsparciem Claude Code, więc proszę o wyrozumiałość ;)

"""

"""
motec_distance_optimized.py — zoptymalizowana kopia motec_distance.py.

Ta sama logika wyboru źródła i te same wyniki (na ścieżce numpy bit w bit),
ale szybciej. Co zmieniono względem oryginału:

  * cache na sesję (_Session): próbki każdego kanału są czytane z pliku i
    skalowane tylko RAZ — oryginał dekodował koła i GPS nawet 3x
    (dystans / szczyt prędkości / test zaszumienia), a pełną fuzję GPS
    (sqrt + całka + percentyl) liczył dwukrotnie,
  * nazwy kanałów lower() raz przy starcie zamiast przy każdym wyszukiwaniu,
  * dekodowanie numpy: jedna alokacja float64 (raw * factor) zamiast
    astype() + mnożenie; shift dodawany w miejscu i tylko gdy != 0,
  * całkowanie numpy: szybka ścieżka bez forward-fill, gdy wszystkie próbki
    są w zakresie (typowy przypadek) — mniej alokacji,
  * zapas bez numpy: moduł array zamiast struct.unpack_from na każdą próbkę
    (kilkukrotnie szybciej), całka liczona w jednej pętli bez listy pośredniej,
  * POPRAWKA błędu z oryginału: bez numpy `lista * 3.6` rzucała TypeError,
    gdy kanał koła był w m/s, a dwa odbiorniki GPS się nie zgadzały
    (_wheel_speed_kmh); tu konwersja działa też na listach.

Interfejs identyczny: parse_ld(path) -> (km, status); CLI jak w oryginale.
Opis formatu .ld, priorytetów źródeł i progów: patrz motec_distance.py.

Użycie: python motec_distance_optimized.py <ścieżka> [-r] [--unit m] [-o wyniki.csv] [-j 8]
"""

import argparse
import array
import csv
import mmap
import struct
import sys
from pathlib import Path
from typing import Optional

# numpy = szybko (log 2 GB w ms zamiast ~1.3 s). Bez niego działa, ale wolniej.
try:
    import numpy as _np
    _HAVE_NP = True
    _trapz = getattr(_np, "trapezoid", None) or _np.trapz   # numpy 2.0: trapz -> trapezoid
except ImportError:                       # pragma: no cover
    _np = None
    _HAVE_NP = False
    _trapz = None


KM_TO_MI = 0.621371

def to_display(km: float, unit: str) -> tuple[float, str]:
    """km -> wybrana jednostka wyświetlania: (wartość, etykieta)."""
    if unit == "miles": return km * KM_TO_MI, "mi"
    if unit == "m":     return km * 1000, "m"
    return km, "km"


# Zapas dla nietypowych loggerów (gdy brak FL/FR/GPS). CELOWO bez kół tylnych
# i gołego "wheel_speed" (pasuje do "..._rl/rr") — z napędowych nie liczymy.
SPEED_PRIORITY = [
    "ground speed", "gps speed", "vehicle speed",
    "wheel_speed_fl",
    "wheel_speed_fr",
    "speed_actual",
]


# ── Parser binarny .ld ──────────────────────────────────────────
# Offsety pól w rekordzie deskryptora kanału.
_C_NEXT   = 0x04   # wskaźnik na następny kanał (lista wiązana)
_C_DATA   = 0x08   # adres próbek
_C_N      = 0x0C   # u32  liczba próbek
_C_DSIZE  = 0x14   # u16  bajtów/próbkę
_C_RATE   = 0x16   # u16  Hz
_C_SHIFT  = 0x18   # i16
_C_MUL    = 0x1A   # i16
_C_SCALE  = 0x1C   # u16
_C_DEC    = 0x1E   # i16  (10^-dec)
_C_NAME   = 0x20   # 32 B nazwa
_C_UNIT   = 0x48   # 12 B jednostka
_C_SIZE   = 0x7C   # długość rekordu

# Odczyt napisu / liczb spod offsetu w buforze.
def _rstr(d, o, n): return d[o:o+n].rstrip(b"\x00").decode("ascii","replace").strip()
def _ru16(d, o):    return struct.unpack_from("<H", d, o)[0]
def _ri16(d, o):    return struct.unpack_from("<h", d, o)[0]
def _ru32(d, o):    return struct.unpack_from("<I", d, o)[0]

def _read_samples(data, ch: dict):
    """Surowe próbki kanału -> wartości fizyczne (tablica numpy lub lista)."""
    n, base, size = ch["n"], ch["data_addr"], ch["dsize"]
    if size not in (2, 4):                       # tylko próbki 2/4-bajtowe
        return _np.empty(0) if _HAVE_NP else []
    end = base + n * size
    if n == 0 or end > len(data):                # pusty lub wychodzi poza plik
        return _np.empty(0) if _HAVE_NP else []
    mul, scale, shift, dec = ch["mul"], ch["scale"], ch["shift"], ch["dec"]
    factor = (mul / scale) * (10.0 ** (-dec))    # skalowanie z deskryptora
    if _HAVE_NP:
        raw = _np.frombuffer(data, dtype="<i4" if size == 4 else "<i2",
                             count=n, offset=base)
        out = raw * factor                       # od razu float64, jedna alokacja
        if shift:
            out += shift
        return out
    # bez numpy: array.frombytes dekoduje cały blok naraz (dużo szybciej niż
    # struct na próbkę); iter_unpack to zapas na egzotyczne rozmiary typów C
    arr = array.array("i" if size == 4 else "h")
    if arr.itemsize == size:
        arr.frombytes(data[base:end])
        if sys.byteorder == "big":               # pragma: no cover
            arr.byteswap()
        if shift:
            return [x * factor + shift for x in arr]
        return [x * factor for x in arr]
    fmt = "<i" if size == 4 else "<h"            # pragma: no cover
    return [v * factor + shift for (v,) in struct.iter_unpack(fmt, data[base:end])]


def _has_signal(samples) -> bool:
    """Czy kanał ma jakąkolwiek niezerową próbkę (czyli jest wypełniony)."""
    if _HAVE_NP and isinstance(samples, _np.ndarray):
        return bool(samples.any())
    return any(samples)

# Limit prędkości próbki (km/h). Wyżej = błąd czujnika (0xFFFF). 119 = jak w MoTeC.
MAX_SPEED_KMH = 119.0

# Różnica przednich kół > 10% (i > MIN km) = jeden czujnik padł -> bierz wyższe.
WHEEL_DISAGREE_FRAC = 0.10
WHEEL_DISAGREE_MIN_KM = 0.02

# Strefa martwa GPS (m/s): tłumi szum na postoju. MoTeC jej nie ma, więc 0.0
# (0.3 zaniżało wynik o ~30%).
GPS_DEADBAND_MS = 0.0
GPS_MAX_MS      = MAX_SPEED_KMH / 3.6   # ten sam limit w m/s
GPS_AGREE_FRAC  = 0.10                  # GPS "potwierdza", gdy różnica <= 10%

# Koło padłe: gdy GPS pokazuje ruch (>= MIN), a koło ma < 50% dystansu GPS,
# uznajemy je za martwe i bierzemy GPS. Próg 0.5 luźny, by nie ruszać zdrowych.
GPS_HEALTHY_MIN_KM      = 0.05   # 50 m — niżej GPS za mały, by rozstrzygać
WHEEL_TRUST_VS_GPS_FRAC = 0.50

# GPS główny (jak w MoTeC): gdy GPS ma > 10% nad kołem, koło zgubiło dystans ->
# bierz GPS. Wyjątek: GPS zaszumiony, gdy jego szczyt > 1.8x szczytu koła.
GPS_PRIMARY_MARGIN     = 1.10
GPS_CORRUPT_PEAK_RATIO = 1.80

# Dwa odbiorniki GPS (np. Xsens 670+680): zgodne (różnica <= 30%) -> średnia
# (szum się znosi); niezgodne -> ten o szczycie bliższym kołom (zaszumiony out).
GPS_UNIT_AGREE = 0.30

# Tylne koła tylko potwierdzają (napędowe zawyżają przez poślizg). Pasmo
# niesymetryczne: szersze w górę (poślizg), wąskie w dół (blokada hamulca).
REAR_CORROB_LO = 0.05
REAR_CORROB_HI = 0.20

def _integrate_speed(samples, rate_hz: int, unit: str = "") -> float:
    """Całkuje prędkość (trapezy) -> km. Próbki poza zakresem: trzymaj ostatnią dobrą."""
    if len(samples) < 2:
        return 0.0
    dt = 1.0 / (rate_hz if rate_hz > 0 else 100)
    in_ms = "m/s" in unit.lower()                # część kanałów jest w m/s

    if _HAVE_NP and isinstance(samples, _np.ndarray):
        s = samples * 3.6 if in_ms else samples
        good = (s >= 0.0) & (s <= MAX_SPEED_KMH)
        if good.all():                           # typowy przypadek: nic do łatania
            return float(_trapz(s, dx=dt) / 3600.0)
        # wypełnij w przód ostatnią dobrą wartością (wektorowa wersja pętli niżej)
        last = _np.where(good, _np.arange(s.size), -1)
        _np.maximum.accumulate(last, out=last)
        clean = _np.zeros(s.size)
        seen = last >= 0
        clean[seen] = s[last[seen]]
        return float(_trapz(clean, dx=dt) / 3600.0)   # km/h·s -> km

    # bez numpy: jedna pętla, suma trapezów bez listy pośredniej
    it = (x * 3.6 for x in samples) if in_ms else samples
    last = prev = 0.0
    acc = 0.0
    first = True
    for x in it:
        if 0.0 <= x <= MAX_SPEED_KMH:            # dobra próbka -> zapamiętaj
            last = x
        if not first:
            acc += prev + last                   # 2 * pole trapezu / dt
        prev = last
        first = False
    return acc * dt / 7200.0                     # /2 (trapez) i /3600 (h -> s)


def _peak_kmh(samples) -> float:
    """Szczyt prędkości (99.5 percentyl) — odporny na pojedynczy glitch."""
    if _HAVE_NP and isinstance(samples, _np.ndarray):
        return float(_np.percentile(_np.clip(samples, 0, MAX_SPEED_KMH), 99.5))
    vals = sorted(x for x in samples if 0 <= x <= MAX_SPEED_KMH)
    return vals[int(0.995 * (len(vals) - 1))] if vals else 0.0


# Skalarny GPS_Speed (auta bez velX/velY). Pomijamy kanały zadane/komendy
# (target/command...), które tylko mają w nazwie gps+speed.
_GPS_SCALAR_EXCLUDE = ("command", "setpoint", "target", "limit", "max", "min", "error")

_WHEEL_POSITIONS = ("FL", "FR", "RL", "RR")


class _Session:
    """Kontekst jednej sesji .ld: bufor + kanały + cache wyników pochodnych.

    Każdy kanał jest dekodowany z pliku najwyżej raz; koła, fuzja GPS i szczyt
    prędkości kół są liczone raz i zapamiętywane — to główne źródło przyspieszenia
    względem oryginału, który powtarzał te przebiegi przy każdym teście spójności.
    """
    __slots__ = ("data", "channels", "lnames", "_samples", "_wheels",
                 "_wheel_dists", "_wheel_peak", "_gps_fused", "_gps_fused_done")

    def __init__(self, data, channels: list[dict]):
        self.data = data
        self.channels = channels
        self.lnames = [ch["name"].lower() for ch in channels]   # lower() raz
        self._samples: dict[int, object] = {}    # idx kanału -> próbki fizyczne
        self._wheels: dict[str, Optional[tuple]] = {}   # "fl" -> (kmh, rate) | None
        self._wheel_dists: dict[str, float] = {}
        self._wheel_peak: Optional[float] = None
        self._gps_fused = None
        self._gps_fused_done = False

    def samples(self, idx: int):
        """Próbki fizyczne kanału o danym indeksie (dekodowane raz, potem cache)."""
        s = self._samples.get(idx)
        if s is None:
            s = _read_samples(self.data, self.channels[idx])
            self._samples[idx] = s
        return s

    def wheel(self, pos: str) -> Optional[tuple]:
        """(prędkość_km/h, rate) dla Wheel_Speed_<pos>, albo None gdy brak/pusty."""
        key = pos.lower()
        if key in self._wheels:
            return self._wheels[key]
        target = f"wheel_speed_{key}"
        result = None
        for idx, nm in enumerate(self.lnames):
            if target in nm:
                s = self.samples(idx)
                if len(s) >= 2 and _has_signal(s):
                    ch = self.channels[idx]
                    if "m/s" in ch["unit"].lower():
                        # poprawka: na liście (bez numpy) mnożymy element po elemencie
                        kmh = s * 3.6 if _HAVE_NP and isinstance(s, _np.ndarray) \
                            else [x * 3.6 for x in s]
                    else:
                        kmh = s
                    result = (kmh, ch["rate"] or 100)
                    break
        self._wheels[key] = result
        return result

    def wheel_dist(self, pos: str) -> Optional[float]:
        """Dystans (km) z Wheel_Speed_<pos>, albo None gdy brak/pusty."""
        key = pos.lower()
        if key not in self._wheel_dists:
            w = self.wheel(pos)
            self._wheel_dists[key] = None if w is None \
                else _integrate_speed(w[0], w[1])
        return self._wheel_dists[key]

    def wheel_peak(self) -> float:
        """Najwyższy szczyt prędkości z czterech kół (km/h)."""
        if self._wheel_peak is None:
            wp = 0.0
            for pos in _WHEEL_POSITIONS:
                w = self.wheel(pos)
                if w is not None:
                    wp = max(wp, _peak_kmh(w[0]))
            self._wheel_peak = wp
        return self._wheel_peak

    def gps_fused(self):
        """Wynik _gps_fused liczony raz (może być None)."""
        if not self._gps_fused_done:
            self._gps_fused = _gps_fused(self)
            self._gps_fused_done = True
        return self._gps_fused


def _component_gps_units(sess: _Session) -> list[dict]:
    """Lista odbiorników velX/velY: {sp(km/h), rate, dist(km), peak, n}.
    Grupowane po przedrostku nazwy (np. Xsens_MTI_680_)."""
    groups: dict[str, dict[str, int]] = {}
    for idx, nm in enumerate(sess.lnames):
        if nm.endswith("velx") or nm.endswith("vely"):
            groups.setdefault(sess.channels[idx]["name"][:-4], {})[nm[-4:]] = idx
    units: list[dict] = []
    lo_kmh = GPS_DEADBAND_MS * 3.6
    for axes in groups.values():
        if "velx" not in axes or "vely" not in axes:   # potrzebne obie osie
            continue
        vx = sess.samples(axes["velx"])
        vy = sess.samples(axes["vely"])
        n = min(len(vx), len(vy))
        if n < 2 or not (_has_signal(vx) or _has_signal(vy)):
            continue
        rate = sess.channels[axes["velx"]]["rate"] or 100
        if _HAVE_NP and isinstance(vx, _np.ndarray):
            vx, vy = vx[:n], vy[:n]
            sp = _np.sqrt(vx * vx + vy * vy)     # prędkość pozioma
            sp *= 3.6                            # km/h, w miejscu
            keep = _np.where((sp >= lo_kmh) & (sp <= MAX_SPEED_KMH), sp, 0.0)
            dist = float(_trapz(keep, dx=1.0 / rate) / 3600.0)
        else:
            dt = 1.0 / rate
            sp = [((vx[i] * vx[i] + vy[i] * vy[i]) ** 0.5) * 3.6 for i in range(n)]
            keep = [s if lo_kmh <= s <= MAX_SPEED_KMH else 0.0 for s in sp]
            dist = sum((keep[i] + keep[i+1]) / 2.0 * (dt / 3600.0) for i in range(n - 1))
        units.append({"sp": sp, "rate": rate, "dist": dist,
                      "peak": _peak_kmh(sp), "n": n})
    return units


def _gps_fused(sess: _Session):
    """Łączy odbiorniki GPS -> (dist_km, prędkość, rate). Jeden -> on; dwa zgodne
    -> średnia; dwa niezgodne -> ten o szczycie bliższym kołom (zaszumiony out)."""
    units = _component_gps_units(sess)
    if not units:
        return None
    if len(units) == 1:
        u = units[0]
        return u["dist"], u["sp"], u["rate"]
    units.sort(key=lambda u: -u["n"])            # dwa najlepiej próbkowane
    a, b = units[0], units[1]
    hi = max(a["dist"], b["dist"])
    if hi > 0 and abs(a["dist"] - b["dist"]) / hi <= GPS_UNIT_AGREE:
        rep = a if a["peak"] <= b["peak"] else b  # czystszy przebieg do dalszych testów
        return (a["dist"] + b["dist"]) / 2.0, rep["sp"], rep["rate"]
    wp = sess.wheel_peak()                       # niezgodne -> bliżej szczytu kół
    pick = a if abs(a["peak"] - wp) <= abs(b["peak"] - wp) else b
    return pick["dist"], pick["sp"], pick["rate"]


def _gps_distance(sess: _Session) -> Optional[float]:
    """Dystans (km) z GPS: składowe velX/velY (z fuzją) lub skalarny GPS_Speed."""
    fused = sess.gps_fused()
    if fused is not None:
        return fused[0]
    # zapas: skalarny kanał prędkości GPS
    for idx, nm in enumerate(sess.lnames):
        if "gps" in nm and "speed" in nm and not any(x in nm for x in _GPS_SCALAR_EXCLUDE):
            samples = sess.samples(idx)
            if len(samples) >= 2 and _has_signal(samples):
                ch = sess.channels[idx]
                return _integrate_speed(samples, ch["rate"], ch["unit"])
    return None


def _gps_speed_kmh(sess: _Session):
    """Prędkość wybranego GPS (próbka po próbce) — dla testów dziury/zepsucia."""
    if not _HAVE_NP:
        return None
    fused = sess.gps_fused()
    if fused is None:
        return None
    return fused[1], fused[2]


def _gps_velocity_suspect(gps_comp, sess: _Session) -> bool:
    """GPS zaszumiony, gdy jego szczyt >> szczyt kół (zapas dla jednego odbiornika)."""
    if not _HAVE_NP or gps_comp is None:
        return False
    gps_peak = _peak_kmh(gps_comp[0])
    wheel_peak = sess.wheel_peak()
    return wheel_peak > 5.0 and gps_peak > wheel_peak * GPS_CORRUPT_PEAK_RATIO


def _rear_confirms(chosen: float, rears: dict) -> bool:
    """Czy tylne koło potwierdza wartość (w paśmie poślizgu, niesymetrycznym)."""
    if not rears or chosen <= 0:
        return False
    lo, hi = chosen * (1 - REAR_CORROB_LO), chosen * (1 + REAR_CORROB_HI)
    return any(lo <= r <= hi for r in rears.values())


def parse_ld(path: Path) -> tuple[Optional[float], str]:
    """(dystans_km, status). status: 'ok ...' / 'CHECK: ...' / 'error: ...'.
    Plik mapowany w pamięć (mmap) — czytamy tylko potrzebne kanały, nie cały plik."""
    try:
        fh = open(path, "rb")
    except OSError:
        return None, "error: unreadable"
    try:
        try:
            data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            return None, "error: not a .ld file"      # pusty / nie do zmapowania
        try:
            return _distance_from_buffer(data)
        finally:
            data.close()
    finally:
        fh.close()


def _distance_from_buffer(data) -> tuple[Optional[float], str]:
    """Liczy (dystans_km, status) z bufora .ld."""
    if len(data) < 0x100 or data[0] != 0x40:      # nagłówek .ld zaczyna się od 0x40
        return None, "error: not a .ld file"

    addr = _ru32(data, 0x08)                       # wskaźnik na 1. deskryptor kanału
    if addr == 0 or addr >= len(data):
        return None, "error: bad channel pointer"

    # Kanały to lista wiązana — idziemy po "next"; visited chroni przed pętlą.
    channels: list[dict] = []
    visited: set[int] = set()

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

    if not channels:
        return None, "error: no channels"

    sess = _Session(data, channels)

    # Liczymy każde źródło osobno, potem wybieramy (logika niżej). W skrócie:
    # zdrowe przednie koło jest wynikiem; GPS przejmuje, gdy koło padło/zgubiło
    # dystans; tylne tylko głosują.
    fl = sess.wheel_dist("FL")
    fr = sess.wheel_dist("FR")
    fronts = {k: v for k, v in (("FL", fl), ("FR", fr)) if v is not None}
    rears  = {k: v for k, v in (("RL", sess.wheel_dist("RL")),
                                ("RR", sess.wheel_dist("RR")))
              if v is not None}
    gps = _gps_distance(sess)

    # Koło padłe: GPS pokazał ruch, a koło ma < 50% jego dystansu -> odrzuć koło
    # (niżej weźmiemy GPS). Zdrowe koła są wyżej, więc bez zmian.
    dead_fronts: list[str] = []
    gps_healthy = gps is not None and gps >= GPS_HEALTHY_MIN_KM
    if gps_healthy:
        floor = gps * WHEEL_TRUST_VS_GPS_FRAC
        for label in ("FL", "FR"):
            if label in fronts and fronts[label] < floor:
                dead_fronts.append(label)
                del fronts[label]
        for label in ("RL", "RR"):
            if label in rears and rears[label] < floor:
                del rears[label]

    # GPS główny (jak MoTeC): gdy zdrowy GPS ze składowych ma > 10% nad kołem,
    # koło zgubiło dystans -> GPS. Chyba że GPS zaszumiony -> zostaw koło i oznacz.
    gps_comp = _gps_speed_kmh(sess) if gps_healthy else None
    if gps_comp is not None and fronts:
        pos = "FL" if "FL" in fronts else "FR"
        if gps > fronts[pos] * GPS_PRIMARY_MARGIN:
            if _gps_velocity_suspect(gps_comp, sess):
                return fronts[pos], (f"CHECK: GPS velocity spiky (peak >> wheel), "
                                     f"used {pos}={fronts[pos]*1000:.0f}m, GPS={gps*1000:.0f}m")
            return gps, (f"ok (GPS primary): {pos} under-recorded, "
                         f"GPS used ({gps*1000:.0f}m)")

    def _agrees(a: float, b: float) -> bool:
        hi = max(abs(a), abs(b))                  # zgodne = różnica <= 10%
        return hi == 0 or abs(a - b) / hi <= GPS_AGREE_FRAC

    def _witnesses(value: float) -> list[str]:
        w = []                                    # kto potwierdza: GPS i/lub tylne
        if gps is not None and _agrees(gps, value):
            w.append(f"GPS={gps*1000:.0f}m")
        if _rear_confirms(value, rears):
            w.append("rears")
        return w

    if len(fronts) == 2:
        lo, hi = min(fl, fr), max(fl, fr)
        disagree = hi > 0 and (hi - lo) > WHEEL_DISAGREE_MIN_KM and (hi - lo) / hi > WHEEL_DISAGREE_FRAC
        if not disagree:
            return fl, "ok (FL)"
        # Realna niezgoda: jeden czujnik padł (zaniża) -> działające = wyższe.
        pct = (hi - lo) / hi * 100
        detail = f"FL={fl*1000:.0f} FR={fr*1000:.0f} m differ {pct:.0f}%"
        w = _witnesses(hi)
        if w:
            return hi, f"ok (recovered): {detail}, used higher wheel, confirmed by {', '.join(w)}"
        gps_note = f", GPS={gps*1000:.0f}m disagrees" if gps is not None else ", no GPS"
        return hi, f"CHECK: {detail}{gps_note}, used higher wheel, no confirmation"

    if len(fronts) == 1:
        pos, val = next(iter(fronts.items()))
        w = _witnesses(val)
        if w:
            return val, f"ok (recovered): only front {pos}={val*1000:.0f}m, confirmed by {', '.join(w)}"
        return val, f"CHECK: only one front sensor ({pos}), no confirmation"

    # Brak przednich -> GPS, jeśli jest.
    if gps is not None:
        if dead_fronts:
            return gps, (f"ok (recovered): front {'+'.join(dead_fronts)} dead vs "
                         f"GPS, used GPS ({gps*1000:.0f}m)")
        return gps, "ok (GPS, no front wheel data)"
    if rears:
        # Tylko koła tylne (napędowe) — możliwa hamownia (kręcą się, auto stoi).
        # Nie liczymy z nich; plik nie wchodzi do sumy (km=None), ale jest wypisany.
        return None, "error: rear-only (no front/GPS) -> excluded, possible dyno"

    # Brak kół i GPS -> ogólne kanały prędkości (SPEED_PRIORITY).
    for keyword in SPEED_PRIORITY:
        for idx, nm in enumerate(sess.lnames):
            if keyword not in nm:
                continue
            samples = sess.samples(idx)
            if len(samples) < 2 or not _has_signal(samples):
                continue
            ch = channels[idx]
            return _integrate_speed(samples, ch["rate"], ch["unit"]), f"ok ({ch['name']})"

    return None, "error: no speed channel"


# ── Wyszukiwanie plików ─────────────────────────────────────────

def discover(paths: list[str], recursive: bool) -> list[Path]:
    """Lista plików .ld ze ścieżek, folderów i wzorców (posortowana, bez duplikatów)."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():                                   # folder
            found.extend(p.glob("**/*.ld" if recursive else "*.ld"))
        elif "*" in str(p) or "?" in str(p):             # wzorzec, np. logs/*.ld
            parent = p.parent if str(p.parent) != "." else Path(".")
            found.extend(f for f in parent.glob(p.name) if f.suffix.lower() == ".ld")
        elif p.is_file() and p.suffix.lower() == ".ld":  # pojedynczy plik
            found.append(p)
    return sorted(set(found))


# ── Interfejs wiersza poleceń ───────────────────────────────────

def _worker(path: Path) -> tuple[Path, Optional[float], str]:
    """Opakowanie dla puli procesów: nigdy nie rzuca wyjątkiem."""
    try:
        km, status = parse_ld(path)
    except Exception:
        km, status = None, "error: exception"
    return path, km, status


def main() -> None:
    ap = argparse.ArgumentParser(description="Distance from MoTeC .ld files (optimized).")
    ap.add_argument("paths", nargs="+", help="File(s), folder(s), or globs")
    ap.add_argument("-r", "--recursive", action="store_true")
    ap.add_argument("--unit", choices=["km","miles","m"], default="km")
    ap.add_argument("-o", "--output", metavar="FILE")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                    help="parallel worker processes (default 1; try 4-8 for "
                         "large batches). I/O-bound past your disk's throughput.")
    args = ap.parse_args()

    files = discover(args.paths, args.recursive)
    if not files:
        print("No .ld files found.", file=sys.stderr); sys.exit(1)

    results: list[tuple[Path, Optional[float], str]] = []
    col_w = max(len(str(f)) for f in files) + 2

    if not args.quiet:
        print(f"\n{'File':<{col_w}}  {'Distance':>12}  Status")
        print("-" * (col_w + 26))

    def emit(f: Path, km: Optional[float], status: str) -> None:
        # zapisz wynik i wypisz wiersz (lub licznik postępu w trybie cichym)
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
        # map() zachowuje kolejność wejścia -> powtarzalne wiersze i CSV
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for f, km, status in ex.map(_worker, files, chunksize=4):
                emit(f, km, status)
    else:
        for f in files:
            _, km, status = _worker(f)
            emit(f, km, status)
    if args.quiet:
        print(file=sys.stderr)

    # podsumowanie: ile policzono, ile odzyskano przez GPS, ile do sprawdzenia
    valid     = [(f,km) for f,km,_ in results if km is not None]
    recovered = [(f,s) for f,km,s in results if km is not None
                 and (s.startswith("ok (recovered") or s.startswith("ok (GPS primary"))]
    flagged   = [(f,s) for f,km,s in results if km is not None and s.startswith("CHECK")]
    total_km  = sum(km for _,km in valid)
    tv, lab   = to_display(total_km, args.unit)

    print(f"\nFiles analyzed      : {len(files)}")
    print(f"Parsed OK           : {len(valid)}")
    print(f"Failed              : {len(files) - len(valid)}")
    print(f"Auto-recovered      : {len(recovered)}  (dead/disagreeing wheel, recovered via GPS/rears)")
    print(f"Needs review        : {len(flagged)}  (no independent confirmation)")
    print(f"Total distance      : {tv:.3f} {lab}\n")

    if flagged and not args.quiet:
        print("Sessions to verify manually:")
        for f, s in flagged:
            print(f"  {f.name}: {s}")
        print()

    if args.output:                                  # opcjonalny zapis do CSV
        out = Path(args.output)
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", f"distance_{lab}", "status", "review"])
            for f, km, status in results:
                review = "REVIEW" if status.startswith("CHECK") else ""
                if km is None: w.writerow([str(f), "", status, review])
                else:
                    v, _ = to_display(km, args.unit)
                    w.writerow([str(f), f"{v:.4f}", status, review])
            w.writerow([])
            w.writerow(["TOTAL", f"{tv:.4f}",
                        f"{len(valid)}/{len(files)} parsed", f"{len(flagged)} to review"])
        print(f"Results saved -> {out}")

if __name__ == "__main__":
    main()
