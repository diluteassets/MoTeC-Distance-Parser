#!/usr/bin/env python3
""" pisane ze wsparciem Claude Code, więc proszę o wyrozumiałość ;)

"""

"""
motec_distance_optimized.py — zoptymalizowana kopia motec_distance.py.

Ta sama logika wyboru źródła i te same statusy; wyniki liczbowo identyczne
w granicach zaokrągleń float (różnice < 1e-12 względnie, dokładność wręcz
lepsza — patrz całkowanie na intach niżej). Co zmieniono względem oryginału:

  * CAŁKOWANIE NA SUROWYCH INTACH: przy stałym kroku czasowym suma trapezów
    to dt * (suma - (pierwsza+ostatnia)/2), a suma wartości fizycznych to
    factor * suma(raw) + n * shift. Gdy wszystkie próbki są w zakresie
    (typowy przypadek — sprawdzane przez min/max na intach), dystans liczymy
    JEDNĄ redukcją int64 po surowych danych: zero tablic float64, zero kopii,
    ~8x mniej ruchu pamięci niż astype+maska+trapz w oryginale. Suma int jest
    bezstratna, więc to dokładniejsze niż akumulacja float.
  * cache na sesję (_Session): surowe próbki to widoki numpy wprost na mmap
    (zero kopii), każdy kanał dotykany raz — oryginał dekodował koła i GPS
    nawet 3x (dystans / szczyt / test zaszumienia), a fuzję GPS liczył 2x,
  * wartości float materializowane leniwie, tylko gdy naprawdę potrzebne
    (naprawa próbek poza zakresem, szczyty prędkości, składowe GPS),
  * test sygnału bez materializacji: kanał jest pusty ⇔ min==max surowych
    i ta jedna wartość fizyczna == 0.0 (równoważne .any() oryginału),
  * potok GPS: operacje w miejscu (2 alokacje zamiast ~6), percentyl szczytu
    liczony tylko, gdy odbiorników jest >= 2 albo trwa test zaszumienia,
  * deskryptory kanałów: jeden prekompilowany struct.Struct zamiast 9 wywołań,
  * mmap.madvise(WILLNEED) na blokach próbek — readahead przy zimnym cache'u,
  * zapas bez numpy: array.frombytes + C-owe sum/min/max zamiast pętli
    struct.unpack_from; szybka ścieżka int działa też tutaj,
  * POPRAWKA błędu z oryginału: bez numpy `lista * 3.6` rzucała TypeError,
    gdy kanał koła był w m/s, a dwa odbiorniki GPS się nie zgadzały.

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
except ImportError:                       # pragma: no cover
    _np = None
    _HAVE_NP = False


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

# Pola 0x04-0x1F jednym unpackiem: next, data, n, (0x10-0x13 pomijane),
# dsize, rate, shift, mul, scale, dec.
_DESC = struct.Struct("<III4xHHhhHh")

# Odczyt napisu / liczb spod offsetu w buforze.
def _rstr(d, o, n): return d[o:o+n].rstrip(b"\x00").decode("ascii","replace").strip()
def _ru32(d, o):    return struct.unpack_from("<I", d, o)[0]


def _madvise_willneed(data, base: int, length: int) -> None:
    """Podpowiedź readahead dla bloku próbek (nic nie robi, gdy niedostępne)."""
    try:
        page = mmap.PAGESIZE
        start = base - base % page
        data.madvise(mmap.MADV_WILLNEED, start, length + (base - start))
    except (AttributeError, ValueError, OSError):    # pragma: no cover
        pass


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

_WHEEL_POSITIONS = ("FL", "FR", "RL", "RR")


class _Session:
    """Kontekst jednej sesji .ld: bufor + kanały + cache wyników pochodnych.

    Surowe próbki to widoki numpy wprost na mmap (bez kopiowania); statystyki
    (min/max), wartości fizyczne, dystanse kół, fuzja GPS i szczyt prędkości
    kół są liczone najwyżej raz. Oryginał powtarzał te przebiegi przy każdym
    teście spójności — to główne źródło przyspieszenia.
    """
    __slots__ = ("data", "channels", "lnames", "_raw", "_stats", "_phys",
                 "_wheel_idx", "_wheel_dists", "_wheel_peak",
                 "_gps_fused", "_gps_fused_done")

    def __init__(self, data, channels: list[dict]):
        self.data = data
        self.channels = channels
        self.lnames = [ch["name"].lower() for ch in channels]   # lower() raz
        self._raw:   dict[int, object] = {}   # idx -> surowe próbki (int) | None
        self._stats: dict[int, tuple]  = {}   # idx -> (min_raw, max_raw)
        self._phys:  dict[int, object] = {}   # idx -> wartości fizyczne (float)
        self._wheel_idx:   dict[str, Optional[int]]   = {}
        self._wheel_dists: dict[str, Optional[float]] = {}
        self._wheel_peak:  Optional[float] = None
        self._gps_fused = None
        self._gps_fused_done = False

    # ── warstwa danych ──────────────────────────────────────────
    def raw(self, idx: int):
        """Surowe próbki kanału: widok numpy na mmap (zero kopii) / array.array,
        albo None gdy kanał nieczytelny (zły rozmiar próbki, poza plikiem)."""
        if idx in self._raw:
            return self._raw[idx]
        ch = self.channels[idx]
        n, base, size = ch["n"], ch["data_addr"], ch["dsize"]
        r = None
        if size in (2, 4) and n > 0 and base + n * size <= len(self.data):
            if _HAVE_NP:
                _madvise_willneed(self.data, base, n * size)
                r = _np.frombuffer(self.data, dtype="<i4" if size == 4 else "<i2",
                                   count=n, offset=base)
            else:
                # array.frombytes dekoduje cały blok naraz; sum/min/max na
                # array chodzą bez budowania listy floatów
                arr = array.array("i" if size == 4 else "h")
                if arr.itemsize == size:
                    arr.frombytes(self.data[base:base + n * size])
                    if sys.byteorder == "big":       # pragma: no cover
                        arr.byteswap()
                    r = arr
                else:                                # pragma: no cover
                    fmt = "<i" if size == 4 else "<h"
                    r = [v for (v,) in
                         struct.iter_unpack(fmt, self.data[base:base + n * size])]
        self._raw[idx] = r
        return r

    def stats(self, idx: int) -> tuple:
        """(min, max) surowych próbek — tanie przebiegi po intach, liczone raz."""
        st = self._stats.get(idx)
        if st is None:
            r = self.raw(idx)
            if _HAVE_NP and isinstance(r, _np.ndarray):
                st = (int(r.min()), int(r.max()))
            else:
                st = (min(r), max(r))
            self._stats[idx] = st
        return st

    def has_signal(self, idx: int) -> bool:
        """Czy kanał ma niezerową próbkę fizyczną — bez materializacji floatów.

        Równoważne .any() oryginału: dla factor != 0 wartość rośnie/maleje
        monotonicznie z raw, więc przy min != max co najwyżej jedna wartość
        raw daje dokładnie 0.0; przy min == max liczymy tę jedną wartość
        tak samo, jak zrobiłby to oryginał."""
        r = self.raw(idx)
        if r is None or len(r) == 0:
            return False
        ch = self.channels[idx]
        f, s = ch["factor"], ch["shift"]
        if f == 0.0:                                 # zdegenerowany dec -> stała
            return bool(s)
        rmin, rmax = self.stats(idx)
        if rmin != rmax:
            return True
        return rmin * f + s != 0.0

    def phys(self, idx: int):
        """Wartości fizyczne kanału (float64/lista) — leniwie, tylko gdy trzeba
        (naprawa próbek poza zakresem, szczyty, składowe GPS)."""
        p = self._phys.get(idx)
        if p is None:
            r = self.raw(idx)
            ch = self.channels[idx]
            f, s = ch["factor"], ch["shift"]
            if _HAVE_NP and isinstance(r, _np.ndarray):
                p = r * f                            # od razu float64, jedna alokacja
                if s:
                    p += s
            elif s:
                p = [x * f + s for x in r]
            else:
                p = [x * f for x in r]
            self._phys[idx] = p
        return p

    # ── koła ────────────────────────────────────────────────────
    def wheel_idx(self, pos: str) -> Optional[int]:
        """Indeks kanału Wheel_Speed_<pos> z sygnałem, albo None."""
        key = pos.lower()
        if key in self._wheel_idx:
            return self._wheel_idx[key]
        target = f"wheel_speed_{key}"
        found = None
        for idx, nm in enumerate(self.lnames):
            if target in nm:
                r = self.raw(idx)
                if r is not None and len(r) >= 2 and self.has_signal(idx):
                    found = idx
                    break
        self._wheel_idx[key] = found
        return found

    def wheel_dist(self, pos: str) -> Optional[float]:
        """Dystans (km) z Wheel_Speed_<pos>, albo None gdy brak/pusty."""
        key = pos.lower()
        if key not in self._wheel_dists:
            idx = self.wheel_idx(pos)
            self._wheel_dists[key] = None if idx is None \
                else _channel_distance(self, idx)
        return self._wheel_dists[key]

    def wheel_kmh(self, pos: str):
        """(prędkość_km/h, rate) — floaty materializowane tylko dla szczytów."""
        idx = self.wheel_idx(pos)
        if idx is None:
            return None
        ch = self.channels[idx]
        p = self.phys(idx)
        if "m/s" in ch["unit"].lower():
            # poprawka: na liście (bez numpy) mnożymy element po elemencie
            p = p * 3.6 if _HAVE_NP and isinstance(p, _np.ndarray) \
                else [x * 3.6 for x in p]
        return p, (ch["rate"] or 100)

    def wheel_peak(self) -> float:
        """Najwyższy szczyt prędkości z czterech kół (km/h)."""
        if self._wheel_peak is None:
            wp = 0.0
            for pos in _WHEEL_POSITIONS:
                w = self.wheel_kmh(pos)
                if w is not None:
                    wp = max(wp, _peak_kmh(w[0]))
            self._wheel_peak = wp
        return self._wheel_peak

    # ── GPS ─────────────────────────────────────────────────────
    def gps_fused(self):
        """Wynik _gps_fused liczony raz (może być None)."""
        if not self._gps_fused_done:
            self._gps_fused = _gps_fused(self)
            self._gps_fused_done = True
        return self._gps_fused


def _channel_distance(sess: _Session, idx: int) -> float:
    """Dystans (km) z kanału prędkości; próbki poza zakresem: trzymaj ostatnią dobrą.

    Szybka ścieżka: gdy min/max surowych próbek mieści cały kanał w
    [0, MAX_SPEED_KMH], suma trapezów przy stałym dt redukuje się do
    dt * (factor*suma(raw) + n*shift - (v0+v_ost)/2) — jedna redukcja
    int64, bez tablic float. Konwersję m/s składamy w factor/shift."""
    ch = sess.channels[idx]
    r = sess.raw(idx)
    n = len(r)
    if n < 2:
        return 0.0
    rate = ch["rate"]
    dt = 1.0 / (rate if rate > 0 else 100)
    f, s = ch["factor"], ch["shift"]
    if "m/s" in ch["unit"].lower():
        f, s = f * 3.6, s * 3.6
    rmin, rmax = sess.stats(idx)
    a, b = rmin * f, rmax * f                        # factor może być ujemny
    lo, hi = min(a, b) + s, max(a, b) + s
    if lo >= 0.0 and hi <= MAX_SPEED_KMH:
        if _HAVE_NP and isinstance(r, _np.ndarray):
            total = int(r.sum(dtype=_np.int64))      # bezstratna suma int
        else:
            total = sum(r)
        ends = (r[0] * f + s) + (r[-1] * f + s)
        return (f * total + n * s - 0.5 * ends) * dt / 3600.0
    return _integrate_speed(sess.phys(idx), rate, ch["unit"])


def _integrate_speed(samples, rate_hz: int, unit: str = "") -> float:
    """Całkuje prędkość (trapezy) -> km, naprawiając próbki poza zakresem.
    Wolna ścieżka _channel_distance — semantyka identyczna z oryginałem."""
    if len(samples) < 2:
        return 0.0
    dt = 1.0 / (rate_hz if rate_hz > 0 else 100)
    in_ms = "m/s" in unit.lower()                    # część kanałów jest w m/s

    if _HAVE_NP and isinstance(samples, _np.ndarray):
        s = samples * 3.6 if in_ms else samples
        good = (s >= 0.0) & (s <= MAX_SPEED_KMH)
        if not good.all():
            # wypełnij w przód ostatnią dobrą wartością (wektorowa wersja pętli niżej)
            last = _np.where(good, _np.arange(s.size), -1)
            _np.maximum.accumulate(last, out=last)
            clean = _np.zeros(s.size)
            seen = last >= 0
            clean[seen] = s[last[seen]]
            s = clean
        # suma trapezów przy stałym dt = suma - połowa końców
        return float((s.sum() - 0.5 * (s[0] + s[-1])) * dt / 3600.0)

    # bez numpy: jedna pętla, suma trapezów bez listy pośredniej
    it = (x * 3.6 for x in samples) if in_ms else samples
    last = prev = 0.0
    acc = 0.0
    first = True
    for x in it:
        if 0.0 <= x <= MAX_SPEED_KMH:                # dobra próbka -> zapamiętaj
            last = x
        if not first:
            acc += prev + last                       # 2 * pole trapezu / dt
        prev = last
        first = False
    return acc * dt / 7200.0                         # /2 (trapez) i /3600 (h -> s)


def _peak_kmh(samples) -> float:
    """Szczyt prędkości (99.5 percentyl) — odporny na pojedynczy glitch."""
    if _HAVE_NP and isinstance(samples, _np.ndarray):
        return float(_np.percentile(_np.clip(samples, 0, MAX_SPEED_KMH), 99.5))
    vals = sorted(x for x in samples if 0 <= x <= MAX_SPEED_KMH)
    return vals[int(0.995 * (len(vals) - 1))] if vals else 0.0


# Skalarny GPS_Speed (auta bez velX/velY). Pomijamy kanały zadane/komendy
# (target/command...), które tylko mają w nazwie gps+speed.
_GPS_SCALAR_EXCLUDE = ("command", "setpoint", "target", "limit", "max", "min", "error")


def _component_gps_units(sess: _Session) -> list[dict]:
    """Lista odbiorników velX/velY: {sp(km/h), rate, dist(km), peak, n}.
    Grupowane po przedrostku nazwy (np. Xsens_MTI_680_). Percentyl szczytu
    (peak) liczony leniwie — potrzebny tylko przy >= 2 odbiornikach."""
    groups: dict[str, dict[str, int]] = {}
    for idx, nm in enumerate(sess.lnames):
        if nm.endswith("velx") or nm.endswith("vely"):
            groups.setdefault(sess.channels[idx]["name"][:-4], {})[nm[-4:]] = idx
    units: list[dict] = []
    lo_kmh = GPS_DEADBAND_MS * 3.6
    for axes in groups.values():
        if "velx" not in axes or "vely" not in axes:   # potrzebne obie osie
            continue
        ix, iy = axes["velx"], axes["vely"]
        rx, ry = sess.raw(ix), sess.raw(iy)
        if rx is None or ry is None:
            continue
        n = min(len(rx), len(ry))
        if n < 2 or not (sess.has_signal(ix) or sess.has_signal(iy)):
            continue
        rate = sess.channels[ix]["rate"] or 100
        dt = 1.0 / rate
        if _HAVE_NP and isinstance(rx, _np.ndarray):
            chx, chy = sess.channels[ix], sess.channels[iy]
            a = rx[:n] * chx["factor"]               # dalej wszystko w miejscu
            if chx["shift"]:
                a += chx["shift"]
            a *= a
            b = ry[:n] * chy["factor"]
            if chy["shift"]:
                b += chy["shift"]
            b *= b
            a += b
            _np.sqrt(a, out=a)
            a *= 3.6                                 # prędkość pozioma km/h
            sp = a
            if float(sp.max()) <= MAX_SPEED_KMH and \
                    (lo_kmh <= 0.0 or float(sp.min()) >= lo_kmh):
                dist = float((sp.sum() - 0.5 * (sp[0] + sp[-1])) * dt / 3600.0)
            else:
                keep = _np.where((sp >= lo_kmh) & (sp <= MAX_SPEED_KMH), sp, 0.0)
                dist = float((keep.sum() - 0.5 * (keep[0] + keep[-1])) * dt / 3600.0)
        else:
            vx, vy = sess.phys(ix), sess.phys(iy)
            sp = [((vx[i] * vx[i] + vy[i] * vy[i]) ** 0.5) * 3.6 for i in range(n)]
            keep = [s if lo_kmh <= s <= MAX_SPEED_KMH else 0.0 for s in sp]
            dist = sum((keep[i] + keep[i+1]) / 2.0 * (dt / 3600.0) for i in range(n - 1))
        units.append({"sp": sp, "rate": rate, "dist": dist, "peak": None, "n": n})
    return units


def _unit_peak(u: dict) -> float:
    """Szczyt prędkości odbiornika — percentyl liczony przy pierwszym użyciu."""
    if u["peak"] is None:
        u["peak"] = _peak_kmh(u["sp"])
    return u["peak"]


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
        rep = a if _unit_peak(a) <= _unit_peak(b) else b   # czystszy przebieg
        return (a["dist"] + b["dist"]) / 2.0, rep["sp"], rep["rate"]
    wp = sess.wheel_peak()                       # niezgodne -> bliżej szczytu kół
    pick = a if abs(_unit_peak(a) - wp) <= abs(_unit_peak(b) - wp) else b
    return pick["dist"], pick["sp"], pick["rate"]


def _gps_distance(sess: _Session) -> Optional[float]:
    """Dystans (km) z GPS: składowe velX/velY (z fuzją) lub skalarny GPS_Speed."""
    fused = sess.gps_fused()
    if fused is not None:
        return fused[0]
    # zapas: skalarny kanał prędkości GPS
    for idx, nm in enumerate(sess.lnames):
        if "gps" in nm and "speed" in nm and not any(x in nm for x in _GPS_SCALAR_EXCLUDE):
            r = sess.raw(idx)
            if r is not None and len(r) >= 2 and sess.has_signal(idx):
                return _channel_distance(sess, idx)
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
        nxt, data_addr, n, dsize, rate, shift, mul, scale, dec = \
            _DESC.unpack_from(data, addr + _C_NEXT)
        mul = mul or 1
        scale = scale or 1
        channels.append({
            "name":      _rstr(data, addr + _C_NAME, 32),
            "unit":      _rstr(data, addr + _C_UNIT, 12),
            "n":         n,
            "dsize":     dsize,
            "rate":      rate,
            "shift":     shift,
            "mul":       mul,
            "scale":     scale,
            "dec":       dec,
            "data_addr": data_addr,
            "factor":    (mul / scale) * (10.0 ** (-dec)),   # skalowanie, raz
        })
        addr = nxt

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
            r = sess.raw(idx)
            if r is None or len(r) < 2 or not sess.has_signal(idx):
                continue
            return _channel_distance(sess, idx), f"ok ({channels[idx]['name']})"

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
