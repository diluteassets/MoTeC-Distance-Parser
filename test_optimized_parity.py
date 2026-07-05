#!/usr/bin/env python3
"""Test parytetu: motec_distance_optimized musi dawać TE SAME wyniki co oryginał.

W repo nie ma prawdziwych plików .ld (test regresji je pomija), więc budujemy
syntetyczne pliki .ld w formacie, który czyta parser (nagłówek 0x40, lista
wiązana deskryptorów 124 B, próbki i2/i4), i sprawdzamy:
  1. parytet: (km, status) z motec_distance == motec_distance_optimized
     dla każdego scenariusza (koła zdrowe/padłe/niezgodne, fuzja GPS, GPS
     główny, GPS zaszumiony, hamownia, kanały zapasowe, próbki poza zakresem,
     jednostki m/s, skalowanie mul/scale/dec/shift, pliki uszkodzone),
  2. wartości bezwzględne: stała prędkość 60 km/h przez 60 s -> 1.000 km,
  3. ścieżka bez numpy (fallback) w wersji zoptymalizowanej daje to samo, co
     ścieżka numpy — w tym scenariusz, na którym ORYGINALNY fallback się
     wywalał (koła w m/s + dwa niezgodne odbiorniki GPS -> lista * 3.6).

Uruchom samodzielnie:   python test_optimized_parity.py
albo przez pytest:      pytest test_optimized_parity.py
"""
import math
import struct
import tempfile
from pathlib import Path

import motec_distance as orig
import motec_distance_optimized as opt

_HDR = 0x100          # rozmiar nagłówka; wskaźnik na 1. deskryptor pod 0x08
_DESC = 0x7C          # rozmiar rekordu deskryptora kanału


def build_ld(chans: list[dict]) -> bytes:
    """Składa minimalny plik .ld: nagłówek + lista wiązana kanałów + próbki.

    Każdy kanał: {name, unit, rate, dsize, raw (lista int), mul, scale, dec, shift}.
    """
    desc_base = _HDR
    data_base = desc_base + len(chans) * _DESC
    offs, cur = [], data_base
    for c in chans:
        offs.append(cur)
        cur += len(c["raw"]) * c.get("dsize", 2)
    buf = bytearray(cur)
    buf[0] = 0x40
    struct.pack_into("<I", buf, 0x08, desc_base if chans else 0)
    for i, c in enumerate(chans):
        a = desc_base + i * _DESC
        dsize = c.get("dsize", 2)
        struct.pack_into("<I", buf, a + 0x04,
                         desc_base + (i + 1) * _DESC if i + 1 < len(chans) else 0)
        struct.pack_into("<I", buf, a + 0x08, offs[i])
        struct.pack_into("<I", buf, a + 0x0C, len(c["raw"]))
        struct.pack_into("<H", buf, a + 0x14, dsize)
        struct.pack_into("<H", buf, a + 0x16, c.get("rate", 100))
        struct.pack_into("<h", buf, a + 0x18, c.get("shift", 0))
        struct.pack_into("<h", buf, a + 0x1A, c.get("mul", 1))
        struct.pack_into("<H", buf, a + 0x1C, c.get("scale", 1))
        struct.pack_into("<h", buf, a + 0x1E, c.get("dec", 0))
        nb = c["name"].encode("ascii")[:32]
        buf[a + 0x20:a + 0x20 + len(nb)] = nb
        ub = c.get("unit", "km/h").encode("ascii")[:12]
        buf[a + 0x48:a + 0x48 + len(ub)] = ub
        fmt = "<i" if dsize == 4 else "<h"
        for j, v in enumerate(c["raw"]):
            struct.pack_into(fmt, buf, offs[i] + j * dsize, v)
    return bytes(buf)


def ch(name, kmh, unit="km/h", rate=100, dec=0, **kw) -> dict:
    """Kanał prędkości: wartości fizyczne -> surowe inty przez 10^dec."""
    return {"name": name, "unit": unit, "rate": rate, "dec": dec,
            "raw": [round(v * 10 ** dec) for v in kmh], **kw}


N = 6000                      # 60 s przy 100 Hz
C60 = [60.0] * N              # stała 60 km/h -> dokładnie 1.000 km
C58 = [58.0] * N
C40 = [40.0] * N
C30 = [30.0] * N
MS_60 = [16.666] * N          # ~60 km/h w m/s (dec=3 -> raw 16666, mieści się w i2)


def _gps(prefix, kmh_vals):
    """Para kanałów velX/velY jednego odbiornika (velY = 0)."""
    ms = [v / 3.6 for v in kmh_vals]
    return [ch(f"{prefix}VelX", ms, unit="m/s", dec=3),
            ch(f"{prefix}VelY", [0.0] * len(ms), unit="m/s", dec=3)]


# Scenariusze: nazwa -> (kanały, oczekiwane_km albo None, fragment statusu)
SCENARIOS = {
    "healthy_four_wheels": (
        [ch("Wheel_Speed_FL", C60), ch("Wheel_Speed_FR", C60),
         ch("Wheel_Speed_RL", C60), ch("Wheel_Speed_RR", C60)],
        1.0, "ok (FL)"),
    "wheels_i4_scaled": (            # próbki 4-bajtowe + mul/scale/dec/shift
        [ch("Wheel_Speed_FL", [59.0] * N, dsize=4, dec=2, mul=2, scale=4, shift=1),
         ch("Wheel_Speed_FR", [59.0] * N, dsize=4, dec=2, mul=2, scale=4, shift=1)],
        # wartość = raw * 2/4 * 10^-2 + 1: raw z ch() liczone dla 59 -> fiz. 30.5
        None, "ok (FL)"),
    "ms_unit_wheels": (
        [ch("Wheel_Speed_FL", MS_60, unit="m/s", dec=3),
         ch("Wheel_Speed_FR", MS_60, unit="m/s", dec=3)],
        0.99996, "ok (FL)"),
    "glitchy_samples_held": (        # 300 km/h poza zakresem -> trzymaj ostatnią dobrą
        [ch("Wheel_Speed_FL", C60[:3000] + [300.0] * 100 + C60[:2900]),
         ch("Wheel_Speed_FR", C60)],
        1.0, "ok (FL)"),
    "front_disagree_gps_confirms": (
        [ch("Wheel_Speed_FL", C60), ch("Wheel_Speed_FR", C30)] + _gps("GPS_", C60),
        1.0, "ok (recovered)"),
    "front_disagree_no_witness": (
        [ch("Wheel_Speed_FL", C60), ch("Wheel_Speed_FR", C30)],
        1.0, "CHECK"),
    "dead_front_vs_gps": (           # FL pełza (5 km/h) -> martwe, zostaje FR
        [ch("Wheel_Speed_FL", [5.0] * N), ch("Wheel_Speed_FR", C60)] + _gps("GPS_", C60),
        1.0, "ok (recovered): only front FR"),
    "both_fronts_dead_gps_takes_over": (
        [ch("Wheel_Speed_FL", [5.0] * N), ch("Wheel_Speed_FR", [5.0] * N)]
        + _gps("GPS_", C60),
        None, "ok (recovered): front FL+FR dead"),
    "gps_primary_wheel_under": (     # koło zgubiło dystans -> GPS przejmuje
        [ch("Wheel_Speed_FL", C40), ch("Wheel_Speed_FR", C40)] + _gps("GPS_", C60),
        None, "ok (GPS primary)"),
    "gps_spiky_kept_wheel": (        # szczyt GPS >> koła -> zostaje koło + CHECK
        [ch("Wheel_Speed_FL", [55.0] * N), ch("Wheel_Speed_FR", [55.0] * N)]
        + _gps("GPS_", [100.0] * N),
        None, "CHECK: GPS velocity spiky"),
    "two_gps_agree_averaged": (
        _gps("Xsens_670_", C60) + _gps("Xsens_680_", C58),
        (1.0 + 58 / 60) / 2, "ok (GPS, no front wheel data)"),
    "two_gps_disagree_pick_near_wheels": (   # tylne koło rozstrzyga, który czysty
        _gps("GPS_A_", C60) + _gps("GPS_B_", [100.0] * N)
        + [ch("Wheel_Speed_RL", C60), ch("Wheel_Speed_RR", C60)],
        None, "ok"),
    "rear_only_dyno": (
        [ch("Wheel_Speed_RL", C60), ch("Wheel_Speed_RR", C60)],
        None, "error: rear-only"),
    "scalar_gps_speed": (
        [ch("GPS Speed", C60)],
        1.0, "ok (GPS, no front wheel data)"),
    "scalar_gps_excludes_setpoint": (        # kanał zadany pomijany, brak źródeł
        [ch("GPS_Speed_Target", C60, rate=0)],
        None, "error: no speed channel"),
    "speed_priority_fallback": (
        [ch("Vehicle Speed", C60)],
        1.0, "ok (Vehicle Speed)"),
    "empty_channels_skipped": (              # kanał bez sygnału -> następne źródło
        [ch("Wheel_Speed_FL", [0.0] * N), ch("Ground Speed", C60)],
        1.0, "ok (Ground Speed)"),
    "no_channels_at_all": ([], None, "error:"),
    # scenariusz błędu oryginalnego fallbacku: koła w m/s + 2 niezgodne GPS
    "ms_wheels_two_gps_disagree": (
        [ch("Wheel_Speed_FL", MS_60, unit="m/s", dec=3),
         ch("Wheel_Speed_FR", MS_60, unit="m/s", dec=3)]
        + _gps("GPS_A_", C60) + _gps("GPS_B_", [100.0] * N),
        None, "ok"),
}

# Uszkodzone pliki: surowe bajty -> oczekiwany status
BROKEN = {
    "empty": b"",
    "too_short": b"\x40" + b"\x00" * 10,
    "bad_magic": b"\x00" * 0x200,
    "bad_channel_ptr": b"\x40" + b"\x00" * 7 + struct.pack("<I", 0xFFFFFF) + b"\x00" * 0x200,
    "self_loop": (lambda b: b)(  # deskryptor wskazujący sam na siebie
        bytes([0x40]) + b"\x00" * 7 + struct.pack("<I", _HDR)
        + b"\x00" * (_HDR - 12)
        + struct.pack("<I", 0) + struct.pack("<I", _HDR) + b"\x00" * (_DESC - 8)),
}


def _write(tmp: Path, name: str, blob: bytes) -> Path:
    p = tmp / f"{name}.ld"
    p.write_bytes(blob)
    return p


def _run_all(tmp: Path):
    """(nazwa, wynik_orig, wynik_opt, oczekiwane) dla każdego scenariusza."""
    rows = []
    for name, (chans, want_km, want_status) in SCENARIOS.items():
        p = _write(tmp, name, build_ld(chans))
        rows.append((name, orig.parse_ld(p), opt.parse_ld(p), want_km, want_status))
    for name, blob in BROKEN.items():
        p = _write(tmp, f"broken_{name}", blob)
        rows.append((name, orig.parse_ld(p), opt.parse_ld(p), None, "error:"))
    return rows


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def test_parity_and_expectations():
    """Optymalizacja nie zmienia wyników; znane scenariusze dają znane wartości."""
    failures = []
    with tempfile.TemporaryDirectory() as td:
        for name, (okm, ost), (nkm, nst), want_km, want_status in _run_all(Path(td)):
            if not _close(okm, nkm) or ost != nst:
                failures.append(f"  {name}: orig=({okm}, {ost!r}) opt=({nkm}, {nst!r})")
                continue
            if want_km is not None and not math.isclose(nkm or 0.0, want_km, rel_tol=1e-3):
                failures.append(f"  {name}: expected ~{want_km} km, got {nkm}")
            if not nst.startswith(want_status):
                failures.append(f"  {name}: expected status {want_status!r}, got {nst!r}")
    assert not failures, "parity/expectation failures:\n" + "\n".join(failures)


# Logika "GPS główny/zaszumiony" celowo wymaga numpy (tak w oryginale, tak tu):
# _gps_speed_kmh zwraca None bez numpy, więc te scenariusze różnią się z założenia.
_NUMPY_ONLY = {"gps_primary_wheel_under", "gps_spiky_kept_wheel"}


def test_pure_python_fallback_matches_numpy():
    """Ścieżka bez numpy w wersji zoptymalizowanej == ścieżka numpy (tolerancja
    float). Obejmuje scenariusz, na którym fallback oryginału rzucał TypeError."""
    failures = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, (chans, _, _) in SCENARIOS.items():
            if name in _NUMPY_ONLY:
                continue
            p = _write(tmp, name, build_ld(chans))
            with_np = opt.parse_ld(p)
            opt._HAVE_NP = False
            try:
                without_np = opt.parse_ld(p)
            finally:
                opt._HAVE_NP = True
            okm, ost = with_np
            nkm, nst = without_np
            km_ok = (okm is None and nkm is None) or (
                okm is not None and nkm is not None
                and math.isclose(okm, nkm, rel_tol=1e-6, abs_tol=1e-9))
            # statusy z liczbami mogą się różnić o zaokrąglenie 1 m; porównaj prefiks
            st_ok = ost.split(":")[0] == nst.split(":")[0]
            if not (km_ok and st_ok):
                failures.append(f"  {name}: numpy=({okm}, {ost!r}) fallback=({nkm}, {nst!r})")
    assert not failures, "fallback mismatches:\n" + "\n".join(failures)


def main() -> int:
    """Uruchomienie samodzielne: tabela wyników i kod wyjścia (0 = wszystko OK)."""
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        rows = _run_all(Path(td))
        print(f"\n{'scenario':38} {'orig km':>10} {'opt km':>10}  parity")
        print("-" * 78)
        for name, (okm, ost), (nkm, nst), _, _ in rows:
            ok = _close(okm, nkm) and ost == nst
            fails += not ok
            o = "-" if okm is None else f"{okm:.5f}"
            n = "-" if nkm is None else f"{nkm:.5f}"
            print(f"{name:38} {o:>10} {n:>10}  {'PASS' if ok else 'FAIL'}")
        print("-" * 78)
    try:
        test_parity_and_expectations()
        test_pure_python_fallback_matches_numpy()
        print("expectation + fallback checks: PASS\n")
    except AssertionError as e:
        print(f"FAIL:\n{e}\n")
        fails += 1
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
