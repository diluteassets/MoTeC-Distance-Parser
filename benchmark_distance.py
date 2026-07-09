#!/usr/bin/env python3
"""Benchmark: motec_distance vs motec_distance_optimized na syntetycznych .ld.

Buduje duże pliki .ld (generator z test_optimized_parity) i mierzy parse_ld
obu wersji na dwóch profilach:
  * healthy  — typowa zdrowa sesja (4 koła + 1 GPS, wynik "ok (FL)"),
  * worst    — ścieżka najdroższa w oryginale (2 niezgodne odbiorniki GPS +
               GPS główny): oryginał liczy fuzję GPS dwa razy i szczyty kół
               trzy razy; wersja z cache — wszystko raz.

Użycie: python benchmark_distance.py [-n PRÓBKI_NA_KANAŁ] [--repeats R] [--no-numpy]
"""
import argparse
import math
import tempfile
import time
from pathlib import Path

import motec_distance as orig
import motec_distance_optimized as opt
from test_optimized_parity import build_ld, ch, _gps


def _profiles(n: int) -> dict[str, list[dict]]:
    kmh = [50.0 + 10.0 * ((i % 200) / 200.0) for i in range(n)]   # 50-60 km/h
    return {
        "healthy": (
            [ch("Wheel_Speed_FL", kmh), ch("Wheel_Speed_FR", kmh),
             ch("Wheel_Speed_RL", kmh), ch("Wheel_Speed_RR", kmh)]
            + _gps("GPS_", kmh)),
        "worst": (
            [ch("Wheel_Speed_FL", [50.0] * n), ch("Wheel_Speed_FR", [50.0] * n),
             ch("Wheel_Speed_RL", [50.0] * n), ch("Wheel_Speed_RR", [50.0] * n)]
            + _gps("GPS_A_", [60.0] * n) + _gps("GPS_B_", [95.0] * n)),
    }


def _time(fn, path: Path, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(path)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=1_000_000, help="próbek na kanał")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-numpy", action="store_true",
                    help="zmierz też ścieżkę zapasową bez numpy")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print(f"\n{args.n:,} próbek/kanał, najlepszy z {args.repeats} przebiegów\n")
        print(f"{'profil':22} {'oryginał':>10} {'optymalizowany':>15} {'przyspieszenie':>15}")
        print("-" * 66)
        for name, chans in _profiles(args.n).items():
            p = tmp / f"{name}.ld"
            p.write_bytes(build_ld(chans))
            r_orig, r_opt = orig.parse_ld(p), opt.parse_ld(p)
            # wersja zoptymalizowana sumuje na intach (bezstratnie), więc wynik
            # może różnić się od float-owej akumulacji oryginału na ostatnich bitach
            assert r_orig[1] == r_opt[1] and math.isclose(
                r_orig[0], r_opt[0], rel_tol=1e-9), \
                f"{name}: wyniki się różnią! {r_orig} vs {r_opt}"
            t_orig = _time(orig.parse_ld, p, args.repeats)
            t_opt = _time(opt.parse_ld, p, args.repeats)
            print(f"{name:22} {t_orig*1000:8.1f}ms {t_opt*1000:13.1f}ms {t_orig/t_opt:14.2f}x")
            if args.no_numpy:
                orig._HAVE_NP = opt._HAVE_NP = False
                try:
                    t_orig = _time(orig.parse_ld, p, args.repeats)
                    t_opt = _time(opt.parse_ld, p, args.repeats)
                finally:
                    orig._HAVE_NP = opt._HAVE_NP = True
                print(f"{name + ' (bez numpy)':22} {t_orig*1000:8.1f}ms "
                      f"{t_opt*1000:13.1f}ms {t_orig/t_opt:14.2f}x")
        print()


if __name__ == "__main__":
    main()
