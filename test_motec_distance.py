#!/usr/bin/env python3
"""Test regresji: pilnuje, by motec_distance.parse_ld zgadzał się z "Corr Dist" MoTeC.

Każda wartość niżej to prawdziwy "Corr Dist [m]" odczytany z MoTeC i2 z danej sesji
i sprawdzony ręcznie. Zamrażają one całą pracę nad dokładnością:
  - GPS jako główne źródło (MoTeC idzie za GPS, gdy jest fix),
  - fuzja dwóch GPS (średnia zgodnych; odrzucenie zaszumionego),
  - zabezpieczenie zepsutego GPS (12_39: velY skoczyło do -70 km/h -> bierz koło),
  - rt11_b zostaje na kołach (jego GPS zaniża przy utracie fixa; MoTeC = koło).

Jeśli przyszła zmiana coś popsuje, ten test krzyknie.

Uruchom samodzielnie (bez pytest):   python test_motec_distance.py
albo przez pytest:                   pytest test_motec_distance.py
Brakujące pliki .ld są POMIJANE, więc test jest przenośny.
"""
from pathlib import Path
from motec_distance import parse_ld

# ścieżka względna  ->  Corr Dist [m] z MoTeC
GROUND_TRUTH = {
    # pierwotny błąd ze zrzutu: płaskie koła, realny ruch z GPS
    "test/00_22_27_02_2026.ld": 1462,
    # sesje z GPS, gdzie koło zgubiło dystans (dziura) -> GPS
    "rt15e/06_46_31_07_2025.ld": 4067,
    "rt15e/07_00_31_07_2025.ld": 2063,
    "rt15e/13_12_28_02_2026.ld": 502,    # średnia z dwóch odbiorników (było +16% przed fuzją)
    "rt15e/12_39_12_05_2026.ld": 205,    # zaszumiony odbiornik odrzucony -> czysty/koło
    "rt15e/17_54_28_02_2026.ld": 531,    # średnia z dwóch odbiorników
    "rt11-rt14e/rt13e/10_00_27_04_2024.ld": 242,
    "rt11-rt14e/rt14e/17_52_06_08_2024.ld": 298,
    "rt11-rt14e/rt12e/15_46_04_12_2022.ld": 291,
    "rt11-rt14e/rt14e/16_34_08_08_2024.ld": 141,
    "rt11-rt14e/rt14e/12_34_07_12_2024.ld": 138,
    "rt11-rt14e/rt13e/09_14_27_04_2024.ld": 128,
    "rt11-rt14e/rt14e/11_58_07_12_2024.ld": 104,
    # rt11_b (2021): GPS zaniża przy utracie fixa; MoTeC = koło. Potwierdzone ręcznie.
    "rt11-rt14e/rt11_b/ENDURANCE_MARCEL_3_12_18_29_06_2021.ld": 3663,
    "rt11-rt14e/rt11_b/skidpad_kubera_3_16_41_01_06_2021.ld": 3032,
    "rt11-rt14e/rt11_b/16_10_01_06_2021.ld": 937,
}

TOL_PCT = 0.05      # 5% — z zapasem nad najgorszym sprawdzonym odchyleniem (13_12, +2.2%)
TOL_ABS_M = 3.0     # dolny próg dla bardzo krótkich sesji


def _evaluate():
    """Sprawdza każdy plik z GROUND_TRUTH; zwraca wiersze (ścieżka, oczekiwane, wynik, status)."""
    base = Path(__file__).resolve().parent
    rows = []
    for rel, truth in GROUND_TRUTH.items():
        p = base / rel
        if not p.exists():                       # pliku nie ma -> pomijamy, nie psujemy testu
            rows.append((rel, truth, None, "SKIP (file not present)"))
            continue
        km, status = parse_ld(p)
        got = km * 1000 if km is not None else 0.0
        tol = max(truth * TOL_PCT, TOL_ABS_M)
        res = "PASS" if abs(got - truth) <= tol else "FAIL"
        rows.append((rel, truth, got, res if res == "PASS" else f"FAIL ({status})"))
    return rows


def test_ground_truth():
    """Punkt wejścia dla pytest: zbiera wszystkie błędy i rzuca asercją, jeśli są."""
    failures = []
    for rel, truth, got, res in _evaluate():
        if res.startswith("FAIL"):
            failures.append(f"  {rel}: expected ~{truth} m, got {got:.0f} m")
    assert not failures, "MoTeC Corr Dist regression(s):\n" + "\n".join(failures)


def main() -> int:
    """Uruchomienie samodzielne: drukuje tabelę i zwraca kod wyjścia (0 = wszystko OK)."""
    rows = _evaluate()
    print(f"\n{'file':46} {'MoTeC':>7} {'program':>8}  result")
    print("-" * 78)
    checked = passed = 0
    for rel, truth, got, res in rows:
        gs = "-" if got is None else f"{got:7.0f}"
        print(f"{rel:46} {truth:7} {gs:>8}  {res}")
        if res != "SKIP (file not present)":
            checked += 1
            passed += res == "PASS"
    print("-" * 78)
    print(f"{passed}/{checked} checked files within {TOL_PCT*100:.0f}% of MoTeC "
          f"({len(rows) - checked} skipped)\n")
    return 0 if passed == checked else 1


if __name__ == "__main__":
    raise SystemExit(main())
