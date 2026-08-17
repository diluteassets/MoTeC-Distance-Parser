#!/usr/bin/env python3
"""Regression test: keeps motec_distance.parse_ld in line with MoTeC's "Corr Dist".

The reference data (session path -> "Corr Dist [m]" read by hand from MoTeC i2)
lives in a JSON file next to this test rather than in the code — see
reference_data.example.json. That way your own session names never have to reach
the repository.

These values freeze all of the accuracy work:
  - GPS as the primary source (MoTeC follows GPS when it has a fix),
  - fusion of two GPS receivers (average the agreeing ones, drop the noisy one),
  - the corrupt-GPS guard (a velY spike -> fall back to the wheel),
  - sessions where GPS under-reads on fix loss stay on the wheels.
If a future change breaks something, this test shouts.

Run standalone (no pytest):   python test_motec_distance.py
Or through pytest:            pytest test_motec_distance.py

Missing .ld files are SKIPPED, so the test is portable.
"""
import json
from pathlib import Path

from motec_distance import parse_ld, load_config

DATA_FILE = "reference_data.json"
EXAMPLE_FILE = "reference_data.example.json"

SKIPPED = "SKIP (file not present)"


def _load_reference() -> tuple[Path, dict, float, float]:
    """(base folder, {path: metres}, tolerance fraction, tolerance metres)."""
    base = Path(__file__).resolve().parent
    for name in (DATA_FILE, EXAMPLE_FILE):
        p = base / name
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return (base,
                    data.get("sessions", {}),
                    data.get("tolerance_pct", 5.0) / 100.0,
                    data.get("tolerance_m", 3.0))
    return base, {}, 0.05, 3.0


def _evaluate():
    """Check every session; return rows of (path, expected, got, result)."""
    base, sessions, tol_pct, tol_abs = _load_reference()
    cfg = load_config()
    rows = []
    for rel, truth in sessions.items():
        p = base / rel
        if not p.exists():                    # no file -> skip, don't fail the test
            rows.append((rel, truth, None, SKIPPED))
            continue
        km, status = parse_ld(p, cfg)
        got = km * 1000 if km is not None else 0.0
        tol = max(truth * tol_pct, tol_abs)
        ok = abs(got - truth) <= tol
        rows.append((rel, truth, got, "PASS" if ok else f"FAIL ({status})"))
    return rows, tol_pct


def test_ground_truth():
    """pytest entry point: collect every mismatch and assert once."""
    rows, _ = _evaluate()
    failures = [
        f"  {rel}: expected ~{truth} m, got {got:.0f} m"
        for rel, truth, got, res in rows if res.startswith("FAIL")
    ]
    assert not failures, "MoTeC Corr Dist regression(s):\n" + "\n".join(failures)


def main() -> int:
    """Standalone run: print a table and return an exit code (0 = all good)."""
    rows, tol_pct = _evaluate()
    if not rows:
        print(f"\nNo reference data — create {DATA_FILE} modelled on "
              f"{EXAMPLE_FILE}.\n")
        return 0

    print(f"\n{'file':46} {'MoTeC':>7} {'program':>8}  result")
    print("-" * 78)
    checked = passed = 0
    for rel, truth, got, res in rows:
        gs = "-" if got is None else f"{got:7.0f}"
        print(f"{rel:46} {truth:7} {gs:>8}  {res}")
        if res != SKIPPED:
            checked += 1
            passed += res == "PASS"
    print("-" * 78)
    print(f"{passed}/{checked} checked files within {tol_pct*100:.0f}% of MoTeC "
          f"({len(rows) - checked} skipped)\n")
    return 0 if passed == checked else 1


if __name__ == "__main__":
    raise SystemExit(main())
