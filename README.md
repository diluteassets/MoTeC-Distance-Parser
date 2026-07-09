# MoTeC-Distance-Parser
A python program with the sole purpose of calculating the distance covered in your MoTeC sessions.

## Usage

```
python motec_distance.py <path> [-r] [--unit km|miles|m] [-o results.csv] [-j 8]
```

`<path>` can be a single `.ld` file, a folder, or a glob like `logs/*.ld`.
Installing `numpy` is optional but strongly recommended (much faster parsing).

## Optimized variant

`motec_distance_optimized.py` is a drop-in copy of `motec_distance.py` with the
same CLI, the same source-selection logic and statuses, and numerically
identical results (agreement well below 1e-9 relative; the integer summation it
uses is actually *more* accurate than float accumulation). Measured speedups:

| profile | with numpy | without numpy |
|---|---|---|
| healthy session (wheels + GPS) | **~11-17x** | **~4.4x** |
| worst case (disagreeing GPS units) | **~3.8x** | **~2.9x** |

Key ideas:

- **integer-domain integration**: with a constant sample rate the trapezoid sum
  reduces to `dt * (sum - (first+last)/2)`, and the physical-value sum is just
  `factor * sum(raw) + n * shift` — so when the raw min/max show every sample is
  in range (the typical case), distance comes from a single lossless int64
  reduction over the raw samples: no float arrays, no copies,
- **zero-copy + decode-once**: raw samples are numpy views straight into the
  mmap, each channel is touched once per session (the original re-decoded wheel
  and GPS channels up to 3x and ran the full GPS fusion twice),
- floats are materialized lazily, only for out-of-range repair, speed peaks and
  GPS velocity components; the GPS pipeline works in-place and skips the peak
  percentile when only one receiver is present,
- signal detection from raw min/max without materializing values, one
  precompiled struct per channel descriptor, `madvise(WILLNEED)` readahead,
  and an `array`-based fast path for the no-numpy fallback,
- fixes a fallback-mode crash from the original: `TypeError` when a wheel-speed
  channel is in m/s and two GPS receivers disagree (list multiplied by float).

Verify and measure yourself:

```
python test_optimized_parity.py    # synthetic .ld parity tests (also via pytest)
python benchmark_distance.py       # original vs optimized timing
```
