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
same CLI and identical results (bit-for-bit on the numpy path), but roughly
2x faster with numpy and ~2.5x faster without it:

- each channel is decoded from the file only **once** per session (the original
  re-read wheel and GPS channels up to 3x across its consistency checks, and ran
  the full GPS fusion twice),
- one-allocation float64 decoding, a fast integration path when no samples need
  patching, and `array`-based decoding on the no-numpy fallback,
- fixes a fallback-mode crash from the original: `TypeError` when a wheel-speed
  channel is in m/s and two GPS receivers disagree (list multiplied by float).

Verify and measure yourself:

```
python test_optimized_parity.py    # synthetic .ld parity tests (also via pytest)
python benchmark_distance.py       # original vs optimized timing
```
