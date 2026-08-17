# MoTeC Distance Parser

A Python program with the sole purpose of calculating the distance covered in
your MoTeC sessions, straight from the `.ld` telemetry logs.

MoTeC i2 shows a **Corr Dist** channel, but it does not exist inside the `.ld`
file itself — i2 computes it with maths channels after loading the log. This
program computes it independently: it integrates speed over time using the
trapezoid rule, picking the most trustworthy speed source available.

Agreement with MoTeC's own values: **median ~0%, worst case ~2%** deviation,
measured against a set of sessions whose Corr Dist was read off by hand.

> Written with help from Claude Code, so please be forgiving ;)

## How the speed source is chosen

| Priority | Source | Why |
|---|---|---|
| 1 | **GPS** | what MoTeC does when it has a fix |
| 2 | **Front wheels** | most trustworthy, because they are undriven |
| 3 | **Rear wheels** | **corroboration only** — driven wheels over-read under wheelspin, and on a dyno they spin while the car stands still |

On top of that sit the guards worked out on real sessions:

- **Dead wheel** — when GPS shows movement but a wheel has under 50% of its
  distance, the wheel is dropped.
- **GPS primary** — when a healthy GPS reads more than 10% above the wheel, the
  wheel lost distance, so the GPS value wins (this is what MoTeC does too).
- **Corrupt GPS** — if the GPS speed peak is 1.8x higher than the wheel peak,
  the GPS is noisy: the result falls back to the wheel and the session is
  flagged `CHECK`.
- **Two-receiver fusion** — two GPS receivers in a log identify themselves;
  agreeing ones (gap ≤ 30%) are averaged, disagreeing ones are settled by whose
  peak sits closer to the wheels.
- **Disagreeing front wheels** — a gap above 10% means a failed sensor (a sensor
  drops pulses, so it under-reads); the higher wheel is taken, provided GPS or
  the rear wheels confirm it.
- **Rear wheels only** — the session is listed but **excluded from the total**
  (most likely a dyno run).

Sessions without independent confirmation get a `CHECK` status and are listed
separately at the end for manual review.

## Installation

Python 3.9+. `numpy` is optional but strongly recommended — with it, a 2 GB log
is processed in milliseconds instead of ~1.3 s.

```bash
git clone https://github.com/diluteassets/MoTeC-Distance-Parser.git
cd MoTeC-Distance-Parser
pip install numpy        # optional, but recommended
```

## Configuring channel names

Every team names its channels differently, so **the names are not hardcoded** —
the program reads them from `channels.json`. Only the placeholder template
(`channels.example.json`) is committed.

1. Find out what the channels are called in your log:

   ```bash
   python motec_distance.py session.ld --list-channels
   ```

2. Copy the template and fill in your own names:

   ```bash
   cp channels.example.json channels.json
   ```

`channels.json` is gitignored, so your channel names never reach the repository.
Without that file the program falls back to the template — and since that holds
nothing but placeholders, it will only recognise standard channels such as
`GPS Speed`.

What you set in `channels.json`:

- `wheel_channels` — name fragments of the wheel-speed channels for the `FL`,
  `FR`, `RL` and `RR` positions;
- `gps.axis_x_suffixes` / `gps.axis_y_suffixes` — the name suffixes of the
  velocity-component channels (whatever precedes the suffix is the receiver
  name, which is how receivers are grouped);
- `gps.scalar_requires` / `gps.scalar_excludes` — how to match a single GPS
  speed channel when no components exist;
- `fallback_channels` — channels used when there are neither front wheels nor GPS.

## Usage

```bash
# a single session
python motec_distance.py session.ld

# a whole folder, recursively
python motec_distance.py logs/ -r

# a glob, results in metres, exported to CSV
python motec_distance.py "logs/*.ld" --unit m -o results.csv

# a large batch: 8 processes, no table
python motec_distance.py logs/ -r -j 8 -q
```

### Arguments

| Argument | Description |
|---|---|
| `PATH...` | file(s), folder(s) or globs, e.g. `logs/*.ld` |
| `-r`, `--recursive` | search folders recursively |
| `--unit {km,miles,m}` | output unit (default `km`) |
| `-o`, `--output FILE` | write results to a CSV file |
| `-q`, `--quiet` | no table, just a progress counter and the summary |
| `-c`, `--channels FILE` | use a different channel-name JSON file |
| `--list-channels` | print the channels found in the files and exit |
| `-j`, `--jobs N` | parallel worker processes (default 1; 4–8 for large batches) |

### Example output

```
File                       Distance  Status
----------------------------------------------------------
logs/session_01.ld         1.462 km  ok (FL)
logs/session_02.ld         4.067 km  ok (GPS primary): FL under-recorded, GPS used (4067m)
logs/session_03.ld              -    error: rear-only (no front/GPS) -> excluded, possible dyno

Files analyzed      : 3
Parsed OK           : 2
Failed              : 1
Auto-recovered      : 1  (dead/disagreeing wheel, recovered via GPS/rears)
Needs review        : 0  (no independent confirmation)
Total distance      : 5.529 km
```

### Statuses

| Prefix | Meaning |
|---|---|
| `ok (...)` | parsed normally, source in brackets |
| `ok (recovered): ...` | a source failed, but the result was confirmed by GPS or the rear wheels |
| `ok (GPS primary): ...` | the wheel lost distance, the GPS value was used |
| `CHECK: ...` | parsed, but without independent confirmation — verify by hand |
| `error: ...` | not parsed, the file is excluded from the total |

## Tests

The regression test keeps code changes from drifting away from the `Corr Dist`
values read by hand out of MoTeC i2. The reference data lives in
`reference_data.json` (template: `reference_data.example.json`), so your own
session names stay out of the repository too. Missing `.ld` files are skipped,
which means the test runs without committing any logs.

```bash
python test_motec_distance.py     # standalone, with a table
pytest test_motec_distance.py     # through pytest
```

## The .ld format, briefly

The file is memory-mapped (`mmap`), so only the channels actually needed are
read rather than the whole log. Channels form a linked list of 124-byte
descriptors, and a sample's physical value is:

```
value = raw * mul / scale * 10^(-dec) + shift
```

The descriptor field offsets are documented next to the `_C_*` constants in
`motec_distance.py`.

## License

MIT — see [LICENSE](LICENSE).
