# Tier C protocol: first self-measured driver validation

Tier B (`../tierb/`) validates against public measurements of a
design the simulator did not control: driver spread, exact driver
parameters, and box construction all enter as unknowns. Tier C
removes those unknowns by measuring **the same physical unit that is
modelled**. No manufacturer data is required — the driver's
parameters are extracted from the unit itself, and what is compared
is the coupled prediction chain (extracted parameters → enclosure →
radiation → CEA-2034-style curves) against that unit's measured
response.

Nothing here is fitted after the fact. The extracted parameters and
the model configuration are frozen and hashed before the acoustic
comparison is run (step 6); the report shows whatever agreement
results.

## 1. Test article

- One woofer, bought as a single unit (the SB13PFCR25-4 from the
  Tier B study is the natural first candidate: its public datasheet
  then serves as a sanity cross-check, never as an input).
- One sealed test box. Sealed first: no port geometry, tuning, or
  loss unknowns. Internal volume chosen near the driver's typical
  alignment (10–15 L for a 5″ woofer), panels ≥ 18 mm, braced,
  gasketed driver mounting, no stuffing for the first run (stuffing
  is an added unknown; it can be a second configuration).
- Record: driver serial/batch if printed, purchase source and date,
  break-in state (recommend 30 min of 25 Hz sine at moderate level,
  then rest ≥ 1 h before measuring).

## 2. Geometry record (`geometry.json`)

Measured with calipers, not taken from drawings:

- External and internal box dimensions, panel thicknesses.
- Driver centre position on the baffle (from two edges), cutout
  diameter, flange recess if any, mounting screw pattern.
- Driver dimensions that enter the radiation model: effective
  piston radius (from measured Sd if available, else surround
  midline diameter / 2, stated which), cone depth.
- Photograph of the box and the mounted driver.

## 3. Electrical measurements

Equipment: impedance measurement jig (REW + sound-card loopback
with a known sense resistor, a Dayton DATS, or an LCR meter with
sweep). Calibrate against a known resistor (e.g. 10 Ω, 1%) in the
same session and record the calibration result.

1. Free-air impedance sweep of the bare driver, 10 Hz–5 kHz,
   small signal (≈ 0.1–0.5 V at the terminals; record the actual
   drive level). Export `freq, |Z|, phase` text.
2. Added-mass or test-box repeat for Vas. Added mass: known mass
   (coins/Blu-Tack weighed to 0.01 g) centred on the dust cap,
   mass chosen to drop Fs by 20–30%.
3. Extract T/S parameters (Fs, Re, Qms, Qes, Qts, Vas, Mms, Cms,
   Rms, BL, Le or a fitted Le model) with the extraction tool of
   record, and **report the fit residuals** (max |Z| error of the
   fitted model against the measured curve). An extraction whose
   model cannot reproduce the measured impedance within ~5% over
   10–500 Hz is not good enough to freeze; fix the extraction
   before continuing.
4. In-box impedance sweep after mounting (same drive level). This
   validates the enclosure model independently of any acoustics:
   the predicted sealed-box impedance peak frequency and height
   must match before the SPL comparison means anything.

Store the raw sweeps as text (`impedance_freeair.txt`,
`impedance_inbox.txt`) — never only the extracted parameters.

## 4. Acoustic measurements

Equipment: calibrated measurement microphone (e.g. miniDSP UMIK-2
or an ECM8000/Dayton EMM-6 with its individual calibration file),
quiet room, rigid mic stand, tape/laser distance measure. Record
mic model, calibration file name and SHA-256, preamp/interface,
sample rate (≥ 48 kHz), and sweep settings.

Drive conditions for every acoustic run:

- Terminal voltage measured with a true-RMS meter at the driver
  terminals during the sweep (target 2.83 V; record the actual
  value and use it in the model).
- Amplifier identified; output impedance noted (or measured).
- Ambient temperature, pressure, humidity logged (Fs and c drift
  with temperature; the model uses the logged values).

Runs:

1. **Near-field** (mic 5–10 mm from the dust cap, low level):
   validates the low-frequency volume velocity against the
   electrical model up to ~300 Hz.
2. **Far-field on-axis** at 1 m from the baffle, driver height,
   gated sweep (window ≥ 5 ms if the room allows; record the
   window) plus a ground-plane or spliced near-field segment for
   20–300 Hz. Record the unsmoothed complex response if the tool
   allows (magnitude + phase text export), else magnitude with
   minimum-phase caveat noted.
3. **Off-axis horizontal** at 1 m: 0–90° in 15° steps (10° if
   time allows), rotating the box on a marked turntable around
   the driver axis. This yields a reduced spinorama (horizontal
   plane) for the same aggregation code path as Tier B.
4. **Repeat run**: re-measure run 2 without changing anything.
   The repeat delta is the measurement-noise floor; report it.

## 5. Session layout (half a day)

1. Calibrations (impedance jig, mic) — 30 min.
2. Free-air impedance + added mass + extraction — 1 h.
3. Box geometry record + mounting — 30 min.
4. In-box impedance — 15 min.
5. Acoustic runs 1–4 — 1.5 h.
6. Data freeze (below) — 15 min.

## 6. Freeze and comparison

Before any acoustic comparison is computed:

1. Write `session.md`: equipment, calibrations, conditions,
   file inventory, and the SHA-256 of every raw data file.
2. Freeze the model input: extracted T/S set, `geometry.json`,
   solver configuration (including the Tier B Krylov settings:
   GCRO-DR restart 200 / recycle 50). Hash the model input file.
   These hashes go in the report; changing any frozen input after
   seeing results requires a new, separately reported run.
3. Run the prediction (lumped chain for the electrical/near-field
   checks; BEM + CEA aggregation for the far-field, same example
   pattern as `mechano23_tierb`).
4. Report, per curve: the broadband sensitivity offset separately,
   then shape MAE / RMSE / max deviation over 100 Hz–4 kHz (the
   band the current solver settings reach; extend only with new
   solver evidence, as in Tier B). No parameter adjustment, no
   rescaling, no smoothing beyond what the measurement export
   itself applied.

Pre-registered expectations (stated now, not targets to tune to):
in-box impedance peak within ±3% in frequency; near-field SPL
shape MAE ≤ 1 dB below 300 Hz; far-field on-axis shape MAE ≤ 2 dB
over 100 Hz–4 kHz. Missing an expectation is a finding about the
model chain, reported as-is.

## 7. Uncertainty budget (report each)

- Mic calibration: manufacturer tolerance (typically ±1 dB class;
  state it).
- Mic distance: ±5 mm at 1 m ≈ ±0.04 dB.
- Drive voltage: meter accuracy, propagated to dB.
- T/S extraction: fit residuals from step 3.3.
- Repeatability: the run-4 delta is the empirical floor; shape
  MAEs below it are not significant.

## 8. What Tier C does and does not prove

Proves: the end-to-end chain (extraction → lumped driver →
enclosure → BEM radiation → aggregation) predicts one real unit
within a stated uncertainty, with every input measured from that
unit. Does not prove: production-spread behaviour, large-signal
validity, or performance above the solver-reachable band. Those
need further units and tiers, in that order.

Data layout when the first unit exists:

```text
fixtures/tierc/<unit-id>/
  session.md            equipment, conditions, hashes
  geometry.json         measured geometry (section 2)
  impedance_freeair.txt raw sweep
  impedance_inbox.txt   raw sweep
  nearfield.txt         complex or magnitude response
  farfield_h/           angle-labelled responses
  model_input.json      frozen T/S + solver config (hashed)
```

Own measurements carry no third-party licence; keep the raw files
so anyone can re-extract and re-compare.
