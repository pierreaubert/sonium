# Tier B loudspeaker references

These fixtures support independent, measured-reference comparisons. They are
not calibration inputs: no curve here is fitted, rescaled in frequency, or used
to tune the simulator. The comparison code reports a single broadband level
offset separately before computing shape errors.

## Primary candidate: XMechanik Mechano23 DIY

Mechano23 was selected because the design and an independent measurement are
both public:

- Design: XMechanik, “Small 2-way speakers with linear on-axis and power
  response characteristics,” ASR DIY thread 41757.
- Independent review/measurement: ASR, “Mechano23 Open-source DIY Speaker
  Review,” thread 54066 (Klippel NFS-derived spinorama data).
- Spinorama dataset entry: `XMechanik Mechano23 DIY`, origin `asr`.
- Drivers: SB Acoustics SB13PFCR25-4 woofer and Scan-Speak H2606/920000
  horn-loaded tweeter. Manufacturer data are public for both.
- Cabinet: 290 × 174 × 263 mm (H × W × D), 18 mm panels, approximately
  8 litres gross internal volume, rear port 35 mm diameter × 145 mm long,
  design tuning approximately 45 Hz.
- Published baffle drawing: tweeter centre 66 mm below the top edge, woofer
  centre 135 mm below the tweeter centre and 89 mm above the bottom edge.
  The cutouts are 77 mm (tweeter) and 127 mm (woofer); flanges are 104 mm and
  150.5 mm.
- The public VituixCAD archive contains the exact crossover project, measured
  in-cabinet driver impedances, and measured per-driver polar responses. It is
  not redistributed here. The ASR design thread links Google Drive file
  `1AYVkshP1xEtOVPwgIuT2JHddrZsg3Xww`; the archive used for this work had
  SHA-256 `89c73403d67072512e99e2135cdacfb54244339ae70c9ee7164bc244424bbe59`.

### Files in `mechano23/`

| File | SHA-256 |
| --- | --- |
| `SPL Horizontal.txt` | `43f897849b9db73f10cfd5257f05b7be58abe501705b4679fdbf8f1cc5010017` |
| `SPL Vertical.txt` | `88f5f3a29fe08cf7ab225fde2ed0fe3e24aa6906d62fea4a991a415cbc8204cd` |
| `LICENSE.txt` | ASR data licence supplied with the dataset entry |

The ASR data are © Audio Science Review LLC and are supplied under
CC BY-NC-SA 4.0; see `mechano23/LICENSE.txt`. The files are retained with that
licence for non-commercial validation use and attribution.

## Secondary candidate: Parts Express C-Note

The top-level `CEA2034.txt` is the ASR measurement for the Parts Express DIY
C-Note, with the same ASR licence in `LICENSE.txt`. Its cabinet and Dayton
Audio driver data are also public, but its baffle positions in the current
reconstruction are less completely documented than Mechano23’s published
drawing, so it remains a secondary candidate.

## Modelling boundaries for Mechano23

The first-principles comparison models flat pistons, an approximate short
horn for the tweeter, the exact published crossover topology and values, the
woofer’s rear vented-box load, and rigid-piston radiation. It does not model
cone/dome breakup, surround or cabinet-panel vibration, port direct radiation,
viscothermal port losses, driver manufacturing spread, or the NFS scanner’s
near-field reconstruction. Those omissions are expected to matter mainly in
the woofer breakup region, at high frequencies, and below the primary
100 Hz comparison band.

## Measured comparison (October 2026)

Reproduce with the example in `crates/sonium-sim-speaker/examples/`:

```sh
MECHANO23_FREQ_POINTS=17 MECHANO23_FREQ_END_HZ=4000 MECHANO23_MAX_MESH_HZ=4000 \
  cargo run --release -p sonium-sim-speaker --example mechano23_tierb -- out-dir
```

That grid (6,494 BEM DOFs, all 17 frequencies converged, true residuals
≤ 1e-7) completed in 9 min 17 s on a 128-core host. Metrics below are
computed over the 14 grid points in 100 Hz–4 kHz, with one broadband
sensitivity offset reported separately per curve before shape errors
(no fitting or rescaling of any measured curve):

| Curve | Sonium vs ASR shape MAE | Vituix-ref vs ASR shape MAE |
| --- | --- | --- |
| On Axis | 2.28 dB (offset −2.37 dB) | 0.98 dB (offset −21.34 dB) |
| Listening Window | 2.31 dB (offset −2.42 dB) | 0.95 dB (offset −21.39 dB) |
| Early Reflections | 1.84 dB (offset −2.23 dB) | 0.86 dB (offset −21.64 dB) |
| Sound Power | 2.30 dB (offset −2.61 dB) | 1.06 dB (offset −22.21 dB) |
| Estimated In-Room | 1.87 dB (offset −2.32 dB) | 0.89 dB (offset −21.76 dB) |

Sonium-vs-Vituix shape MAE on the same grid is 1.54–1.79 dB. The large
Vituix offsets are expected: the project FRDs are relative-level driver
measurements, while Sonium and ASR are absolute SPL at 2.83 V. The
Vituix column validates the aggregation path (same CEA-2034 groups and
weights applied to all three curve sets). In absolute terms Sonium
agrees with ASR within ±0.3 dB from 150–600 Hz (−1.1 dB at 114 Hz);
relative to that low-frequency agreement the prediction then falls
progressively below the measurement above ~700 Hz, reaching −6.6 dB
(offset-removed) at 3 kHz on axis — consistent with the modelling
boundaries above (flat-piston tweeter with approximate horn, lumped
tweeter load, no breakup). An independent Python recomputation from
the emitted CSV reproduces every metric above to the printed
precision.

Solver-cost boundary, measured on the same host: the default GCRO-DR
restart 40 / recycle 20 does not converge above ~1 kHz on these meshes
(12,000-iteration cap, residuals 1e-3..1e-1); the example therefore
uses restart 200 / recycle 50, which converged at every probed
frequency. Per-frequency H-matrix cost then rises steeply with DOF
count: 14,052 DOFs (6 kHz mesh) produced no completed frequency within
31 minutes, and at 25,014 DOFs (8 kHz mesh) only four of 21
frequencies (229 Hz–2.2 kHz) completed in the first hour. A dedicated
single-frequency 8 kHz run, one lane on the otherwise idle 128-core
host (~53 cores busy), also produced no completed solve within a
25-minute cap. The dense CUDA path (RTX PRO 6000, 98 GB) was also
probed: at 14,052 DOFs it assembles and iterates in ~90 s, but its
mixed-precision GMRES plateaus at ~8e-4 true residual. Under the
default Qualification policy the independent CPU reference reassembly
exceeded 100 GB with unrestricted threads; with
`RAYON_NUM_THREADS=2` that reference completed, but the crate's f64
acceptance gate rejected the GPU field at 6 kHz (physical observable
error 5.995e-2 against a 1e-3 tolerance). At 25,014 DOFs the resident
CUDA workspace exceeds the device quota (~127 GB wanted).
Extending the Sonium curves
to 8 kHz is therefore blocked on H-matrix assembly/iteration
performance, not on the comparison setup; the dense FGMRES+ILU path is
not a fallback at these sizes (out-of-memory at 236 GB with nine
6,494-DOF lanes). Treat any future 8 kHz numbers as new evidence, not
as an extrapolation of the table above.

## Tier C

The self-measured tier — same physical unit measured and modelled,
no manufacturer data anywhere in the chain — is specified in
`../tierc/PROTOCOL.md`. It is the next validation step once a first
driver unit is available.
