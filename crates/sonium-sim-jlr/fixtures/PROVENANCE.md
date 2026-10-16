# Fixture provenance

Source paper: Melchior, Weibel, Reviol & von Harbou, "Validation of
single-phase fluid dynamics in a Lattice Boltzmann simulated JLR",
International Journal of Heat and Fluid Flow 121 (2026) 110519.
DOI: https://doi.org/10.1016/j.ijheatfluidflow.2026.110519
PII: S0142727X26002857. Journal version CC BY 4.0 (Crossref licence record).

Open manuscript used for extraction (authors' revised submitted version,
identical in all tables and figures used here): Zenodo record 21246454,
file `JLR_LBM_RPTU_LRF.pdf`, https://zenodo.org/records/21246454
(retrieved 2026-10-16; local scratch copy in `tmp/jlr/`, not committed).

| File | Source | Extraction |
|---|---|---|
| `measured_annular_gap_table_a4.csv` | Paper Table A.4 (24 runs) | Parsed from the PDF text layer by regex; flow column kept in L/h as printed (header unit in the PDF is ambiguous; 4156 L/h = 1.154e-3 m3/s) |
| `circulation_number_table_4_2.csv` | Paper Table 4.2 | Typed from PDF text layer |
| `reynolds_table_4_1.csv` | Paper Table 4.1 | Typed from PDF text layer |
| `geometry_table_a1.csv` | Paper Table A.1 + Sections 3.1.1, 3.2.2, Fig. A.2 | Typed; rows marked `derived_*` are our interpretations, configurable in the case config, not paper values |
| `sim_profiles_fig4_3_digitised.csv` | Paper Fig. 4.3, blue Sim curves | `digitize_fig43.py` on a 400 dpi render of PDF page 20; axis boxes detected, x calibrated on the 60/70/80 mm ticks, y on 0-0.7 m/s; error about +-0.005 m/s, +-0.15 mm. Panel A legend region excluded |

Known paper inconsistency: Fig. 4.3 caption prints phi = 4.083 kW/m3 where
the text, Table 4.2 and Table A.4 give 4.038 kW/m3 (panel B header in the
figure also reads 4.083). Fixtures use 4.038.

Companion sources for dimensions the paper does not print (support struts,
outlet): Weibel et al., "Modeling and Experimental Validation of Internal
Circulation in Single Phase Jet Loop Reactors", Zenodo record 21134142,
and the experiments paper cited as ref. [9] (DOI 10.5281/zenodo.17775734).
Any dimension taken from a drawing rather than a table is marked `derived`.
