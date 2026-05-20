# Jade Valley Flood Simulation — User Guide

A step-by-step guide for installing, running, and interpreting the Jade Valley flood-simulation system.

**Project:** Jade Valley Subdivision Flood Simulation
**Location:** Davao City, Philippines
**System builds:** `25%` · `50%` · `75%` · `100%` (master)

---

## 1. What this system does

The system models how floodwater spreads across the Jade Valley Subdivision (Davao City) under a range of storm scenarios — from a 15 mm convective shower to a 400 mm Severe Typhoon. It uses a terrain-aware physics engine (D8 flow routing, Green-Ampt-inspired infiltration, BFS river overflow) operating on a real 57 × 61 cell GeoTIFF DEM derived from a 3D topographic survey of the subdivision.

You can:

- Run any of the **6 PAGASA-calibrated storm scenarios** interactively via a Tkinter GUI.
- Toggle up to **4 prevention measures**: Riverbank Floodwall · Drainage Canal Network · Retention Basin · Elevated Emergency Road.
- See a live multi-layer animated map of flooding with playback controls (play / pause / scrub / speed / step).
- Export animations as GIFs and time-series data as CSVs.
- Run the **full 6 × 4 = 24-cell** evaluation matrix headlessly with one command, producing a master CSV / JSON dataset.
- Generate the canonical HAND risk-zone assessment (5-class zonation + return-period maps).

---

## 2. Repository layout

```
Final Project (Simulation)/
├── Documents/
│   ├── Simulation_Documentation.txt         ← formal documentation
│   ├── USER_GUIDE.md                        ← this file
│   ├── User_Guide.docx                      ← Word edition of this guide
│   ├── Jade_Valley_Flood_Simulation_FINAL.pptx
│   └── PAGASA_ARTC_2017.pdf                 ← rainfall reference
├── Main/
│   ├── flood_simulation_25%.py              ← Baseline, no prevention
│   ├── flood_simulation_50%.py              ← + Floodwall, Drainage Canal
│   ├── flood_simulation_75%.py              ← + Retention Basin, Elevated Road
│   ├── flood_simulation_100%.py             ← Master headless 24-cell runner
│   ├── hand_risk_analysis.py                ← HAND model + risk zonation
│   ├── batch_comparison.py                  ← Legacy batch runner (75% only)
│   ├── run_validation.py                    ← Compares sim to historical typhoons
│   └── requirements.txt
├── Map Topography/
│   ├── 2D/JVS_2D.jpg                        ← Satellite background
│   ├── 2D/...2D_vectorial.dxf
│   ├── 3D/JVS_Simulation.tif                ← REQUIRED — GeoTIFF DEM
│   └── 3D/...3D_modeling.dxf
└── Results/
    ├── animations/                          ← Exported GIFs land here
    ├── data/
    │   ├── master_results_table.csv         ← 24-row master dataset
    │   ├── master_results_table.json
    │   ├── master_results_summary.txt
    │   ├── master_reduction_table.txt
    │   ├── flood_statistics.json            ← HAND risk-zone stats
    │   ├── flood_risk_report.txt
    │   ├── hand_model.npy / risk_map.npy / dem_processed.npy / slope.npy / flow_accumulation.npy
    │   └── sim_*.csv                        ← Per-run time series
    └── maps/                                ← Static risk-zone PNG maps
```

---

## 3. Installation

### 3.1 Prerequisites

- **Python 3.10+** (3.13 verified). Check with `python --version`.
- **Windows / macOS / Linux.** GUI tested on Windows 11.
- ~200 MB free disk space for results and animations.

### 3.2 Install dependencies

From the project root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r Main/requirements.txt
# Plus deliverable tooling (PPT/Word generation):
pip install python-pptx python-docx
```

### 3.3 Verify required files

The system requires the GeoTIFF DEM. Confirm it exists:

```
Map Topography/3D/JVS_Simulation.tif        (mandatory)
Map Topography/2D/JVS_2D.jpg                (background image)
```

The two `.dxf` files in `Map Topography/` are optional fallbacks if the JPEG fails to load.

---

## 4. Running the interactive builds (25% / 50% / 75%)

Each build launches a Tkinter configuration GUI and then opens a Matplotlib animation window. Use whichever build matches the level of prevention infrastructure you want to model:

```bash
python "Main/flood_simulation_25%.py"    # Baseline (no prevention)
python "Main/flood_simulation_50%.py"    # + Floodwall + Drainage Canal
python "Main/flood_simulation_75%.py"    # + Retention Basin + Elevated Road
```

### 4.1 GUI walkthrough

1. **Scenario Preset** — pick one of the six PAGASA-calibrated storms (Light Rain through Typhoon Signal 3), or choose Custom (Scenario 7) and enter your own rainfall / duration.
2. **Storm Pattern** — uniform · progressive · burst · decreasing. Burst is a Gaussian peak mid-storm (typhoon-like); progressive ramps up gradually.
3. **Simulation Parameters (right column)** — adjust rainfall total, duration, timestep, wind, soil saturation, drainage capacity. Defaults are sensible for the selected preset.
4. **Prevention Measures tab (50% / 75% only)** — toggle each measure on/off and set its height / depth.
5. **RANDOMIZE** — populate all fields with a plausible random storm. Useful for stress-testing.
6. **▶ RUN SIMULATION** — closes the GUI, starts preprocessing (~1–3 s), then opens the animation window.

### 4.2 Animation playback

- **Play / Pause** — start/stop the animation.
- **Frame slider** — scrub to any timestep.
- **Speed slider** — 0.25× to 4× playback.
- **◀◀ / ▶▶** — step one frame backward / forward.
- **Save GIF** — export full animation to `Results/animations/`. Takes 30–60 s.
- **Save CSV** — export the live time-series stats (one row per frame) to `Results/animations/`.

### 4.3 Reading the live stats panel

| Field | Meaning |
|---|---|
| Total Rainfall | Configured storm total (mm) |
| Avg Rate / Peak Rate | mm/hr over the storm duration / peak frame |
| Return Period | Estimated PAGASA-IDF return period for this rate |
| Peak Discharge | Rational-method Q = 0.55 · i · A (m³/s) |
| Flooded Area | % of grid currently above 1 cm depth + ha equivalent |
| Max Depth | Peak water depth anywhere in the grid (mm) |
| River Overflow | % of grid currently receiving river overflow |
| Risk Assessment | PAGASA-aligned alert level (NORMAL → EVACUATE NOW) |

---

## 5. Running the 100% complete-system build

The 100% build is the **complete simulation system**: it has the same Tkinter GUI as the 75% build (storm tab + four prevention measures) plus an additional **COLLECT ALL DATA** button that runs the full 6 × 4 evaluation matrix and writes the master dataset.

### 5.1 GUI mode (default)

```bash
python "Main/flood_simulation_100%.py"
```

In the window:

- **RUN SIMULATION** — runs the selected scenario interactively (same as 75%): preprocessing → animation window → playback controls → save GIF/CSV. The run is also appended to `Results/data/master_runs_log.csv`.
- **COLLECT ALL DATA  (6×4 matrix)** — runs all 6 PAGASA scenarios × 4 prevention configurations = 24 simulations silently with a live progress bar (≈ under a minute total). Writes the four master files described below.
- **RANDOMIZE** — fills the form with a plausible random storm + prevention toggle set.

### 5.2 Headless matrix mode (CI / scripts)

```bash
python "Main/flood_simulation_100%.py" --matrix-only            # full 6×4
python "Main/flood_simulation_100%.py" --matrix-only --quick    # 3×2 smoke
```

### 5.3 CLI options

| Flag | Effect |
|---|---|
| `--matrix-only` | Skip the GUI; run the matrix and exit. |
| `--quick` | With `--matrix-only`: smoke matrix (3 × 2). |
| `--no-hand` | Skip the HAND risk-zone verification step. |
| `--force-hand` | Force regenerate the canonical HAND files. |
| `--dt N` | Simulation timestep in minutes (default 20). |

### 5.4 What gets written by COLLECT ALL DATA / --matrix-only

| File | Contents |
|---|---|
| `Results/data/master_results_table.csv` | One row per (scenario × config) with peak %, peak ha, max depth, river %, onset, etc. |
| `Results/data/master_results_table.json` | Same data, machine-readable, includes metadata. |
| `Results/data/master_results_summary.txt` | Human-readable per-scenario table. |
| `Results/data/master_reduction_table.txt` | Δ% reduction of each config vs. its scenario's baseline. |
| `Results/data/master_runs_log.csv` | One row per *interactive* RUN SIMULATION click (accumulates over time). |

---

## 6. Standalone HAND risk-zone analysis

The HAND (Height Above Nearest Drainage) analysis produces the canonical 5-class risk zonation referenced throughout the documentation:

```bash
python "Main/hand_risk_analysis.py"            # idempotent (no-op if files exist)
python "Main/hand_risk_analysis.py" --force    # regenerate
```

Outputs in `Results/data/`:

- `flood_risk_report.txt` — formal report (terrain stats, return-period scenarios, 5-class risk table, methodology)
- `flood_statistics.json` — same content as JSON
- `dem_processed.npy` · `slope.npy` · `flow_accumulation.npy` · `hand_model.npy` · `risk_map.npy`

---

## 7. Validating against a historical typhoon

Compare simulated peak metrics against a known event recorded in the model's reference database:

1. Run the 75% build with the typhoon's rainfall / duration.
2. Open `Results/data/quantitative_results_<scenario>.txt` after the run.
3. Paste the simulated peak-flooded-% and peak-depth-mm into `Main/run_validation.py` (the two clearly marked variables at the top).
4. Run `python "Main/run_validation.py"`.

The script computes the relative error vs. the recorded event (currently: Typhoon Vinta 2017, Typhoon Pablo 2012, Typhoon Odette 2021, Habagat).

---

## 8. Storm scenario reference

| ID | Name | Rainfall | Duration | Pattern |
|---|---|---|---|---|
| 1 | Light Rain | 15 mm | 2.0 h | uniform |
| 2 | Moderate Rain | 36 mm | 3.0 h | progressive |
| 3 | Heavy Rain | 90 mm | 4.0 h | progressive |
| 4 | Typhoon Signal 1 (Tropical Depression) | 150 mm | 8.0 h | progressive |
| 5 | Typhoon Signal 2 (Tropical Storm) | 250 mm | 12.0 h | burst |
| 6 | Typhoon Signal 3 (Severe Typhoon) | 400 mm | 18.0 h | burst |
| 7 | Custom | (user) | (user) | (user) |

---

## 9. Prevention measures reference

| Measure | Build | Default | What it does |
|---|---|---|---|
| Riverbank Floodwall | 50% / 75% / 100% | 1.5 m crest | Raises western bank cells; blocks river overflow until the wall is overtopped. |
| Drainage Canal Network | 50% / 75% / 100% | 2.0 m depth | Lowers an east-running and a south-running corridor; provides fast drainage out of the residential core. |
| Retention Basin | 75% / 100% | 6.0 m depth | Excavates a basin at the highest flow-accumulation sink; stores peak runoff. |
| Elevated Emergency Road | 75% / 100% | 1.5 m raise | Raises a mid-grid east-west corridor; cross-subdivision emergency access stays dry. |

In the 100% master matrix, four configurations are evaluated:

1. **Baseline** — no measures
2. **Wall+Canal** — Floodwall + Drainage Canal (matches 50% build, default heights)
3. **Full Prevention** — all four measures (matches 75% build, default sizes)
4. **Large Prevention** — all four measures at larger physical dimensions (stress-test)

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `[ERROR] DEM file not found` | Confirm `Map Topography/3D/JVS_Simulation.tif` exists. |
| `[ERROR] rasterio is required` | `pip install rasterio` |
| GUI does not open | On headless servers Tkinter is unavailable; use the 100% build instead. |
| Matplotlib window is blank or tiny | Resize the window manually; some Linux DPI configs render small. |
| `UnicodeEncodeError` in console | Reconfiguration to UTF-8 is automatic; if it fails, run `set PYTHONIOENCODING=utf-8` (Windows). |
| GIF export fails with "Pillow not installed" | `pip install Pillow` |
| HAND analysis says "files already present" | This is by design. Pass `--force` to regenerate. |
| `run_validation.py` reports `[!] You haven't filled in...` | Edit the two `SIMULATED_PEAK_*` variables at the top and re-run. |

---

## 11. Output catalogue (where does what land?)

| Output | Path | Produced by |
|---|---|---|
| GIF animation | `Results/animations/flood_*.gif` | 25 / 50 / 75 builds via Save GIF |
| Per-run time series CSV | `Results/animations/flood_*_stats.csv` | 25 / 50 / 75 builds via Save CSV |
| Baseline / prevention CSVs | `Results/data/sim_*_BASELINE.csv` / `sim_*_WITH_PREVENTION.csv` | 75% build (single-run quantitative export) |
| HAND model artefacts | `Results/data/*.npy`, `flood_risk_report.txt`, `flood_statistics.json` | `hand_risk_analysis.py` |
| Master 24-cell dataset | `Results/data/master_results_*.csv/.json/.txt` | `flood_simulation_100%.py` |
| Risk-zone maps | `Results/maps/*.png` | Pre-generated (visual reference) |
| Documentation | `Documents/Simulation_Documentation.txt` | Authoritative formal documentation |
| Presentation | `Documents/Jade_Valley_Flood_Simulation_FINAL.pptx` | Generated by `Documents/_build_ppt.py` |

---

## 12. Quick reference

| Task | Command |
|---|---|
| Show me a single storm with all prevention | `python "Main/flood_simulation_75%.py"` |
| Generate the master dataset | `python "Main/flood_simulation_100%.py"` |
| Regenerate the HAND analysis | `python "Main/hand_risk_analysis.py" --force` |
| Regenerate the PPT | `python "Documents/_build_ppt.py"` |
| Validate against a historical typhoon | edit + run `python "Main/run_validation.py"` |

---

*For the formal specification (sections, variables, assumptions, equations) see `Documents/Simulation_Documentation.txt`. For the slide deck summarising everything, see `Documents/Jade_Valley_Flood_Simulation_FINAL.pptx`.*
