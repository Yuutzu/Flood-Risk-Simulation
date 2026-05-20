"""
=============================================================================
 JADE VALLEY SUBDIVISION - FLOOD SIMULATION  (100% BUILD)  -  COMPLETE SYSTEM
 Davao City, Philippines
=============================================================================
 The 100% build is the COMPLETE simulation system. It is fully interactive
 (GUI + animation) like the 25/50/75 builds AND it additionally collects
 data across ALL scenarios into a master dataset.

 What you get in this build:

   * Same Tkinter GUI as 75% - storm scenario tab + 4 prevention measures
     (Floodwall, Drainage Canal, Retention Basin, Elevated Emergency Road).

   * The familiar "RUN SIMULATION" button - runs the selected scenario
     interactively (animation, playback controls, GIF/CSV export).

   * NEW button: "COLLECT ALL DATA" - runs the FULL evaluation matrix
     (6 PAGASA scenarios x 4 prevention configurations = 24 simulations)
     headlessly in the background, with a live progress bar, then writes
     the master dataset to Results/data/:

         master_results_table.csv       per-row results
         master_results_table.json      machine-readable copy
         master_results_summary.txt     human-readable summary
         master_reduction_table.txt     % reduction vs baseline

   * NEW: HAND/risk-zone analysis is auto-generated on first use
     (delegates to Main/hand_risk_analysis.py, idempotent).

   * Behind the scenes, every interactive run is ALSO appended to the
     master_runs_log.csv so progressive use builds up the dataset over time.

 The physics engine is the SAME 75% engine (imported as a module). The
 25%/50%/75% scripts remain the individual interactive builds for those
 prevention configurations. This 100% script is the union.

 Run:
   python "Main/flood_simulation_100%.py"
=============================================================================
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import tkinter as tk
import warnings
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import numpy as np

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR  = Path(__file__).resolve().parent.parent
MAIN_DIR  = BASE_DIR / "Main"
SIM_75    = MAIN_DIR / "flood_simulation_75%.py"
HAND_MOD  = MAIN_DIR / "hand_risk_analysis.py"
DATA_OUT  = BASE_DIR / "Results" / "data"
ANIM_DIR  = BASE_DIR / "Results" / "animations"
DATA_OUT.mkdir(parents=True, exist_ok=True)
ANIM_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DELEGATE LOADER (handles % in module filenames)
# =============================================================================

def _load_module(path: Path, name: str):
    if not path.exists():
        raise SystemExit(f"\n[ERROR] Required module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"\n[ERROR] Could not build import spec for {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load 75% engine + HAND module up front
sim75    = _load_module(SIM_75,   "sim75")
hand_mod = _load_module(HAND_MOD, "hand_mod")

# Re-export key symbols
load_dem                  = sim75.load_dem
apply_prevention_measures = sim75.apply_prevention_measures
FloodSimulation           = sim75.FloodSimulation
run_simulation_75         = sim75.run_simulation
wind_rainfall_map         = sim75.wind_rainfall_map
_intensity_factor         = sim75._intensity_factor
SCENARIOS_75              = sim75.SCENARIOS


# =============================================================================
# MASTER MATRIX DEFINITIONS
# =============================================================================

# All 6 PAGASA-calibrated storm scenarios.
MATRIX_SCENARIOS = [
    (1, "Light Rain",                            15,   2.0,  "uniform"),
    (2, "Moderate Rain",                         36,   3.0,  "progressive"),
    (3, "Heavy Rain",                            90,   4.0,  "progressive"),
    (4, "Typhoon Signal 1 (Tropical Depression)", 150,  8.0,  "progressive"),
    (5, "Typhoon Signal 2 (Tropical Storm)",     250,  12.0, "burst"),
    (6, "Typhoon Signal 3 (Severe Typhoon)",     400,  18.0, "burst"),
]

# Four prevention configurations (baseline + 3 mitigation levels).
MATRIX_CONFIGS = [
    ("Baseline",         False, False, False, False, 0.0, 0.0, 0.0, 0.0),
    ("Wall+Canal",       True,  True,  False, False, 1.5, 2.0, 0.0, 0.0),
    ("Full Prevention",  True,  True,  True,  True,  1.5, 2.0, 6.0, 1.5),
    ("Large Prevention", True,  True,  True,  True,  2.5, 3.0, 9.0, 2.5),
]

MATRIX_DT_MIN = 20    # 20-min timestep keeps the full matrix under a minute


# =============================================================================
# RUN-ONE HELPER (used by both interactive log and the full matrix)
# =============================================================================

def _run_one_headless(dem: np.ndarray, cellsize: float,
                      scen: tuple, cfg: tuple, dt_min: int) -> dict:
    """Run one (scenario, config) pair without GUI/animation."""
    scen_id, scen_name, rainfall_mm, duration_h, pattern = scen
    (cfg_label, use_wall, use_canal, use_basin, use_road,
     wall_h, canal_d, basin_d, road_h) = cfg

    any_prev = use_wall or use_canal or use_basin or use_road
    if any_prev:
        sim_dem, wall_mask, canal_mask, basin_mask, road_mask = \
            apply_prevention_measures(
                dem, cellsize,
                use_wall, use_canal, use_basin, use_road,
                wall_h, canal_d, basin_d, road_h)
    else:
        sim_dem    = dem.copy()
        wall_mask  = np.zeros(dem.shape, dtype=bool)
        canal_mask = np.zeros(dem.shape, dtype=bool)
        basin_mask = np.zeros(dem.shape, dtype=bool)
        road_mask  = np.zeros(dem.shape, dtype=bool)

    rate_mmhr  = rainfall_mm / duration_h
    dt_h       = dt_min / 60.0
    num_frames = int(np.ceil(duration_h / dt_h))
    total_area_m2 = float(dem.size * cellsize * cellsize)

    wmap = wind_rainfall_map(dem, 0.0, 0.0)
    sim = FloodSimulation(
        sim_dem, cellsize,
        soil_saturation_pct    = 30.0,
        drainage_capacity_mmhr = 5.0,
        canal_mask  = canal_mask,
        wall_mask   = wall_mask,
        basin_mask  = basin_mask,
        road_mask   = road_mask,
        rainfall_mm = rainfall_mm,
        original_dem = dem,
    )

    peak_flooded_pct  = 0.0
    peak_depth_mm     = 0.0
    peak_river_pct    = 0.0
    overflow_min      = None
    rain_at_overflow  = None
    final_flooded_pct = 0.0
    final_depth_mm    = 0.0
    t0 = time.time()
    for fr in range(num_frames):
        inten = _intensity_factor(fr, num_frames, pattern)
        sim.step(rate_mmhr, dt_h, intensity=inten, wind_map=wmap)
        total       = sim.rain_water + sim.river_water
        flooded_pct = float(np.sum(total > 0.01) / total.size * 100)
        river_pct   = float(np.sum(sim.river_water > 0.005) / total.size * 100)
        max_depth   = float(total.max() * 1000)
        if flooded_pct > peak_flooded_pct: peak_flooded_pct = flooded_pct
        if max_depth   > peak_depth_mm:    peak_depth_mm    = max_depth
        if river_pct   > peak_river_pct:   peak_river_pct   = river_pct
        if overflow_min is None and river_pct > 0.5:
            overflow_min     = (fr + 1) * dt_min
            rain_at_overflow = float(sim.rainfall_accumulated * 1000)
        final_flooded_pct = flooded_pct
        final_depth_mm    = max_depth
    return {
        "scenario_id":        scen_id,
        "scenario":           scen_name,
        "config":             cfg_label,
        "rainfall_mm":        rainfall_mm,
        "duration_h":         duration_h,
        "pattern":            pattern,
        "use_wall":           use_wall,
        "use_canal":          use_canal,
        "use_basin":          use_basin,
        "use_road":           use_road,
        "peak_flooded_pct":   round(peak_flooded_pct, 2),
        "peak_flooded_ha":    round(peak_flooded_pct / 100.0 * total_area_m2 / 10_000.0, 2),
        "peak_depth_mm":      round(peak_depth_mm, 1),
        "peak_river_pct":     round(peak_river_pct, 2),
        "overflow_min":       overflow_min,
        "rain_at_overflow":   round(rain_at_overflow, 1) if rain_at_overflow else None,
        "final_flooded_pct":  round(final_flooded_pct, 2),
        "final_depth_mm":     round(final_depth_mm, 1),
        "wall_cells":         int(wall_mask.sum()),
        "canal_cells":        int(canal_mask.sum()),
        "basin_cells":        int(basin_mask.sum()),
        "road_cells":         int(road_mask.sum()),
        "num_frames":         num_frames,
        "wallclock_s":        round(time.time() - t0, 1),
    }


# =============================================================================
# MASTER TABLE WRITERS
# =============================================================================

MASTER_FIELDS = [
    "scenario_id", "scenario", "config",
    "rainfall_mm", "duration_h", "pattern",
    "peak_flooded_pct", "peak_flooded_ha", "peak_depth_mm", "peak_river_pct",
    "overflow_min", "rain_at_overflow",
    "final_flooded_pct", "final_depth_mm",
    "wall_cells", "canal_cells", "basin_cells", "road_cells",
    "num_frames", "wallclock_s",
]


def _write_master_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_master_json(rows: list[dict], path: Path,
                       total_area_ha: float, dt_min: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "generated":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system":        "Jade Valley Flood Simulation - 100% Build",
            "total_area_ha": round(total_area_ha, 2),
            "timestep_min":  dt_min,
            "n_runs":        len(rows),
            "results":       rows,
        }, f, indent=2)


def _fmt_or_dash(v):
    return "-" if v is None else str(v)


def _write_master_summary_txt(rows: list[dict], path: Path,
                              total_area_ha: float, dt_min: int) -> None:
    by_scen: dict = {}
    for r in rows:
        by_scen.setdefault(r["scenario"], []).append(r)

    lines = []
    lines.append("=" * 100)
    lines.append("  JADE VALLEY SUBDIVISION - MASTER RESULTS TABLE  (100% BUILD)")
    lines.append("  Davao City, Philippines")
    lines.append("=" * 100)
    lines.append(f"  Generated      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total area     : {total_area_ha:.2f} ha")
    lines.append(f"  Timestep       : {dt_min} min")
    lines.append(f"  Total runs     : {len(rows)}")
    lines.append("")
    lines.append("-" * 100)
    lines.append(f"  {'Scenario':<38} {'Config':<18} "
                 f"{'Peak%':>8} {'Peak ha':>9} {'MaxDep mm':>10} "
                 f"{'River%':>7} {'Onset min':>10}")
    lines.append("-" * 100)
    for scen, scen_rows in by_scen.items():
        for r in scen_rows:
            lines.append(
                f"  {scen:<38} {r['config']:<18} "
                f"{r['peak_flooded_pct']:>8.2f} {r['peak_flooded_ha']:>9.2f} "
                f"{r['peak_depth_mm']:>10.0f} {r['peak_river_pct']:>7.2f} "
                f"{_fmt_or_dash(r['overflow_min']):>10}"
            )
        lines.append("")
    lines.append("=" * 100)
    lines.append("  END OF MASTER RESULTS")
    lines.append("=" * 100)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_reduction_table_txt(rows: list[dict], path: Path) -> None:
    by_scen: dict = {}
    for r in rows:
        by_scen.setdefault(r["scenario"], {})[r["config"]] = r

    lines = []
    lines.append("=" * 100)
    lines.append("  JADE VALLEY SUBDIVISION - REDUCTION vs BASELINE")
    lines.append("=" * 100)
    lines.append("  Values are improved - baseline. Negative = reduction (good).")
    lines.append("")
    lines.append(f"  {'Scenario':<38} {'Config':<18} "
                 f"{'d Flooded%':>11} {'d Flooded ha':>13} {'d Depth mm':>11} "
                 f"{'Reduction %':>12}")
    lines.append("-" * 100)
    for scen, cfgs in by_scen.items():
        if "Baseline" not in cfgs:
            continue
        base = cfgs["Baseline"]
        for cfg_label, r in cfgs.items():
            if cfg_label == "Baseline":
                continue
            d_pct = r["peak_flooded_pct"] - base["peak_flooded_pct"]
            d_ha  = r["peak_flooded_ha"]  - base["peak_flooded_ha"]
            d_dep = r["peak_depth_mm"]    - base["peak_depth_mm"]
            red   = (d_pct / base["peak_flooded_pct"] * 100) if base["peak_flooded_pct"] > 0 else 0.0
            lines.append(
                f"  {scen:<38} {cfg_label:<18} "
                f"{d_pct:>+11.2f} {d_ha:>+13.2f} {d_dep:>+11.0f} {red:>+11.1f}%"
            )
        lines.append("")
    lines.append("=" * 100)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_master_dataset(rows: list[dict], total_area_ha: float, dt_min: int):
    """Write all four master files to DATA_OUT."""
    _write_master_csv          (rows, DATA_OUT / "master_results_table.csv")
    _write_master_json         (rows, DATA_OUT / "master_results_table.json",
                                total_area_ha, dt_min)
    _write_master_summary_txt  (rows, DATA_OUT / "master_results_summary.txt",
                                total_area_ha, dt_min)
    _write_reduction_table_txt (rows, DATA_OUT / "master_reduction_table.txt")


# =============================================================================
# FULL MATRIX RUNNER (headless)
# =============================================================================

def run_full_matrix(dem: np.ndarray, cellsize: float,
                    progress_cb=None, dt_min: int = MATRIX_DT_MIN) -> list[dict]:
    """Run the 6 x 4 = 24 simulation matrix. progress_cb(idx, total, label) optional."""
    rows: list[dict] = []
    total_runs = len(MATRIX_SCENARIOS) * len(MATRIX_CONFIGS)
    idx = 0
    for scen in MATRIX_SCENARIOS:
        for cfg in MATRIX_CONFIGS:
            idx += 1
            label = f"{scen[1]} | {cfg[0]}"
            if progress_cb:
                progress_cb(idx, total_runs, label)
            try:
                r = _run_one_headless(dem, cellsize, scen, cfg, dt_min)
                rows.append(r)
            except Exception as e:
                print(f"  [error] {label}: {e}")
                rows.append({"scenario_id": scen[0], "scenario": scen[1],
                             "config": cfg[0], "error": str(e)})
    return rows


# =============================================================================
# INTERACTIVE-RUN LOGGING
# =============================================================================

INTERACTIVE_LOG = DATA_OUT / "master_runs_log.csv"
INTERACTIVE_LOG_FIELDS = [
    "timestamp", "scenario", "rainfall_mm", "duration_h", "pattern",
    "use_wall", "use_canal", "use_basin", "use_road",
    "wall_height", "canal_depth", "basin_depth", "road_height",
    "wind_speed", "wind_dir", "soil_sat_pct", "drain_cap",
]


def append_interactive_log(row: dict) -> None:
    """Append one interactive run to master_runs_log.csv."""
    is_new = not INTERACTIVE_LOG.exists()
    with open(INTERACTIVE_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INTERACTIVE_LOG_FIELDS,
                           extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({**{k: "" for k in INTERACTIVE_LOG_FIELDS}, **row})


# =============================================================================
# GUI
# =============================================================================

class SimulationGUI100:
    """100% Build GUI.

    A wrapper around the 75% engine that adds a "COLLECT ALL DATA" button
    to drive the full matrix run from inside the same Tkinter window. The
    Storm tab and Prevention tab are identical to the 75% build.
    """

    # Colour palette (same as 75% build)
    BG     = '#0D1117'
    PANEL  = '#161B22'
    CARD   = '#1C2333'
    BORDER = '#30363D'
    TEXT   = '#E6EDF3'
    ACCENT = '#4FC3F7'
    GREEN  = '#2EA043'
    ORANGE = '#F0883E'
    RED    = '#DA3633'
    PURPLE = '#6E40C9'
    WHITE  = '#FFFFFF'
    WALL   = '#FF5555'
    CANAL  = '#00DDEE'
    BASIN  = '#3FB950'
    ROAD   = '#FF9800'

    def __init__(self, dem: np.ndarray, cellsize: float):
        self.dem = dem
        self.cellsize = cellsize
        # We import random here to mirror 75%
        import random as _random
        self._random = _random

        # Declare all attributes that are populated later by tab/_build helpers
        # so static analysis can see them. They are wired up by _build_storm_tab
        # and _build_prevention_tab below.
        self.scenario_var: tk.StringVar
        self.pattern_var:  tk.StringVar
        self.desc_var:     tk.StringVar
        self.start_time_var: tk.StringVar
        self.sliders: dict[str, tk.Scale] = {}
        self.fw_var: tk.BooleanVar
        self.cn_var: tk.BooleanVar
        self.rb_var: tk.BooleanVar
        self.er_var: tk.BooleanVar
        self.fw_badge: tk.Label
        self.cn_badge: tk.Label
        self.rb_badge: tk.Label
        self.er_badge: tk.Label
        self.wall_height_slider:  tk.Scale
        self.canal_depth_slider:  tk.Scale
        self.basin_depth_slider:  tk.Scale
        self.road_height_slider:  tk.Scale

        self.root = tk.Tk()
        self.root.title(
            "Jade Valley Flood Simulator - Complete System (100% Build)")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ww = min(1280, int(sw * 0.80))
        wh = min(820,  int(sh * 0.86))
        x = (sw - ww) // 2
        y = (sh - wh) // 2
        self.root.geometry(f"{ww}x{wh}+{x}+{y}")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame',        background=self.BG)
        style.configure('Card.TFrame',   background=self.CARD)
        style.configure('TNotebook',     background=self.BG, tabmargins=[2, 5, 0, 0])
        style.configure('TNotebook.Tab', background=self.PANEL,
                        foreground=self.TEXT, font=('Segoe UI', 11, 'bold'),
                        padding=[16, 7])
        style.map('TNotebook.Tab',
                  background=[('selected', self.CARD)],
                  foreground=[('selected', self.ACCENT)])
        style.configure('TLabel',  background=self.BG,
                        foreground=self.TEXT, font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=self.BG,
                        foreground=self.ACCENT, font=('Segoe UI', 12, 'bold'))
        style.configure('Title.TLabel',  background=self.BG,
                        foreground=self.WHITE, font=('Segoe UI', 16, 'bold'))
        style.configure('Desc.TLabel',   background=self.CARD,
                        foreground='#8B949E', font=('Segoe UI', 9))
        style.configure('TCheckbutton',  background=self.BG,
                        foreground=self.TEXT, font=('Segoe UI', 10))
        style.configure("Black.Horizontal.TProgressbar",
                        troughcolor=self.PANEL, background=self.ACCENT,
                        lightcolor=self.ACCENT, darkcolor=self.ACCENT)

        # Title
        ttk.Label(self.root, text="JADE VALLEY SUBDIVISION",
                  style='Title.TLabel').pack(pady=(14, 0))
        ttk.Label(self.root,
                  text="Complete Flood Simulation System  -  100% Build  "
                       "|  4 Prevention Measures + Full Data Collection",
                  style='Desc.TLabel').pack()

        # Notebook (two tabs)
        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=16, pady=10)

        tab_storm = ttk.Frame(nb)
        tab_prev  = ttk.Frame(nb)
        nb.add(tab_storm, text='Storm Scenario')
        nb.add(tab_prev,  text='Prevention Measures')

        self._build_storm_tab(tab_storm)
        self._build_prevention_tab(tab_prev)

        # ── Progress indicator + status (shared by interactive + matrix) ─
        self._progress_frame = tk.Frame(self.root, bg=self.BG)
        self._progress_frame.pack(fill='x', padx=16, pady=(0, 4))
        self.progress_label = tk.Label(
            self._progress_frame,
            text="Ready.",
            bg=self.BG, fg=self.ACCENT,
            font=('Consolas', 9, 'bold'), anchor='w')
        self.progress_label.pack(fill='x', pady=(0, 2))
        self.progress_bar = ttk.Progressbar(
            self._progress_frame,
            style="Black.Horizontal.TProgressbar",
            mode='determinate', maximum=100, value=0)
        self.progress_bar.pack(fill='x')

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=16, pady=(4, 14))
        _BF = ('Segoe UI', 10, 'bold')

        tk.Button(btn_frame, text="RANDOMIZE",
                  bg=self.PURPLE, fg='white', activebackground='#8957E5',
                  font=_BF, width=14, height=1, bd=0, cursor='hand2',
                  command=self._randomize).pack(side='left', padx=4)
        tk.Button(btn_frame, text="▶  RUN SIMULATION",
                  bg=self.GREEN, fg='white', activebackground='#3FB950',
                  font=_BF, width=20, height=1, bd=0, cursor='hand2',
                  command=self._run_interactive).pack(side='left', padx=4)
        tk.Button(btn_frame, text="■  COLLECT ALL DATA  (6x4 matrix)",
                  bg=self.ORANGE, fg='black', activebackground='#FFB74D',
                  font=_BF, width=28, height=1, bd=0, cursor='hand2',
                  command=self._run_full_matrix).pack(side='left', padx=4)
        tk.Button(btn_frame, text="✕  EXIT",
                  bg=self.RED, fg='white', activebackground='#F85149',
                  font=_BF, width=10, height=1, bd=0, cursor='hand2',
                  command=self.root.destroy).pack(side='right', padx=4)

        # Info bar
        info = (f"DEM: {dem.shape[0]}x{dem.shape[1]}  |  "
                f"Cell: {cellsize:.1f} m  |  "
                f"Elev: {dem.min():.1f}-{dem.max():.1f} m  |  "
                f"Area: {dem.size * cellsize**2 / 10000:.1f} ha  |  "
                f"Matrix output: Results/data/master_results_*.csv/.json/.txt")
        tk.Label(self.root, text=info, bg=self.BORDER, fg='#8B949E',
                 font=('Consolas', 9), pady=4,
                 wraplength=ww-40, justify='center').pack(fill='x', side='bottom')

        self._is_randomized = False
        self._on_preset_change()
        self.root.mainloop()

    # ── Tab 1: Storm Scenario ─────────────────────────────────────────────

    def _on_preset_change(self, *_):
        self._is_randomized = False
        key = self.scenario_var.get()
        sc = SCENARIOS_75.get(key)
        if sc and sc["rainfall_mm"] is not None:
            self.sliders['rainfall_mm'].set(sc["rainfall_mm"])
            self.sliders['duration_h'].set(sc["duration_h"])
            self.pattern_var.set(sc["pattern"])
            self.desc_var.set(sc.get("desc", ""))
        else:
            self.desc_var.set("")
        self._update_measure_display()

    def _build_storm_tab(self, parent):
        main = ttk.Frame(parent)
        main.pack(fill='both', expand=True, padx=12, pady=10)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        left  = ttk.Frame(main); left .grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        right = ttk.Frame(main); right.grid(row=0, column=1, sticky='nsew', padx=(8, 0))

        ttk.Label(left, text="SCENARIO PRESET",
                  style='Header.TLabel').pack(anchor='w')
        self.scenario_var = tk.StringVar(value="3")
        sc_frame = ttk.Frame(left); sc_frame.pack(fill='x', pady=4)
        _SC_COLORS = {"1":"#3FB950","2":"#79C0FF","3":"#FFD740",
                      "4":"#F0883E","5":"#FF6B6B","6":"#FF1744"}
        for key, sc in SCENARIOS_75.items():
            if sc["rainfall_mm"] is None: continue
            color = _SC_COLORS.get(key, self.TEXT)
            label = f"{sc['name']}  ({sc['rainfall_mm']} mm / {sc['duration_h']} h)"
            tk.Radiobutton(
                sc_frame, text=label,
                variable=self.scenario_var, value=key,
                bg=self.BG, fg=color, selectcolor=self.PANEL,
                activebackground=self.BG, activeforeground=self.ACCENT,
                font=('Segoe UI', 9), anchor='w',
                command=self._on_preset_change).pack(fill='x', pady=1)

        self.desc_var = tk.StringVar()
        tk.Label(sc_frame, textvariable=self.desc_var, bg=self.CARD,
                 fg='#8B949E', font=('Segoe UI', 9, 'italic'),
                 wraplength=380, justify='left', padx=8, pady=6
                 ).pack(fill='x', pady=(6, 0))

        # PAGASA reference bands
        ref_frame = tk.Frame(sc_frame, bg=self.CARD)
        ref_frame.pack(fill='x', padx=4, pady=(4, 2))
        tk.Label(ref_frame, text="PAGASA Classification (mm/hr):",
                 bg=self.CARD, fg='#8B949E',
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=4)
        band_row = tk.Frame(ref_frame, bg=self.CARD)
        band_row.pack(fill='x', padx=4, pady=2)
        for lbl, col in [("Light  0-7.5","#3FB950"),
                         ("Moderate 7.5-15","#79C0FF"),
                         ("Heavy 15-30","#FFD740"),
                         ("Intense >30","#FF6B6B")]:
            tk.Label(band_row, text=f"  {lbl}  ", bg=col, fg='#0D1117',
                     font=('Segoe UI', 7, 'bold'), padx=2
                     ).pack(side='left', padx=2, pady=1)

        # Storm pattern
        sep = ttk.Frame(left); sep.pack(fill='x', pady=8)
        ttk.Label(sep, text="STORM PATTERN",
                  style='Header.TLabel').pack(anchor='w')
        self.pattern_var = tk.StringVar(value="burst")
        for val, desc in [("uniform","Constant rate"),
                          ("progressive","Builds up -> peaks"),
                          ("burst","Bell-curve peak mid-storm"),
                          ("decreasing","Heavy start -> tapers off")]:
            tk.Radiobutton(sep, text=f"{val}  -  {desc}",
                           variable=self.pattern_var, value=val,
                           bg=self.BG, fg=self.TEXT, selectcolor=self.PANEL,
                           activebackground=self.BG, activeforeground=self.ACCENT,
                           font=('Segoe UI', 9), anchor='w').pack(fill='x', pady=1)

        # Parameter sliders
        ttk.Label(right, text="SIMULATION PARAMETERS",
                  style='Header.TLabel').pack(anchor='w')
        self.sliders = {}

        def add_slider(par, key, label, lo, hi, default, res=1.0):
            f = ttk.Frame(par); f.pack(fill='x', pady=2)
            ttk.Label(f, text=label).pack(anchor='w')
            s = tk.Scale(f, from_=lo, to=hi, orient='horizontal', resolution=res,
                         length=300, bg=self.BG, fg=self.TEXT,
                         troughcolor=self.PANEL, highlightthickness=0,
                         font=('Consolas', 9), activebackground=self.ACCENT)
            s.set(default); s.pack(fill='x')
            self.sliders[key] = s

        add_slider(right, 'rainfall_mm', 'Rainfall Total (mm)       [5-500]',      5,   500, 130)
        add_slider(right, 'duration_h',  'Storm Duration (hours)    [0.5-24]',     0.5,  24,  4.0, res=0.5)
        add_slider(right, 'timestep_min','Timestep (minutes)        [5-60]',        5,   60,  10)
        add_slider(right, 'wind_speed',  'Wind Speed (km/h)         [0-200]',       0,  200,   0)
        add_slider(right, 'wind_dir',    'Wind Direction (deg)      [0-360]',       0,  359, 270)
        add_slider(right, 'soil_sat',    'Soil Saturation (%)       [0-100]',       0,  100,  30)
        add_slider(right, 'drain_cap',   'Drainage Capacity (mm/hr) [0.5-50]',    0.5,  50,   5.0, res=0.5)

        tf = ttk.Frame(right); tf.pack(fill='x', pady=4)
        ttk.Label(tf, text='Start Time (HH:MM)').pack(anchor='w')
        self.start_time_var = tk.StringVar(value="14:00")
        tk.Entry(tf, textvariable=self.start_time_var, bg=self.PANEL, fg=self.TEXT,
                 insertbackground=self.TEXT, font=('Consolas', 11), width=8
                 ).pack(anchor='w')

    # ── Tab 2: Prevention Measures (4 cards) ──────────────────────────────

    def _update_measure_display(self, *_):
        def _set(badge, var, on_col, off_col):
            if hasattr(self, badge) and hasattr(self, var):
                getattr(self, badge).config(
                    text="ON" if getattr(self, var).get() else "OFF",
                    bg=on_col if getattr(self, var).get() else off_col)
        _set('fw_badge','fw_var','#3FB950','#AA2222')
        _set('cn_badge','cn_var','#3FB950','#008899')
        _set('rb_badge','rb_var','#3FB950','#1a5c28')
        _set('er_badge','er_var','#3FB950','#7a4800')

    def _build_prevention_tab(self, parent):
        outer = ttk.Frame(parent)
        outer.pack(fill='both', expand=True, padx=12, pady=10)
        outer.columnconfigure(0, weight=1); outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1);    outer.rowconfigure(2, weight=1)

        tk.Label(outer,
                 text="Enable any combination of measures, then click RUN SIMULATION "
                      "on the Storm tab. All measures physically modify the DEM. "
                      "Or click COLLECT ALL DATA to ignore these toggles and run the "
                      "full 6x4 evaluation matrix.",
                 bg=self.CARD, fg='#8B949E', font=('Segoe UI', 9, 'italic'),
                 wraplength=780, justify='left', padx=14, pady=8
                 ).grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))

        # Card builder helper
        def _card(parent_, grid_pos, title, accent, off_col, var_name, badge_name,
                  description, slider_label, slider_range, slider_default,
                  slider_res, effect_text, hilite):
            card = tk.Frame(parent_, bg=self.CARD, bd=0,
                            highlightthickness=2, highlightbackground=hilite)
            card.grid(**grid_pos)
            hdr = tk.Frame(card, bg=accent); hdr.pack(fill='x')
            v = tk.BooleanVar(value=False)
            setattr(self, var_name, v)
            tk.Checkbutton(
                hdr, text=f"  {title}",
                variable=v,
                bg=accent, fg='white' if accent != self.CANAL else '#003333',
                selectcolor=off_col, activebackground=accent,
                activeforeground='white' if accent != self.CANAL else '#003333',
                font=('Segoe UI', 11, 'bold'), anchor='w',
                command=self._update_measure_display).pack(side='left', padx=6, pady=6)
            badge = tk.Label(hdr, text="OFF", bg=off_col, fg='white',
                             font=('Segoe UI', 9, 'bold'), padx=8, pady=2)
            badge.pack(side='right', padx=8)
            setattr(self, badge_name, badge)
            tk.Label(card, text=description,
                     bg=self.CARD, fg='#8B949E', font=('Segoe UI', 9),
                     justify='left', padx=14, pady=6).pack(anchor='w')
            sf = tk.Frame(card, bg=self.CARD); sf.pack(fill='x', padx=12)
            tk.Label(sf, text=slider_label,
                     bg=self.CARD, fg=self.TEXT,
                     font=('Segoe UI', 10)).pack(anchor='w')
            s = tk.Scale(sf, from_=slider_range[0], to=slider_range[1],
                         orient='horizontal', resolution=slider_res,
                         length=260, bg=self.CARD, fg=accent,
                         troughcolor=self.PANEL, highlightthickness=0,
                         font=('Consolas', 9), activebackground=accent)
            s.set(slider_default); s.pack(fill='x')
            tk.Label(card, text=effect_text,
                     bg=self.CARD, fg='#3FB950', font=('Segoe UI', 9, 'italic'),
                     padx=14, pady=4).pack(anchor='w', pady=(0, 8))
            return s

        self.wall_height_slider = _card(
            outer, dict(row=1, column=0, sticky='nsew', padx=(0,5), pady=(0,5)),
            "Riverbank Floodwall", self.WALL, '#AA2222',
            'fw_var', 'fw_badge',
            "Raises the western river-bank cells by the chosen height.\n"
            "Creates a physical DEM barrier - flood routing cannot spill\n"
            "until the river level exceeds the raised crest elevation.",
            "Wall Height (m)  [1.0 - 3.0]", (1.0, 3.0), 1.5, 0.25,
            "Effect: delays 100-yr flood onset by 2-4 hrs",
            '#3a1a1a')
        self.canal_depth_slider = _card(
            outer, dict(row=1, column=1, sticky='nsew', padx=(5,0), pady=(0,5)),
            "Drainage Canal Network", self.CANAL, '#008899',
            'cn_var', 'cn_badge',
            "Lowers a corridor of cells (east outlet + south branch)\n"
            "to create open channels in the DEM. Runoff naturally flows\n"
            "into the canals, draining away from the residential core.",
            "Canal Depth (m)  [1.0 - 4.0]", (1.0, 4.0), 2.0, 0.25,
            "Effect: reduces inundation area by ~15-25%",
            '#003a3a')
        self.basin_depth_slider = _card(
            outer, dict(row=2, column=0, sticky='nsew', padx=(0,5), pady=(5,0)),
            "Retention Basin", self.BASIN, '#1a5c28',
            'rb_var', 'rb_badge',
            "Excavates a stormwater pond on undeveloped perimeter land.\n"
            "Intercepts runoff, stores it at peak rainfall, releases\n"
            "slowly - cuts inundation depth across the residential core.",
            "Basin Depth (m)  [3.0 - 10.0]", (3.0, 10.0), 6.0, 0.5,
            "Effect: lowers peak depth by ~20-40%; basin fills live",
            '#1a3a1a')
        self.road_height_slider = _card(
            outer, dict(row=2, column=1, sticky='nsew', padx=(5,0), pady=(5,0)),
            "Elevated Emergency Road", self.ROAD, '#7a4800',
            'er_var', 'er_badge',
            "Raises a cross-subdivision road corridor above flood level.\n"
            "Acts as a berm that guides flow east-west while keeping\n"
            "emergency vehicle access open throughout the flood event.",
            "Road Height (m)  [0.5 - 3.0]", (0.5, 3.0), 1.5, 0.25,
            "Effect: diverts runoff; road stays dry up to design flood",
            '#3a2a00')

    # ── Randomize ─────────────────────────────────────────────────────────

    def _randomize(self):
        classes = [
            ("Light Rain",        10,  50,  1.0,  3.0),
            ("Moderate Rain",     40, 100,  2.0,  5.0),
            ("Heavy Rain",        80, 180,  3.0,  6.0),
            ("Typhoon Signal 1", 140, 220,  4.0,  8.0),
            ("Typhoon Signal 2", 200, 300,  6.0, 10.0),
            ("Typhoon Signal 3", 280, 450,  8.0, 14.0),
        ]
        name, rlo, rhi, dlo, dhi = self._random.choice(classes)
        rain = self._random.randint(rlo, rhi)
        dur  = round(self._random.uniform(dlo, dhi) * 2) / 2
        wind = self._random.choice([0, 0, self._random.randint(20, 60),
                                    self._random.randint(60, 120),
                                    self._random.randint(100, 180)])
        self.sliders['rainfall_mm'].set(rain)
        self.sliders['duration_h'].set(dur)
        self.sliders['timestep_min'].set(10)
        self.sliders['wind_speed'].set(wind)
        self.sliders['wind_dir'].set(self._random.randint(0, 359))
        self.sliders['soil_sat'].set(self._random.randint(10, 85))
        self.sliders['drain_cap'].set(self._random.choice([1.5,2.0,3.0,5.0,8.0]))
        self.pattern_var.set(self._random.choice(
            ["uniform","progressive","burst","decreasing"]))
        self.start_time_var.set(f"{self._random.randint(0,23):02d}:00")
        self._is_randomized = True
        self.desc_var.set(f"Randomized: {name} | {rain} mm / {dur} h")
        # Reasonable defaults for prevention based on rainfall
        if rain <= 36:
            self.fw_var.set(False); self.cn_var.set(False)
            self.rb_var.set(False); self.er_var.set(False)
        elif rain <= 90:
            self.fw_var.set(True);  self.cn_var.set(False)
            self.rb_var.set(False); self.er_var.set(False)
        elif rain <= 150:
            self.fw_var.set(True);  self.cn_var.set(True)
            self.rb_var.set(False); self.er_var.set(False)
        else:
            self.fw_var.set(True);  self.cn_var.set(True)
            self.rb_var.set(self._random.choice([True, False]))
            self.er_var.set(self._random.choice([True, False]))
        self._update_measure_display()

    # ── Run handlers ──────────────────────────────────────────────────────

    def _set_progress(self, pct: float, label: str):
        self.progress_bar['value'] = max(0, min(100, pct))
        self.progress_label.config(text=label)
        self.root.update_idletasks()

    def _gather_params(self) -> dict:
        return dict(
            rainfall_mm  = float(self.sliders['rainfall_mm'].get()),
            duration_h   = float(self.sliders['duration_h'].get()),
            timestep_min = int(self.sliders['timestep_min'].get()),
            wind_speed   = float(self.sliders['wind_speed'].get()),
            wind_dir     = float(self.sliders['wind_dir'].get()),
            soil_sat_pct = float(self.sliders['soil_sat'].get()),
            drain_cap    = float(self.sliders['drain_cap'].get()),
            pattern      = self.pattern_var.get(),
            start_time_str = self.start_time_var.get().strip(),
            use_floodwall = self.fw_var.get(),
            use_canal     = self.cn_var.get(),
            use_basin     = self.rb_var.get(),
            use_road      = self.er_var.get(),
            wall_height   = float(self.wall_height_slider.get()),
            canal_depth   = float(self.canal_depth_slider.get()),
            basin_depth   = float(self.basin_depth_slider.get()),
            road_height   = float(self.road_height_slider.get()),
        )

    def _run_interactive(self):
        p = self._gather_params()
        # Validate start time
        try:
            h, m = map(int, p['start_time_str'].split(':'))
            if not (0 <= h < 24 and 0 <= m < 60): raise ValueError
        except (ValueError, AttributeError):
            p['start_time_str'] = "14:00"

        if self._is_randomized:
            scen_name = (f"Randomized ({p['rainfall_mm']:.0f} mm / "
                         f"{p['duration_h']:.1f} h)")
        else:
            key = self.scenario_var.get()
            sc  = SCENARIOS_75.get(key)
            scen_name = (sc["name"] if sc and sc["rainfall_mm"] is not None
                         else f"Custom ({p['rainfall_mm']:.0f} mm / "
                              f"{p['duration_h']:.1f} h)")
        if p['wind_speed'] >= 100:
            scen_name += f" + Wind {p['wind_speed']:.0f} km/h"

        # Log this interactive run
        append_interactive_log({
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario":    scen_name,
            "rainfall_mm": p["rainfall_mm"],
            "duration_h":  p["duration_h"],
            "pattern":     p["pattern"],
            "use_wall":    p["use_floodwall"],
            "use_canal":   p["use_canal"],
            "use_basin":   p["use_basin"],
            "use_road":    p["use_road"],
            "wall_height": p["wall_height"],
            "canal_depth": p["canal_depth"],
            "basin_depth": p["basin_depth"],
            "road_height": p["road_height"],
            "wind_speed":  p["wind_speed"],
            "wind_dir":    p["wind_dir"],
            "soil_sat_pct": p["soil_sat_pct"],
            "drain_cap":   p["drain_cap"],
        })

        self.root.destroy()
        # Delegate to the 75% engine
        run_simulation_75(
            dem=self.dem, cellsize=self.cellsize,
            rainfall_mm=p['rainfall_mm'], duration_h=p['duration_h'],
            timestep_min=p['timestep_min'],
            start_time_str=p['start_time_str'],
            wind_speed=p['wind_speed'], wind_dir=p['wind_dir'],
            soil_sat_pct=p['soil_sat_pct'], drain_cap=p['drain_cap'],
            pattern=p['pattern'], scenario_name=scen_name,
            use_floodwall=p['use_floodwall'], use_canal=p['use_canal'],
            use_basin=p['use_basin'], use_road=p['use_road'],
            wall_height=p['wall_height'], canal_depth=p['canal_depth'],
            basin_depth=p['basin_depth'], road_height=p['road_height'],
        )

    def _run_full_matrix(self):
        """Run the 6x4 matrix in the same Tk thread (it is fast - seconds)."""
        # 1) Ensure HAND/risk-zone artefacts exist
        self._set_progress(2, "[1/3] Verifying HAND risk-zone artefacts...")
        hand_mod.run_pipeline(verbose=False, force=False)

        # 2) Run the matrix
        total = len(MATRIX_SCENARIOS) * len(MATRIX_CONFIGS)
        rows: list[dict] = []
        idx = 0
        for scen in MATRIX_SCENARIOS:
            for cfg in MATRIX_CONFIGS:
                idx += 1
                pct = 5 + (idx / total) * 88
                self._set_progress(pct, f"[2/3] Run {idx}/{total}: "
                                        f"{scen[1]} | {cfg[0]}")
                try:
                    rows.append(_run_one_headless(
                        self.dem, self.cellsize, scen, cfg, MATRIX_DT_MIN))
                except Exception as e:
                    rows.append({"scenario_id": scen[0], "scenario": scen[1],
                                 "config": cfg[0], "error": str(e)})

        # 3) Export master tables
        self._set_progress(96, "[3/3] Writing master tables...")
        total_area_ha = float(self.dem.size * self.cellsize ** 2 / 10_000.0)
        export_master_dataset(rows, total_area_ha, MATRIX_DT_MIN)
        self._set_progress(
            100,
            f"Done. {len(rows)} runs collected. See Results/data/master_results_*.")
        # Print a console summary too
        print("\n[100% BUILD] Master matrix complete.")
        print(f"  Wrote: master_results_table.csv / .json / _summary.txt / "
              "_reduction_table.txt")


# =============================================================================
# CLI WRAPPER (for headless invocations, e.g. CI or batch)
# =============================================================================

def cli_main():
    ap = argparse.ArgumentParser(
        description="Jade Valley Flood Simulation - 100% Complete System.")
    ap.add_argument("--matrix-only", action="store_true",
                    help="Skip GUI; run the full 6x4 matrix headlessly and exit.")
    ap.add_argument("--quick", action="store_true",
                    help="With --matrix-only: smoke matrix (3 scenarios x 2 configs).")
    ap.add_argument("--no-hand", action="store_true",
                    help="Skip the HAND risk-zone analysis.")
    ap.add_argument("--force-hand", action="store_true",
                    help="Force regenerate HAND canonical files.")
    ap.add_argument("--dt", type=int, default=MATRIX_DT_MIN,
                    help=f"Timestep in minutes (default: {MATRIX_DT_MIN}).")
    args = ap.parse_args()

    print("=" * 70)
    print("  JADE VALLEY SUBDIVISION - FLOOD SIMULATION  (100% COMPLETE BUILD)")
    print("  Davao City, Philippines")
    print("=" * 70)
    print("\nLoading terrain data...")
    dem, cellsize = load_dem()

    if args.matrix_only:
        if not args.no_hand:
            print("\n[1/3] HAND / risk-zone analysis")
            hand_mod.run_pipeline(verbose=True, force=args.force_hand)
        else:
            print("\n[1/3] HAND / risk-zone analysis  (skipped)")

        scenarios = MATRIX_SCENARIOS
        configs   = MATRIX_CONFIGS
        if args.quick:
            scenarios = [MATRIX_SCENARIOS[0], MATRIX_SCENARIOS[2], MATRIX_SCENARIOS[5]]
            configs   = [MATRIX_CONFIGS[0],   MATRIX_CONFIGS[2]]
        n = len(scenarios) * len(configs)
        print(f"\n[2/3] Running {n} simulations (timestep {args.dt} min)")
        rows: list[dict] = []
        idx = 0
        for scen in scenarios:
            for cfg in configs:
                idx += 1
                print(f"  [{idx}/{n}] {scen[1]} | {cfg[0]}", flush=True)
                try:
                    r = _run_one_headless(dem, cellsize, scen, cfg, args.dt)
                    rows.append(r)
                    print(f"    -> peak {r['peak_flooded_pct']:.2f}% "
                          f"({r['peak_flooded_ha']:.1f} ha)  "
                          f"max depth {r['peak_depth_mm']:.0f} mm")
                except Exception as e:
                    print(f"    [error] {e}")
                    rows.append({"scenario_id": scen[0], "scenario": scen[1],
                                 "config": cfg[0], "error": str(e)})
        total_area_ha = float(dem.size * cellsize ** 2 / 10_000.0)
        print("\n[3/3] Writing master tables...")
        export_master_dataset(rows, total_area_ha, args.dt)
        print("\n  Wrote master_results_table.csv / .json / _summary.txt / "
              "_reduction_table.txt")
        print("\n" + "=" * 70)
        print("  100% MATRIX COMPLETE")
        print("=" * 70)
        return 0

    # Default: launch GUI
    if not args.no_hand:
        print("\nVerifying HAND risk-zone artefacts...")
        hand_mod.run_pipeline(verbose=False, force=args.force_hand)

    print("\nLaunching GUI...")
    SimulationGUI100(dem, cellsize)
    print("\n" + "=" * 70)
    print("  DONE - See Results/data/ and Results/animations/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
