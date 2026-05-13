"""
batch_comparison.py
───────────────────
Headless batch runner — no GUI, no animation.
Runs all 6 rainfall scenarios × 4 prevention configurations
(Baseline / Small / Medium / Large prevention measures)
and prints peak flood statistics for the comparison tables.

Usage:
    python Main/batch_comparison.py
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── Load the main simulation module ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
SIM_FILE = BASE_DIR / "Main" / "flood_simulation_75%.py"

spec = importlib.util.spec_from_file_location("flood_sim", str(SIM_FILE))
mod  = importlib.util.module_from_spec(spec)          # type: ignore
spec.loader.exec_module(mod)                           # type: ignore

load_dem                  = mod.load_dem
apply_prevention_measures = mod.apply_prevention_measures
FloodSimulation           = mod.FloodSimulation
_intensity_factor         = mod._intensity_factor
wind_rainfall_map         = mod.wind_rainfall_map

# ── Load DEM once ─────────────────────────────────────────────────────────────
print("Loading DEM…")
dem, cellsize = load_dem()
TOTAL_AREA_HA = 326.44

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS = [
    ("Light Rain",      15,   2.0,  "uniform"),
    ("Moderate Rain",   36,   3.0,  "progressive"),
    ("Heavy Rain",      90,   4.0,  "progressive"),
    ("Typhoon Sig 1",  150,   8.0,  "progressive"),
    ("Typhoon Sig 2",  250,  12.0,  "burst"),
    ("Typhoon Sig 3",  400,  18.0,  "burst"),
]

# ── Prevention size variants ───────────────────────────────────────────────────
# (label, use_wall, use_canal, use_basin, use_road,
#          wall_h, canal_d, basin_d, road_h)
CONFIGS = [
    ("Baseline", False, False, False, False, 1.5, 2.0, 6.0, 1.5),
    ("Small",    True,  True,  True,  True,  1.0, 1.0, 3.0, 1.0),
    ("Medium",   True,  True,  True,  True,  1.5, 2.0, 6.0, 1.5),
    ("Large",    True,  True,  True,  True,  2.5, 3.0, 9.0, 2.5),
]

# Use 20-min timestep to keep total runtime under ~5 minutes
DT_MIN = 20

# ── Runner ────────────────────────────────────────────────────────────────────
def run_one(dem, cellsize, rainfall_mm, duration_h, pattern,
            use_wall, use_canal, use_basin, use_road,
            wall_h, canal_d, basin_d, road_h):
    """Run one simulation and return peak statistics."""
    if use_wall or use_canal or use_basin or use_road:
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
    dt_h       = DT_MIN / 60.0
    num_frames = int(np.ceil(duration_h / dt_h))

    # No wind — consistent baseline for comparison
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
    )

    peak_flooded_pct  = 0.0
    peak_depth_mm     = 0.0
    overflow_frame    = None
    rain_at_overflow  = None

    for fr in range(num_frames):
        inten = _intensity_factor(fr, num_frames, pattern)
        sim.step(rate_mmhr, dt_h, intensity=inten, wind_map=wmap)

        total       = sim.rain_water + sim.river_water
        flooded_pct = float(np.sum(total > 0.01) / total.size * 100)
        river_pct   = float(np.sum(sim.river_water > 0.005) / total.size * 100)
        max_depth   = float(total.max() * 1000)
        rain_mm_now = float(sim.rainfall_accumulated * 1000)

        if flooded_pct > peak_flooded_pct:
            peak_flooded_pct = flooded_pct
        if max_depth > peak_depth_mm:
            peak_depth_mm = max_depth

        # Overflow onset = first frame where river_pct > 0.5%
        if overflow_frame is None and river_pct > 0.5:
            overflow_frame   = (fr + 1) * DT_MIN   # minutes from start
            rain_at_overflow = rain_mm_now

    return {
        "peak_flooded_pct" : round(peak_flooded_pct, 2),
        "peak_flooded_ha"  : round(peak_flooded_pct / 100.0 * TOTAL_AREA_HA, 1),
        "peak_depth_mm"    : round(peak_depth_mm, 0),
        "overflow_min"     : overflow_frame,
        "rain_at_overflow" : round(rain_at_overflow, 1) if rain_at_overflow else None,
    }

# ── Run all combinations ──────────────────────────────────────────────────────
results = {}
total_runs = len(SCENARIOS) * len(CONFIGS)
run_idx    = 0
for scen_name, rainfall_mm, duration_h, pattern in SCENARIOS:
    results[scen_name] = {}
    for cfg in CONFIGS:
        cfg_name = cfg[0]
        run_idx += 1
        print(f"\n[{run_idx}/{total_runs}] {scen_name} / {cfg_name}", flush=True)
        r = run_one(dem, cellsize, rainfall_mm, duration_h, pattern, *cfg[1:])
        results[scen_name][cfg_name] = r
        ot = f"{r['overflow_min']} min" if r['overflow_min'] else "None"
        print(f"  peak={r['peak_flooded_pct']:.2f}%  {r['peak_flooded_ha']:.1f} ha  "
              f"depth={r['peak_depth_mm']:.0f} mm  overflow@{ot}", flush=True)

# ── Print final table ─────────────────────────────────────────────────────────
print("\n\n" + "=" * 100)
print("FULL COMPARISON TABLE")
print("=" * 100)
header = (f"{'Scenario':<20} {'Config':<10} "
          f"{'Peak Flooded%':>14} {'Peak Flooded ha':>16} "
          f"{'Max Depth mm':>13} {'Overflow min':>13} {'Rain@Onset mm':>14}")
print(header)
print("-" * 100)
for scen_name, *_ in SCENARIOS:
    for cfg in CONFIGS:
        cfg_name = cfg[0]
        r = results[scen_name][cfg_name]
        ot = str(r['overflow_min']) if r['overflow_min'] else "—"
        ra = str(r['rain_at_overflow']) if r['rain_at_overflow'] else "—"
        print(f"{scen_name:<20} {cfg_name:<10} "
              f"{r['peak_flooded_pct']:>14.2f} {r['peak_flooded_ha']:>16.1f} "
              f"{r['peak_depth_mm']:>13.0f} {ot:>13} {ra:>14}")
    print()

# ── Print reduction table ─────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("REDUCTION vs BASELINE")
print("=" * 100)
header2 = (f"{'Scenario':<20} {'Config':<10} "
           f"{'Δ Flooded%':>11} {'Δ Flooded ha':>13} {'Δ Depth mm':>11} "
           f"{'Red. %':>8}")
print(header2)
print("-" * 100)
for scen_name, *_ in SCENARIOS:
    base = results[scen_name]["Baseline"]
    for cfg in CONFIGS[1:]:   # skip baseline itself
        cfg_name = cfg[0]
        r = results[scen_name][cfg_name]
        d_pct  = r["peak_flooded_pct"] - base["peak_flooded_pct"]
        d_ha   = r["peak_flooded_ha"]  - base["peak_flooded_ha"]
        d_dep  = r["peak_depth_mm"]    - base["peak_depth_mm"]
        red_pct = (d_pct / base["peak_flooded_pct"] * 100) if base["peak_flooded_pct"] > 0 else 0
        print(f"{scen_name:<20} {cfg_name:<10} "
              f"{d_pct:>+11.2f} {d_ha:>+13.1f} {d_dep:>+11.0f} "
              f"{red_pct:>+7.1f}%")
    print()

print("\nDone.")
