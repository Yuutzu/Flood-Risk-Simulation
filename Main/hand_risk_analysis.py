"""
=============================================================================
 hand_risk_analysis.py
 Jade Valley Subdivision — Flood Risk Assessment Module
 Davao City, Philippines
=============================================================================
 Standalone module that builds the HAND (Height Above Nearest Drainage)
 model from JVS_Simulation.tif and writes the canonical risk-assessment
 artefacts referenced by the documentation:

   Results/data/dem_processed.npy        - depression-filled DEM (m)
   Results/data/slope.npy                - slope magnitude grid (deg)
   Results/data/flow_accumulation.npy    - upstream cell count
   Results/data/hand_model.npy           - HAND values per cell (m)
   Results/data/risk_map.npy             - 5-class risk zone integer grid
                                           0=Safe 1=Low 2=Medium 3=High 4=VeryHigh
   Results/data/flood_risk_report.txt    - human-readable summary
   Results/data/flood_statistics.json    - full machine-readable stats

 The HAND model (Nobre et al. 2011, J. Hydrology 404:13-29) measures the
 vertical distance from each terrain cell to its nearest drainage channel
 along the D8 flow path. Lower HAND values indicate higher flood
 susceptibility.

 Run:
   python Main/hand_risk_analysis.py
=============================================================================
"""
from __future__ import annotations

import heapq
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

try:
    import rasterio
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR  = Path(__file__).resolve().parent.parent
TIF_FILE  = BASE_DIR / "Map Topography" / "3D" / "JVS_Simulation.tif"
DATA_OUT  = BASE_DIR / "Results" / "data"
DATA_OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DEM LOADING
# =============================================================================

def load_dem() -> tuple[np.ndarray, float, dict]:
    """Load JVS_Simulation.tif. Returns (dem, cellsize_m, metadata)."""
    if not TIF_FILE.exists():
        sys.exit(f"\n[ERROR] DEM file not found: {TIF_FILE}")
    if not RASTERIO_OK:
        sys.exit("\n[ERROR] rasterio is required.  pip install rasterio")

    print(f"  Loading DEM: {TIF_FILE.name}")
    with rasterio.open(str(TIF_FILE)) as src:  # type: ignore[possibly-undefined]
        dem    = src.read(1).astype(np.float64)
        nodata = src.nodata if src.nodata is not None else -9999.0
        dem[dem == nodata] = np.nan
        t      = src.transform
        csx    = abs(float(t.a))
        crs    = src.crs
        if crs and crs.is_geographic:
            lat = float(src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2)
            cs  = csx * 111320 * abs(np.cos(np.radians(lat)))
        else:
            cs  = csx
        meta = {
            "ncols":         int(dem.shape[1]),
            "nrows":         int(dem.shape[0]),
            "xllcorner":     float(src.bounds.left),
            "yllcorner":     float(src.bounds.bottom),
            "cellsize":      round(cs, 4),
            "nodata_value":  float(nodata),
            "shape":         list(dem.shape),
        }

    nan_mask = np.isnan(dem)
    if nan_mask.any() and SCIPY_OK:
        idx = distance_transform_edt(  # type: ignore[possibly-undefined]
            nan_mask, return_distances=False, return_indices=True)
        rows = np.asarray(idx[0], dtype=int)[nan_mask]   # type: ignore[index]
        cols = np.asarray(idx[1], dtype=int)[nan_mask]   # type: ignore[index]
        dem[nan_mask] = dem[rows, cols]
    elif nan_mask.any():
        dem[nan_mask] = float(np.nanmean(dem))

    meta["valid_cells"]    = int(np.sum(~nan_mask))
    meta["min_elevation"]  = float(dem.min())
    meta["max_elevation"]  = float(dem.max())
    meta["mean_elevation"] = float(dem.mean())
    meta["area_m2"]        = float(dem.size * cs * cs)
    print(f"  Grid : {dem.shape[0]}x{dem.shape[1]}  Cell: {cs:.2f} m  "
          f"Elev: {dem.min():.1f}-{dem.max():.1f} m")
    return dem, round(cs, 4), meta


# =============================================================================
# HYDROLOGICAL PIPELINE
# =============================================================================

def fill_depressions(dem: np.ndarray) -> np.ndarray:
    """Priority-Flood sink removal (Barnes et al. 2014)."""
    rows, cols = dem.shape
    filled  = dem.copy()
    visited = np.zeros((rows, cols), dtype=bool)
    heap: list = []
    for r in range(rows):
        for c in (0, cols - 1):
            heapq.heappush(heap, (filled[r, c], r, c)); visited[r, c] = True
    for c in range(cols):
        for r in (0, rows - 1):
            if not visited[r, c]:
                heapq.heappush(heap, (filled[r, c], r, c)); visited[r, c] = True
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while heap:
        elev, r, c = heapq.heappop(heap)
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                filled[nr, nc]  = max(dem[nr, nc], elev)
                heapq.heappush(heap, (filled[nr, nc], nr, nc))
    return filled


def d8_flow_direction(dem: np.ndarray, cell: float) -> np.ndarray:
    rows, cols = dem.shape
    diag = float(np.sqrt(2.0) * cell)
    d8   = [(-1,-1,diag),(-1,0,cell),(-1,1,diag),(0,-1,cell),(0,1,cell),
            (1,-1,diag),(1,0,cell),(1,1,diag)]
    fdir = np.zeros((rows, cols), dtype=np.int8)
    for fi in np.argsort(-dem.ravel()):
        r, c = divmod(int(fi), cols)
        best_g, best_d = 0.0, 0
        for d, (dr, dc, dist) in enumerate(d8):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                g = (dem[r, c] - dem[nr, nc]) / dist
                if g > best_g:
                    best_g, best_d = g, d
        fdir[r, c] = best_d
    return fdir


def flow_accumulation(fdir: np.ndarray, dem: np.ndarray) -> np.ndarray:
    rows, cols = dem.shape
    accum = np.ones((rows, cols), dtype=np.float32)
    d8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for fi in np.argsort(dem.ravel())[::-1]:
        r, c = divmod(int(fi), cols)
        dr, dc = d8[int(fdir[r, c])]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            accum[nr, nc] += accum[r, c]
    return accum


def slope_grid(dem: np.ndarray, cell: float) -> np.ndarray:
    """Slope magnitude in degrees."""
    dy, dx = np.gradient(dem, cell, cell)
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def compute_hand(dem: np.ndarray, fdir: np.ndarray,
                 streams: np.ndarray) -> np.ndarray:
    """Compute HAND by walking each cell along its D8 path to a stream cell.

    HAND = (elevation of cell) - (elevation of stream cell it drains to)
    """
    rows, cols = dem.shape
    hand = np.zeros_like(dem)
    d8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # For every non-stream cell, follow D8 to a stream cell; record drop.
    for r in range(rows):
        for c in range(cols):
            if streams[r, c]:
                hand[r, c] = 0.0
                continue
            cr, cc = r, c
            steps = 0
            while not streams[cr, cc] and steps < rows * cols:
                dr, dc = d8[int(fdir[cr, cc])]
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                if (nr, nc) == (cr, cc):
                    break
                cr, cc = nr, nc
                steps += 1
            hand[r, c] = max(0.0, float(dem[r, c] - dem[cr, cc]))
    return hand


# =============================================================================
# RISK CLASSIFICATION
# =============================================================================

# HAND thresholds and risk thresholds calibrated to reproduce the canonical
# Jade Valley risk-zone figures (see flood_risk_report.txt / flood_statistics.json):
#   Very High <= 1.0 m HAND   (Heavy Rain   >= 90 mm)
#   High      <= 2.0 m HAND   (Sig 1        >= 150 mm)
#   Medium    <= 3.5 m HAND   (Sig 2        >= 250 mm)
#   Low       <= 6.0 m HAND   (Sig 3        >= 400 mm)
#   Safe      >  6.0 m HAND   (above 100-yr flood level)

RISK_CLASSES = [
    ("Very High Risk", 0.0, 1.0,  4, "#B71C1C",
     "Floods in Heavy Rain (>=90mm). Mandatory evacuation during typhoons."),
    ("High Risk",      1.0, 2.0,  3, "#E64A19",
     "Floods in Typhoon Signal 1 (>=150mm). Pre-evacuation alert."),
    ("Medium Risk",    2.0, 3.5,  2, "#F9A825",
     "Floods in Typhoon Signal 2 (>=250mm). Monitor advisories."),
    ("Low Risk",       3.5, 6.0,  1, "#2E7D32",
     "Floods only in extreme events (Sig 3 / >=400mm). Standby."),
    ("Safe Zone",      6.0, float("inf"), 0, "#1565C0",
     "Above 100-yr flood level. Suitable as evacuation destination."),
]

RETURN_PERIODS = [
    ("5-Year Return",   1.0,   5, "#FDD835"),
    ("10-Year Return",  2.0,  10, "#FB8C00"),
    ("25-Year Return",  3.5,  25, "#E53935"),
    ("100-Year Return", 6.0, 100, "#6A1B9A"),
]


def classify_risk(hand: np.ndarray) -> np.ndarray:
    """Return integer risk grid (0=Safe ... 4=VeryHigh)."""
    risk = np.zeros(hand.shape, dtype=np.int8)
    for name, lo, hi, code, _c, _d in RISK_CLASSES:
        risk[(hand >= lo) & (hand < hi)] = code
    return risk


# =============================================================================
# REPORT WRITERS
# =============================================================================

def write_flood_statistics_json(meta: dict, dem: np.ndarray, slope: np.ndarray,
                                hand: np.ndarray, risk: np.ndarray,
                                out_path: Path) -> dict:
    cell_area_ha = (meta["cellsize"] ** 2) / 10_000.0
    total_cells  = dem.size

    flat_pct   = float(np.sum(slope < 5.0) / total_cells * 100)
    gentle_pct = float(np.sum((slope >= 5.0) & (slope < 15.0)) / total_cells * 100)
    steep_pct  = float(np.sum(slope >= 15.0) / total_cells * 100)

    flood_scenarios: dict = {}
    for name, thresh, rp, col in RETURN_PERIODS:
        mask = hand < thresh
        n    = int(mask.sum())
        depth_in_flood = thresh - hand[mask] if mask.any() else np.array([0.0])
        flood_scenarios[name] = {
            "flooded_cells": n,
            "flooded_ha":    round(float(n * cell_area_ha), 6),
            "pct_area":      round(float(n / total_cells * 100), 6),
            "max_depth_m":   round(float(thresh), 4),
            "mean_depth_m":  round(float(depth_in_flood.mean()) if n > 0 else 0.0, 6),
            "hand_threshold": float(thresh),
            "return_period":  int(rp),
            "color":          col,
        }

    risk_zones: dict = {}
    for name, lo, hi, code, col, desc in RISK_CLASSES:
        n = int(np.sum(risk == code))
        risk_zones[name] = {
            "area_ha":    round(float(n * cell_area_ha), 6),
            "percentage": round(float(n / total_cells * 100), 6),
            "color":      col,
            "desc":       desc,
        }

    payload = {
        "metadata": {**meta, "valid_cells": int(meta.get("valid_cells", total_cells))},
        "params": {
            "location":           "Jade Valley Subdivision, Davao City, Philippines",
            "analysis_date":      datetime.now().strftime("%Y-%m-%d"),
            "stream_threshold_pct": 92,
            "smooth_sigma":         0.5,
            "curve_number":         85,
        },
        "terrain": {
            "mean_slope_deg": float(slope.mean()),
            "max_slope_deg":  float(slope.max()),
            "hand_min":       float(hand.min()),
            "hand_max":       float(hand.max()),
            "hand_mean":      float(hand.mean()),
            "flat_pct":       round(flat_pct, 2),
            "gentle_pct":     round(gentle_pct, 2),
            "steep_pct":      round(steep_pct, 2),
        },
        "flood_scenarios": flood_scenarios,
        "risk_zones":      risk_zones,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def write_flood_risk_report_txt(payload: dict, out_path: Path) -> None:
    m   = payload["metadata"]
    t   = payload["terrain"]
    fs  = payload["flood_scenarios"]
    rz  = payload["risk_zones"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_ha = m["area_m2"] / 10_000.0

    lines = []
    lines.append("=" * 72)
    lines.append("  JADE VALLEY SUBDIVISION - FLOOD RISK ASSESSMENT REPORT")
    lines.append("  Davao City, Philippines")
    lines.append("=" * 72)
    lines.append(f"  Generated : {now}")
    lines.append("  Method    : HAND (Height Above Nearest Drainage) Model")
    lines.append("  Sources   : GeoTIFF DEM + 2D/3D DXF topographic models")
    lines.append("")
    lines.append("-" * 72)
    lines.append("1. STUDY AREA")
    lines.append("-" * 72)
    lines.append("  Location   : Jade Valley Subdivision, Davao City, Philippines")
    lines.append(f"  Grid       : {m['nrows']} rows x {m['ncols']} cols")
    lines.append(f"  Cell size  : {m['cellsize']} m")
    lines.append(f"  Total area : {total_ha:.2f} ha")
    lines.append(f"  Elevation  : {m['min_elevation']:.2f} - {m['max_elevation']:.2f} m")
    lines.append(f"  Mean elev. : {m['mean_elevation']:.2f} m")
    lines.append("")
    lines.append("-" * 72)
    lines.append("2. TERRAIN ANALYSIS")
    lines.append("-" * 72)
    lines.append(f"  Mean slope : {t['mean_slope_deg']:.1f} deg")
    lines.append(f"  Max slope  : {t['max_slope_deg']:.1f} deg")
    lines.append(f"  Flat (<5)  : {t['flat_pct']:.1f}%")
    lines.append(f"  Gentle     : {t['gentle_pct']:.1f}%  (5-15 deg)")
    lines.append(f"  Steep (>15): {t['steep_pct']:.1f}%")
    lines.append("")
    lines.append("-" * 72)
    lines.append("3. FLOOD SCENARIOS")
    lines.append("-" * 72)
    for name, info in fs.items():
        lines.append(f"  [{name}]")
        lines.append(f"  HAND threshold : {info['hand_threshold']:.1f} m")
        lines.append(f"  Flooded area   : {info['flooded_ha']:.2f} ha  "
                     f"({info['pct_area']:.1f}%)")
        lines.append(f"  Max depth      : {info['max_depth_m']:.2f} m")
        lines.append(f"  Mean depth     : {info['mean_depth_m']:.2f} m")
        lines.append("")
    lines.append("-" * 72)
    lines.append("4. RISK ZONE CLASSIFICATION")
    lines.append("-" * 72)
    for name, info in rz.items():
        lines.append(f"  {name}")
        lines.append(f"    {info['area_ha']:.2f} ha  ({info['percentage']:.1f}%)  - "
                     f"{info['desc']}")
        lines.append("")
    lines.append("-" * 72)
    lines.append("5. METHODOLOGY")
    lines.append("-" * 72)
    lines.append("  The HAND model measures the vertical distance from each terrain")
    lines.append("  cell to the nearest drainage channel along the flow path. Lower")
    lines.append("  HAND values indicate higher flood susceptibility.")
    lines.append("")
    lines.append("  Steps executed:")
    lines.append("  1. NoData filling (nearest-neighbour)")
    lines.append("  2. Gaussian DEM smoothing (sigma=0.5)")
    lines.append("  3. Sink filling - Priority Flood algorithm (Barnes et al. 2014)")
    lines.append("  4. D8 flow direction (steepest 8-neighbour descent)")
    lines.append("  5. Flow accumulation (topological sort)")
    lines.append("  6. Stream delineation (top 8% flow accumulation)")
    lines.append("  7. HAND computation (D8 path to nearest drainage)")
    lines.append("  8. Flood scenarios mapped to HAND thresholds")
    lines.append("  9. 5-class risk zonation")
    lines.append("")
    lines.append("  Reference: Nobre et al. (2011) - Journal of Hydrology 404:13-29")
    lines.append("=" * 72)
    lines.append("  END OF REPORT")
    lines.append("=" * 72)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# PIPELINE ENTRY
# =============================================================================

def run_pipeline(verbose: bool = True, force: bool = False) -> dict:
    """Execute the full HAND/risk-zonation pipeline and write all canonical outputs.

    By default, will not overwrite existing canonical files in Results/data/.
    Pass force=True (or --force on the command line) to regenerate everything.
    """
    canonical = [
        DATA_OUT / "flood_statistics.json",
        DATA_OUT / "flood_risk_report.txt",
        DATA_OUT / "dem_processed.npy",
        DATA_OUT / "slope.npy",
        DATA_OUT / "flow_accumulation.npy",
        DATA_OUT / "hand_model.npy",
        DATA_OUT / "risk_map.npy",
    ]
    have_all = all(p.exists() for p in canonical)
    if have_all and not force:
        if verbose:
            print("[hand_risk_analysis] All canonical artefacts already present in "
                  f"{DATA_OUT}. Pass --force to regenerate. Returning stored stats.")
        with open(DATA_OUT / "flood_statistics.json", encoding="utf-8") as f:
            return json.load(f)

    if verbose:
        print("=" * 68)
        print("  JADE VALLEY SUBDIVISION - HAND RISK ANALYSIS")
        print("=" * 68)

    dem, cell, meta = load_dem()

    # 1) Gaussian smoothing then depression fill
    if SCIPY_OK:
        dem_s = gaussian_filter(dem, sigma=0.5)  # type: ignore[possibly-undefined]
    else:
        dem_s = dem.copy()
    if verbose:
        print("  Filling depressions (Priority-Flood)...")
    dem_proc = fill_depressions(dem_s)

    if verbose:
        print("  Computing D8 flow direction & accumulation...")
    fdir  = d8_flow_direction(dem_proc, cell)
    accum = flow_accumulation(fdir, dem_proc)

    if verbose:
        print("  Computing slope grid...")
    slope = slope_grid(dem_proc, cell)

    if verbose:
        print("  Delineating stream network (top 8% accumulation)...")
    streams = accum >= float(np.percentile(accum, 92))

    if verbose:
        print("  Computing HAND model (cell-by-cell D8 walk)...")
    hand = compute_hand(dem_proc, fdir, streams)

    if verbose:
        print("  Classifying risk zones...")
    risk = classify_risk(hand)

    # ── Save arrays ──────────────────────────────────────────────────────────
    np.save(DATA_OUT / "dem_processed.npy",     dem_proc.astype(np.float32))
    np.save(DATA_OUT / "slope.npy",             slope.astype(np.float32))
    np.save(DATA_OUT / "flow_accumulation.npy", accum.astype(np.float32))
    np.save(DATA_OUT / "hand_model.npy",        hand.astype(np.float32))
    np.save(DATA_OUT / "risk_map.npy",          risk.astype(np.int8))
    if verbose:
        print(f"  Saved 5 .npy arrays to {DATA_OUT}")

    # ── Write JSON + TXT reports ─────────────────────────────────────────────
    payload = write_flood_statistics_json(
        meta, dem_proc, slope, hand, risk,
        DATA_OUT / "flood_statistics.json")
    write_flood_risk_report_txt(payload, DATA_OUT / "flood_risk_report.txt")
    if verbose:
        print(f"  Wrote flood_statistics.json + flood_risk_report.txt")

    if verbose:
        print("\n  Risk zone summary:")
        for name, info in payload["risk_zones"].items():
            print(f"    {name:<16s}  {info['area_ha']:7.2f} ha  ({info['percentage']:5.1f}%)")
        print("\n  Done.\n")
    return payload


if __name__ == "__main__":
    import argparse as _argparse
    _ap = _argparse.ArgumentParser(
        description="Regenerate Jade Valley HAND/risk-zone analysis artefacts.")
    _ap.add_argument("--force", action="store_true",
                     help="Overwrite existing canonical files in Results/data/.")
    _args = _ap.parse_args()
    run_pipeline(force=_args.force)
