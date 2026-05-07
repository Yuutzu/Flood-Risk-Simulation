"""
=============================================================================
 JADE VALLEY SUBDIVISION — FLOOD SIMULATION WITH PREVENTION MEASURES  (75%)
 Davao City, Philippines
=============================================================================
 This build extends the 50 percent baseline with four configurable prevention
 measures that physically modify the DEM before the simulation runs.

 Prevention Measure 1 — Riverbank Floodwall: raises western river bank cells
 by a user-specified height, creating a physical DEM barrier that delays
 river overflow until the water surface exceeds the raised crest.

 Prevention Measure 2 — Drainage Canal Network: lowers a corridor of cells
 (east outlet plus south branch) to create open channels. Runoff naturally
 drains into the canals and away from the residential core.

 Prevention Measure 3 — Retention Basin: excavates a stormwater pond on
 undeveloped perimeter land (outer 20 % buffer, never inside the residential
 core). The basin intercepts runoff draining off the residential blocks,
 stores it during peak rainfall, and releases it slowly — reducing peak
 inundation depth and flooded area across the residential core.

 Prevention Measure 4 — Elevated Emergency Road: raises a cross-subdivision
 road corridor above expected flood level. Acts as a raised berm that guides
 flow to either side while keeping emergency vehicle access intact throughout
 the flood event.

 The animation renders seven overlapping layers: JPEG background, river
 channel band, rain accumulation depth, river overflow, and four prevention
 infrastructure overlays: red (floodwall), cyan (canal), green (basin),
 and orange (emergency road).

 The stats panel shows live flood numbers for both the improved and baseline
 (no prevention) runs so the effectiveness is visible frame by frame.

 Run:  python Main/flood_simulation_75%.py
=============================================================================
"""

import csv
import heapq
import io
import json
import os
import random
import sys
import tkinter as tk
import warnings
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import ttk

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Button, Slider

warnings.filterwarnings("ignore")

# ── Optional libraries ────────────────────────────────────────────────────────
try:
    import rasterio
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    from scipy.ndimage import binary_dilation, distance_transform_edt
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import ezdxf
    EZDXF_OK = True
except ImportError:
    EZDXF_OK = False

# =============================================================================
# PATHS  (same layout as flood_animation.py)
# =============================================================================

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "Map Topography"
TIF_FILE  = DATA_DIR / "3D" / "JVS_Simulation.tif"
DXF_2D    = DATA_DIR / "2D" / "Jade_Valley_Subdivision_2D_vectorial.dxf"
JPEG_2D   = DATA_DIR / "2D" / "JVS_2D.jpg"
ANIM_DIR  = BASE_DIR / "Results" / "animations"
DATA_OUT  = BASE_DIR / "Results" / "data"
MAPS_DIR  = BASE_DIR / "Results" / "maps"
for _d in (ANIM_DIR, DATA_OUT, MAPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STORM SCENARIOS  (identical to 25% — drop-in compatible)
# =============================================================================

SCENARIOS = {
    "1": {
        "name"       : "Light Rain",
        "rainfall_mm": 15,
        "duration_h" : 2.0,
        "pattern"    : "uniform",
        "desc"       : "Avg 7.5 mm/hr — surface puddles in low areas; drains within hours",
    },
    "2": {
        "name"       : "Moderate Rain",
        "rainfall_mm": 36,
        "duration_h" : 3.0,
        "pattern"    : "progressive",
        "desc"       : "Avg 12 mm/hr — minor street flooding; low zones collect water",
    },
    "3": {
        "name"       : "Heavy Rain",
        "rainfall_mm": 90,
        "duration_h" : 4.0,
        "pattern"    : "progressive",
        "desc"       : "Avg 22.5 mm/hr — widespread street flooding; monitor river levels",
    },
    "4": {
        "name"       : "Typhoon Signal 1 (Tropical Depression)",
        "rainfall_mm": 150,
        "duration_h" : 8.0,
        "pattern"    : "progressive",
        "desc"       : "Avg 18.75 mm/hr — river rises; some low areas flood; prepare",
    },
    "5": {
        "name"       : "Typhoon Signal 2 (Tropical Storm)",
        "rainfall_mm": 250,
        "duration_h" : 12.0,
        "pattern"    : "burst",
        "desc"       : "Avg 20.8 mm/hr — widespread flooding; voluntary evacuation",
    },
    "6": {
        "name"       : "Typhoon Signal 3 (Severe Typhoon)",
        "rainfall_mm": 400,
        "duration_h" : 18.0,
        "pattern"    : "burst",
        "desc"       : "Avg 22.2 mm/hr — catastrophic river overflow; mandatory evacuation",
    },
    "7": {
        "name"       : "Custom — I will enter my own values",
        "rainfall_mm": None,
        "duration_h" : None,
        "pattern"    : None,
        "desc"       : "",
    },
}

# =============================================================================
# DEM LOADING  (identical to 25%)
# =============================================================================

def load_dem() -> tuple:
    """Load JVS_Simulation.tif.  Returns (dem_array, cellsize_m)."""
    if not TIF_FILE.exists():
        sys.exit(f"\n[ERROR] DEM file not found:\n  {TIF_FILE}")
    if not RASTERIO_OK:
        sys.exit("\n[ERROR] rasterio required.  Run:  pip install rasterio")
    print(f"  Loading DEM: {TIF_FILE.name}")
    with rasterio.open(str(TIF_FILE)) as src:           # type: ignore
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
            cs = csx
    nan_mask = np.isnan(dem)
    if nan_mask.any() and SCIPY_OK:
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        if idx is not None:
            dem[nan_mask] = dem[np.asarray(idx[0], int)[nan_mask], np.asarray(idx[1], int)[nan_mask]]
        else:
            dem[nan_mask] = float(np.nanmean(dem))
    elif nan_mask.any():
        dem[nan_mask] = float(np.nanmean(dem))
    print(f"  Grid : {dem.shape[0]}×{dem.shape[1]}  |  Cell: {cs:.1f} m  |  "
          f"Elev: {dem.min():.1f}–{dem.max():.1f} m")
    return dem, round(cs, 2)

# =============================================================================
# HYDROLOGICAL HELPERS  (identical to 25%)
# =============================================================================

def _fill_depressions(dem: np.ndarray) -> np.ndarray:
    rows, cols = dem.shape
    filled  = dem.copy()
    visited = np.zeros((rows, cols), dtype=bool)
    heap: list = []
    for r in range(rows):
        for c in (0, cols - 1):
            if not visited[r, c]:
                heapq.heappush(heap, (filled[r, c], r, c))
                visited[r, c] = True
    for c in range(cols):
        for r in (0, rows - 1):
            if not visited[r, c]:
                heapq.heappush(heap, (filled[r, c], r, c))
                visited[r, c] = True
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


def _d8_flow_direction(dem, cell_w, cell_h):
    rows, cols = dem.shape
    diag = float(np.sqrt(cell_w**2 + cell_h**2))
    d8   = [(-1,-1,diag),(-1,0,cell_h),(-1,1,diag),(0,-1,cell_w),(0,1,cell_w),
            (1,-1,diag),(1,0,cell_h),(1,1,diag)]
    fdir = np.zeros((rows, cols), dtype=np.int8)
    for fi in np.argsort(-dem.ravel()):
        r, c = divmod(int(fi), cols)
        bg, bd = 0.0, 0
        for d, (dr, dc, dist) in enumerate(d8):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                g = (dem[r, c] - dem[nr, nc]) / dist
                if g > bg:
                    bg, bd = g, d
        fdir[r, c] = bd
    return fdir


def _flow_accumulation(fdir, dem):
    rows, cols = dem.shape
    accum = np.ones((rows, cols), dtype=np.float32)
    d8    = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for fi in np.argsort(dem.ravel())[::-1]:
        r, c = divmod(int(fi), cols)
        dr, dc = d8[int(fdir[r, c])]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            accum[nr, nc] += accum[r, c]
    return accum


def build_stream_mask(accum, pct=92.0):
    return accum >= float(np.percentile(accum, pct))

# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PREVENTION MEASURES  ← NEW in 50%                                     ║
# ║                                                                          ║
# ║  Two functions that return a MODIFIED DEM.  The simulation physics       ║
# ║  then runs on this modified DEM so water genuinely responds to the       ║
# ║  infrastructure — it is blocked by the wall, or drains faster through   ║
# ║  the canal — not just drawn on top as decoration.                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

def apply_floodwall(dem: np.ndarray, cellsize: float,
                    wall_height_m: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Prevention Measure 1 — Riverbank Floodwall
    ────────────────────────────────────────────
    Raises the elevation of cells that form the western river bank by
    wall_height_m.  This creates a physical barrier in the DEM: the flood
    routing engine cannot spill water over the bank until the river level
    exceeds the raised crest.

    Target cells:  Any cell with Z ≤ 6 m that lies within ~3 cell-widths
    of the river channel (top-8% flow accumulation cells).  This matches
    the geometry visible in the JVS terrain — the Davao River bank is the
    lowest-elevation strip on the western edge of the study area.

    Returns a copy of the DEM with the wall cells raised.
    """
    modified = dem.copy()
    rows, cols = dem.shape

    # Derive a quick flow-accumulation river mask (reuses DEM helpers)
    filled = _fill_depressions(dem)
    fdir   = _d8_flow_direction(filled, cellsize, cellsize)
    accum  = _flow_accumulation(fdir, filled)
    river  = accum >= float(np.percentile(accum, 92))

    # Dilate river mask by 3 cells to catch bank cells
    if SCIPY_OK:
        bank_zone = binary_dilation(river,   # type: ignore
                                    structure=np.ones((3, 3), bool),
                                    iterations=3)
    else:
        bank_zone = river.copy()
        for _ in range(3):
            shifted = (np.roll(bank_zone, 1, 0) | np.roll(bank_zone, -1, 0) |
                       np.roll(bank_zone, 1, 1) | np.roll(bank_zone, -1, 1))
            bank_zone |= shifted

    # Only raise cells that are: near the bank AND below 6 m elevation
    wall_cells = bank_zone & (dem <= 6.0).astype(bool)
    modified[wall_cells] = dem[wall_cells] + wall_height_m

    n = int(wall_cells.sum())
    length_m = n * cellsize
    print(f"  Floodwall: raised {n} cells  ≈ {length_m:.0f} m wall length  "
          f"(+{wall_height_m:.1f} m crest)")
    return modified, wall_cells   # also return mask for visualisation


def apply_drainage_canal(dem: np.ndarray, cellsize: float,
                         canal_depth_m: float = 2.0,
                         canal_width_cells: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Prevention Measure 2 — Drainage Canal Network
    ───────────────────────────────────────────────
    Creates two canal segments in the DEM by lowering cell elevations:

    Canal A — East outlet channel
        Runs eastward from the flood-prone core (DEM centroid area)
        to the eastern edge of the study area.  Width = canal_width_cells.
        This gives surface runoff a fast path OUT of the subdivision.

    Canal B — South branch
        A shorter north-to-south segment connecting the lowest-lying
        row to the south boundary.  Mimics a diversion ditch that
        channels water away from the residential blocks.

    The canal cells are lowered to  max(dem_cell - canal_depth_m, 0.3 m)
    so they are always below the surrounding terrain but never go negative.

    Returns (modified_dem, canal_mask) for simulation and visualisation.
    """
    modified   = dem.copy()
    rows, cols = dem.shape
    canal_mask = np.zeros((rows, cols), dtype=bool)

    # ── Canal A: east outlet  (runs along row ~40% from top) ──────────────
    # Find the row of the lowest mean elevation in the centre band
    mid_col_start = cols // 4
    mid_col_end   = 3 * cols // 4
    row_means = dem[:, mid_col_start:mid_col_end].mean(axis=1)
    canal_a_row = int(np.argmin(row_means))
    # Extend from 20% of width to eastern edge
    ca_col_start = max(0, cols // 5)
    for r in range(max(0, canal_a_row - canal_width_cells // 2),
                   min(rows, canal_a_row + canal_width_cells // 2 + 1)):
        for c in range(ca_col_start, cols):
            new_z = max(dem[r, c] - canal_depth_m, 0.3)
            if new_z < modified[r, c]:
                modified[r, c]  = new_z
                canal_mask[r, c] = True

    # ── Canal B: south branch  (runs downward from canal A row) ──────────
    # Locate the column of highest flow accumulation in the lower half
    lower_half    = dem[rows // 2:, :]
    peak_col      = int(np.unravel_index(np.argmax(lower_half), lower_half.shape)[1])
    canal_b_col   = peak_col
    cb_row_start  = canal_a_row
    for c in range(max(0, canal_b_col - canal_width_cells // 2),
                   min(cols, canal_b_col + canal_width_cells // 2 + 1)):
        for r in range(cb_row_start, rows):
            new_z = max(dem[r, c] - canal_depth_m, 0.3)
            if new_z < modified[r, c]:
                modified[r, c]  = new_z
                canal_mask[r, c] = True

    n = int(canal_mask.sum())
    print(f"  Drainage canals: modified {n} cells  "
          f"(Canal A: row {canal_a_row}, Canal B: col {canal_b_col})")
    return modified, canal_mask


def apply_retention_basin(dem: np.ndarray, cellsize: float,
                           basin_depth_m: float = 6.0,
                           basin_size_pct: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """
    Prevention Measure 3 — Retention Basin
    ────────────────────────────────────────
    Excavates a stormwater retention basin on undeveloped perimeter land,
    NEVER inside the residential core.  In real engineering practice, a
    retention basin requires land clearance before construction — placing
    one over existing houses is structurally and legally impossible.

    Placement strategy (three-step filter)
    ───────────────────────────────────────
    1.  River exclusion — cells in the top-8 % flow-accumulation band are
        the active river/stream channel.  These are fully excluded.
    2.  Residential core exclusion — the inner 60 % of the grid (rows and
        columns both between 20 % and 80 % of the grid extent) represents
        the occupied residential blocks of Jade Valley.  The basin must NOT
        be placed here.
    3.  Open perimeter zone — the remaining outer ~20 % perimeter band is
        treated as undeveloped buffer land or open lots that could realistically
        host a retention pond.  Candidates are scored by flow accumulation
        (higher = more runoff intercepted) and low elevation (higher storage
        efficiency), then the highest-scoring open cell becomes the basin centre.

    The basin footprint is clipped to the open zone so no residential cell is
    ever excavated, even if the circular patch would otherwise overlap it.

    Size : basin_size_pct of the grid area (default 6 %).
    Depth: excavated to max(dem_cell − basin_depth_m, 0.5 m).

    Returns (modified_dem, basin_mask) for simulation and visualisation.
    """
    modified   = dem.copy()
    rows, cols = dem.shape
    basin_mask = np.zeros((rows, cols), dtype=bool)

    # ── Step 1: Hydrological preprocessing ───────────────────────────────────
    filled = _fill_depressions(dem)
    fdir   = _d8_flow_direction(filled, cellsize, cellsize)
    accum  = _flow_accumulation(fdir, filled)
    river  = accum >= float(np.percentile(accum, 92))

    # ── Step 2: Exclude the residential core ──────────────────────────────────
    # Inner 60 % of the grid (both axes) = occupied residential area.
    # Basin candidates are restricted to the outer perimeter buffer only.
    row_lo = int(rows * 0.20)
    row_hi = int(rows * 0.80)
    col_lo = int(cols * 0.20)
    col_hi = int(cols * 0.80)
    residential_core = np.zeros((rows, cols), dtype=bool)
    residential_core[row_lo:row_hi, col_lo:col_hi] = True

    # ── Step 3: Open perimeter candidate zone ────────────────────────────────
    # Valid cells: outside the residential core AND not in the river channel.
    # This matches buffer/greenspace land on the subdivision perimeter where
    # a real stormwater pond would be engineered.
    open_zone = ~residential_core & ~river
    if not open_zone.any():
        open_zone = ~river   # failsafe: at least avoid the river

    # ── Step 4: Score candidates — highest accumulation + lowest elevation ────
    elev_norm  = (dem - dem.min()) / (dem.max() - dem.min() + 1e-10)
    accum_norm = np.log1p(accum) / (np.log1p(accum).max() + 1e-10)
    score = accum_norm - 0.6 * elev_norm
    score[~open_zone] = -np.inf

    center_flat = int(np.argmax(score.ravel()))
    cr, cc = divmod(center_flat, cols)

    # ── Step 5: Circular basin footprint clipped to open zone ────────────────
    total_cells  = rows * cols
    target_n     = max(4, int(total_cells * basin_size_pct))
    radius_cells = int(np.sqrt(target_n / np.pi)) + 1

    rr      = np.arange(rows)[:, None]
    cc_grid = np.arange(cols)[None, :]
    dist    = np.sqrt((rr - cr) ** 2 + (cc_grid - cc) ** 2)
    # Clip to open_zone — no residential cell is ever excavated
    basin_zone = (dist <= radius_cells) & open_zone

    # ── Step 6: Excavate ──────────────────────────────────────────────────────
    for r in range(rows):
        for c in range(cols):
            if basin_zone[r, c]:
                new_z = max(dem[r, c] - basin_depth_m, 0.5)
                if new_z < modified[r, c]:
                    modified[r, c] = new_z
                    basin_mask[r, c] = True

    n        = int(basin_mask.sum())
    area_ha  = n * cellsize ** 2 / 10_000
    vol_m3   = float(np.sum(dem[basin_mask] - modified[basin_mask]) * cellsize ** 2)
    print(f"  Retention basin: {n} cells  ≈ {area_ha:.2f} ha  "
          f"|  storage ≈ {vol_m3:.0f} m³  (centre r={cr}, c={cc})")
    return modified, basin_mask


def apply_elevated_road(dem: np.ndarray, cellsize: float,
                        road_height_m: float = 1.5,
                        road_width_cells: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Prevention Measure 4 — Elevated Emergency Road
    ────────────────────────────────────────────────
    Raises a cross-subdivision road corridor above the expected flood surface.
    The road runs roughly east-to-west through the middle third of the grid,
    perpendicular to the primary flood-flow direction (south-to-north from the
    river).

    Two effects in the DEM physics:
    • The raised road acts as a partial berm: surface runoff from the north
      (residential core) is deflected east or west rather than flowing freely
      south into already-flooded low ground.
    • Emergency vehicles can traverse the corridor even at design flood level
      because the road surface stays dry up to road_height_m above natural
      terrain.

    Road cells are raised by road_height_m (clamped so no cell exceeds
    a gentle slope gradient that avoids artificial ridges in the DEM).

    Returns (modified_dem, road_mask) for simulation and visualisation.
    """
    modified   = dem.copy()
    rows, cols = dem.shape
    road_mask  = np.zeros((rows, cols), dtype=bool)

    # Road centreline: horizontal band at ~45 % from the top
    road_row = int(rows * 0.45)

    # Quick river mask to avoid placing the road inside the channel
    filled = _fill_depressions(dem)
    fdir   = _d8_flow_direction(filled, cellsize, cellsize)
    accum  = _flow_accumulation(fdir, filled)
    river  = accum >= float(np.percentile(accum, 92))

    half_w = road_width_cells // 2
    for r in range(max(0, road_row - half_w),
                   min(rows, road_row + half_w + 1)):
        for c in range(0, cols):
            if river[r, c]:
                continue  # skip river cells
            raised = dem[r, c] + road_height_m
            if raised > modified[r, c]:
                modified[r, c] = raised
                road_mask[r, c] = True

    n        = int(road_mask.sum())
    length_m = n * cellsize
    print(f"  Elevated road  : {n} cells  ≈ {length_m:.0f} m road length  "
          f"(+{road_height_m:.1f} m above terrain, row {road_row})")
    return modified, road_mask


def apply_prevention_measures(dem: np.ndarray, cellsize: float,
                               use_floodwall: bool,
                               use_canal: bool,
                               use_basin: bool = False,
                               use_road: bool = False,
                               wall_height: float = 1.5,
                               canal_depth: float = 2.0,
                               basin_depth: float = 6.0,
                               road_height: float = 1.5) -> tuple:
    """
    Combine the requested prevention measures into a single modified DEM.
    Returns (modified_dem, wall_mask, canal_mask, basin_mask, road_mask).
    All five always returned so the animation layer code is simple even when
    a measure is off.
    """
    modified    = dem.copy()
    wall_mask   = np.zeros(dem.shape, dtype=bool)
    canal_mask  = np.zeros(dem.shape, dtype=bool)
    basin_mask  = np.zeros(dem.shape, dtype=bool)
    road_mask   = np.zeros(dem.shape, dtype=bool)

    # Apply measures in order: basin first (excavates), then canal (channels),
    # then floodwall (raises bank), then road (raises corridor).
    # This order matters: the canal benefits from the basin already being lower.
    if use_basin:
        modified, basin_mask = apply_retention_basin(modified, cellsize, basin_depth)
    if use_canal:
        modified, canal_mask = apply_drainage_canal(modified, cellsize, canal_depth)
    if use_floodwall:
        modified, wall_mask = apply_floodwall(modified, cellsize, wall_height)
    if use_road:
        modified, road_mask = apply_elevated_road(modified, cellsize, road_height)
    return modified, wall_mask, canal_mask, basin_mask, road_mask

# =============================================================================
# FLOOD SIMULATION ENGINE  (identical physics to 25% — now accepts modified DEM)
# =============================================================================

class FloodSimulation:
    """
    Two-layer terrain-aware flood simulation.
    Accepts any DEM — pass the modified (prevention) DEM for the improved run.
    """

    def __init__(self, dem: np.ndarray, cellsize: float,
                 soil_saturation_pct: float = 30.0,
                 drainage_capacity_mmhr: float = 5.0,
                 canal_mask: 'np.ndarray | None' = None,
                 wall_mask: 'np.ndarray | None' = None,
                 basin_mask: 'np.ndarray | None' = None,
                 road_mask: 'np.ndarray | None' = None,
                 rainfall_mm: float = 90.0):
        self.dem_raw = dem.copy()  # DEM passed in (already modified if prevention is enabled)
        self.dem     = _fill_depressions(dem)
        self.rows, self.cols = dem.shape
        self.cell    = cellsize

        self.rain_water           = np.zeros_like(dem)
        self.river_water          = np.zeros_like(dem)
        self.river_level          = self.dem.copy()
        self.rainfall_accumulated = 0.0
        self.drainage_capacity    = drainage_capacity_mmhr

        self.canal_mask  = canal_mask  if canal_mask  is not None else np.zeros_like(dem, dtype=bool)
        self.wall_mask   = wall_mask   if wall_mask   is not None else np.zeros_like(dem, dtype=bool)
        self.basin_mask  = basin_mask  if basin_mask  is not None else np.zeros_like(dem, dtype=bool)
        self.road_mask   = road_mask   if road_mask   is not None else np.zeros_like(dem, dtype=bool)
        # Always set rainfall_mm for scenario factor
        self.rainfall_mm = rainfall_mm

        init_sat = float(np.clip(soil_saturation_pct / 100.0, 0.0, 1.0))
        self.saturation = np.full_like(dem, init_sat)

        print("  Preprocessing terrain…")
        fdir = _d8_flow_direction(self.dem, cellsize, cellsize)
        self.fdir        = fdir
        self.accum       = _flow_accumulation(fdir, self.dem)
        self.streams     = build_stream_mask(self.accum)

        e = self.dem
        self.elev_norm   = (e - e.min()) / (e.max() - e.min() + 1e-10)
        fa_log           = np.log1p(self.accum)
        self.flow_weight = fa_log / (fa_log.max() + 1e-10)
        self.slope       = np.hypot(*np.gradient(self.dem, cellsize, cellsize))
        slope_n          = self.slope / (self.slope.max() + 1e-10)
        # Scenario-adaptive runoff and infiltration
        # More runoff and less infiltration for severe scenarios
        self.runoff_coeff = np.clip(0.18 + 0.18 * slope_n + 0.12 * self._scenario_factor(), 0.12, 0.85)
        self.max_inf      = 1.2 + 1.2 * self.elev_norm * (1.0 - init_sat) - 0.7 * self._scenario_factor()

        self.river_mask   = self.accum >= float(np.percentile(self.accum, 92))

        # Use the current DEM (with prevention) for bank elevations
        nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        self.bank_elev = np.full((self.rows, self.cols), np.inf)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.river_mask[r, c]:
                    for dr, dc in nbrs:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.rows and 0 <= nc < self.cols
                                and not self.river_mask[nr, nc]):
                            be = self.dem_raw[nr, nc]
                            if be < self.bank_elev[r, c]:
                                self.bank_elev[r, c] = be
        valid = self.river_mask & np.isfinite(self.bank_elev)
        if valid.any():
            self.bank_elev[~valid & self.river_mask] = self.dem_raw[~valid & self.river_mask]
            self.river_level[valid]  = self.bank_elev[valid]
            self.flood_base_wse      = float(np.nanpercentile(self.bank_elev[valid], 50))
        else:
            self.flood_base_wse = float(self.dem.min())
        self.river_level_init = self.river_level.copy()
        self.bfs_front        = self.river_mask.copy()
        if not np.allclose(self.dem, dem):
            pass  # DEM was modified by prevention measures (floodwall/canal)
        if SCIPY_OK:
            self.river_dist = distance_transform_edt(~self.river_mask)  # type: ignore
        else:
            self.river_dist = np.zeros_like(dem)
        # Canal drainage mask
        self.canal_drainage = np.zeros_like(dem, dtype=float)
        if self.canal_mask is not None and self.canal_mask.any():
            self.canal_drainage[self.canal_mask] = 10.0
        # Wall elevation mask
        self.wall_elev = np.zeros_like(dem, dtype=float)
        if self.wall_mask is not None and self.wall_mask.any():
            self.wall_elev[self.wall_mask] = self.dem[self.wall_mask]
        # Basin: track how much water the basin has stored (m³ equivalent depth)
        self.basin_storage   = 0.0        # cumulative water captured (m averaged over basin)
        self.basin_capacity  = 0.0        # total storage capacity (m)
        if self.basin_mask is not None and self.basin_mask.any():
            # Capacity = (original_elev - excavated_elev) averaged over basin cells
            orig_mean = float(np.mean(dem[self.basin_mask]))
            self.basin_capacity = max(orig_mean - float(np.mean(self.dem_raw[self.basin_mask])), 0.1)
        # Road elevation mask — used to block cross-road flow below crest
        self.road_elev = np.zeros_like(dem, dtype=float)
        if self.road_mask is not None and self.road_mask.any():
            self.road_elev[self.road_mask] = self.dem[self.road_mask]

    def _scenario_factor(self):
        # Returns 0 for light, up to 1.0 for typhoon
        # Use rainfall_mm as proxy for severity
        mm = getattr(self, 'rainfall_mm', 90)
        if mm <= 20: return 0.0
        if mm <= 40: return 0.2
        if mm <= 90: return 0.5
        if mm <= 150: return 0.8
        return 1.0

    @property
    def water_depth(self):
        return self.rain_water + self.river_water

    def add_rainfall(self, total_mm, wind_map=None):
        # Store rainfall_mm for scenario factor
        if not hasattr(self, 'rainfall_mm'):
            self.rainfall_mm = total_mm
        mod = self.runoff_coeff.copy()
        if wind_map is not None:
            mod *= wind_map
        self.rain_water           += (total_mm / 1000.0) * mod
        self.rainfall_accumulated += total_mm / 1000.0

    def apply_river_overflow(self, rate_mmhr, dt_h, intensity=1.0):
        # Scenario-adaptive overflow and blue cell shading, tightly coupled to DEM/prevention
        if not self.river_mask.any():
            return
        effective_rate = rate_mmhr * intensity
        rain_m   = (rate_mmhr / 1000.0) * dt_h * intensity
        accum_mm = self.rainfall_accumulated * 1000.0
        # Overflow threshold: lower for light, higher for typhoon
        if accum_mm < 30.0:
            return
        # Scenario-adaptive overflow ramp and blue shading
        if accum_mm < 36.0:
            # Light rain: almost no overflow
            ramp = np.clip((accum_mm - 20.0) / 16.0, 0.0, 1.0) * 0.15
            rise_mult = ramp * (2.0 + 4.0 * self.flow_weight[self.river_mask])
            hops = 1
        elif accum_mm < 60.0:
            # Moderate rain: reduced overflow
            ramp = np.clip((accum_mm - 36.0) / 24.0, 0.0, 1.0) * 0.18 + 0.10
            rise_mult = ramp * (1.5 + 2.5 * self.flow_weight[self.river_mask])
            hops = 1
        elif accum_mm < 120.0:
            # Heavy rain: more pronounced overflow
            ramp = np.clip((accum_mm - 60.0) / 60.0, 0.0, 1.0) * 0.7 + 0.5
            rise_mult = ramp * (3.5 + 8.0 * self.flow_weight[self.river_mask])
            hops = 2
        else:
            # Typhoon: severe overflow
            ramp = 1.0
            rise_mult = ramp * (4.0 + 10.0 * self.flow_weight[self.river_mask])
            hops = 4
        if effective_rate > 20:
            rise_mult *= 1.2
        # If floodwall is present, require overtopping for overflow
        if self.wall_mask is not None and self.wall_mask.any():
            # Only allow overflow if river level exceeds wall crest by >2cm
            crest = self.wall_elev[self.river_mask]
            river_wse = self.river_level[self.river_mask]
            overtopped = river_wse > crest + 0.02
            rise_mult = rise_mult * overtopped.astype(float)
        self.river_level[self.river_mask] += rain_m * rise_mult
        rise = self.river_level[self.river_mask] - self.river_level_init[self.river_mask]
        flood_rise = float(np.percentile(rise, 90))
        if flood_rise <= 0:
            return
        # Cap water surface elevation by scenario severity
        if accum_mm < 36:
            wse_cap = 0.25
        elif accum_mm < 60:
            wse_cap = 0.6
        elif accum_mm < 120:
            wse_cap = 1.5
        else:
            wse_cap = 3.0
        eff_wse = self.flood_base_wse + min(flood_rise, wse_cap)
        # BFS dilation — more hops for severe scenarios, less for moderate
        can_flood = (self.dem_raw < eff_wse) & ~self.river_mask
        if self.wall_mask is not None and self.wall_mask.any():
            can_flood = can_flood & ~self.wall_mask
        struct = np.ones((3, 3), dtype=bool)
        for _ in range(hops):
            if SCIPY_OK:
                exp = binary_dilation(self.bfs_front, structure=struct)  # type: ignore
                new = exp.astype(bool) & can_flood.astype(bool) & ~self.bfs_front.astype(bool)
            else:
                f  = self.bfs_front
                sh = (np.roll(f,1,0)|np.roll(f,-1,0)|
                      np.roll(f,1,1)|np.roll(f,-1,1)|
                      np.roll(np.roll(f,1,0),1,1)|
                      np.roll(np.roll(f,1,0),-1,1)|
                      np.roll(np.roll(f,-1,0),1,1)|
                      np.roll(np.roll(f,-1,0),-1,1))
                new = sh & can_flood & ~f
            if not new.any():
                break
            self.bfs_front |= new
        land = self.bfs_front & ~self.river_mask
        # Blue cell shading: scenario-unique color and intensity
        blue_depth = np.clip(eff_wse - self.dem_raw, 0.0, wse_cap)
        blue_depth[self.river_mask] = 0.0
        # Canal: reduces inundation in canal cells
        if self.canal_mask is not None and self.canal_mask.any():
            blue_depth[self.canal_mask] *= 0.4
        # Basin: captures overflow — reduce blue shading in/near basin and
        # track storage; if basin is full, overflow continues normally
        if self.basin_mask is not None and self.basin_mask.any():
            basin_intake = float(np.mean(blue_depth[self.basin_mask]))
            remaining_cap = max(self.basin_capacity - self.basin_storage, 0.0)
            captured = min(basin_intake, remaining_cap)
            self.basin_storage = min(self.basin_storage + captured, self.basin_capacity)
            fill_ratio = self.basin_storage / max(self.basin_capacity, 0.01)
            # Reduce water in basin area proportional to available capacity
            blue_depth[self.basin_mask] *= fill_ratio  # more water shows as basin fills
            # Spill onto BFS front is reduced while basin has headroom
            if fill_ratio < 0.95:
                rise_mult *= max(0.25, fill_ratio)
        # Road: blocks overflow BFS front from crossing road unless overtopped
        if self.road_mask is not None and self.road_mask.any():
            # Suppress BFS front from spreading across road cells below crest
            road_wse = eff_wse
            road_blocked = self.road_mask & (road_wse < self.road_elev + 0.05)
            self.bfs_front[road_blocked] = False
            land = self.bfs_front & ~self.river_mask  # recalculate after road block
        delta = (blue_depth[land] - self.river_water[land]) * (0.25 + 0.18 * ramp)
        self.river_water[land] = np.maximum(self.river_water[land] + delta, 0.0)
        draining = ~self.bfs_front & ~self.river_mask & (self.river_water > 0)
        self.river_water[draining] *= (0.80 + 0.15 * (1.0 - ramp))
        np.clip(self.river_water, 0.0, wse_cap, out=self.river_water)

    def apply_infiltration(self, dt_h):
        sat_inc = np.minimum(0.12 * dt_h, 1.0 - self.saturation)
        sat_inc *= (0.5 + 0.5 * (1.0 - self.elev_norm))
        self.saturation = np.clip(self.saturation + sat_inc, 0.0, 1.0)
        inf_m = (self.max_inf * (1.0 - self.saturation) / 1000.0) * dt_h
        self.rain_water  = np.maximum(self.rain_water  - inf_m,       0.0)
        self.river_water = np.maximum(self.river_water - inf_m * 0.5, 0.0)

    def route_water(self, iters=12):
        d8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for _ in range(iters):
            wse = self.dem + self.rain_water
            dw  = np.zeros_like(self.rain_water)
            for dr, dc in d8:
                r0 = max(0,-dr); r1 = self.rows - max(0,dr)
                c0 = max(0,-dc); c1 = self.cols - max(0,dc)
                nr0, nr1 = r0+dr, r1+dr
                nc0, nc1 = c0+dc, c1+dc
                diff  = wse[r0:r1,c0:c1] - wse[nr0:nr1,nc0:nc1]
                sn    = self.slope[r0:r1,c0:c1] / (self.slope.max() + 1e-10)
                tf    = np.clip(0.03 + 0.22 * sn, 0.03, 0.35)
                flow  = np.clip(diff * tf, 0, self.rain_water[r0:r1,c0:c1] * 0.20)
                # Block flow across floodwall unless overtopped
                if self.wall_mask is not None and self.wall_mask.any():
                    src_wall = self.wall_mask[r0:r1, c0:c1]
                    dst_wall = self.wall_mask[nr0:nr1, nc0:nc1]
                    src_elev = self.wall_elev[r0:r1, c0:c1]
                    dst_elev = self.wall_elev[nr0:nr1, nc0:nc1]
                    block    = (src_wall != dst_wall)
                    src_wse  = wse[r0:r1, c0:c1]
                    dst_wse  = wse[nr0:nr1, nc0:nc1]
                    crest    = np.maximum(src_elev, dst_elev)
                    allow    = (src_wse > crest + 0.02) | (dst_wse > crest + 0.02)
                    flow[block & ~allow] = 0.0
                # Block flow across elevated road unless overtopped
                if self.road_mask is not None and self.road_mask.any():
                    src_road = self.road_mask[r0:r1, c0:c1]
                    dst_road = self.road_mask[nr0:nr1, nc0:nc1]
                    src_relev = self.road_elev[r0:r1, c0:c1]
                    dst_relev = self.road_elev[nr0:nr1, nc0:nc1]
                    rblock   = (src_road != dst_road)
                    src_wse  = wse[r0:r1, c0:c1]
                    dst_wse  = wse[nr0:nr1, nc0:nc1]
                    rcrest   = np.maximum(src_relev, dst_relev)
                    rallow   = (src_wse > rcrest + 0.05) | (dst_wse > rcrest + 0.05)
                    flow[rblock & ~rallow] = 0.0
                dw[r0:r1,c0:c1]     -= flow
                dw[nr0:nr1,nc0:nc1] += flow
            self.rain_water = np.maximum(self.rain_water + dw, 0.0)
        np.clip(self.rain_water, 0.0, 5.0, out=self.rain_water)

    def apply_drainage(self, dt_h):
        scenario_factor = self._scenario_factor()
        ch_rate    = 80.0 * self.flow_weight ** 2
        base_drain = self.drainage_capacity * (1.0 + 2.0 * self.elev_norm)
        base_drain = base_drain * (1.0 - 0.45 * scenario_factor)
        rate = base_drain + ch_rate
        if self.canal_mask is not None and self.canal_mask.any():
            rate = rate + self.canal_drainage * 300.0  # 30× faster in canal
        drain_m = (rate / 1000.0) * dt_h
        self.rain_water  = np.maximum(self.rain_water  - drain_m,       0.0)
        self.river_water = np.maximum(self.river_water - drain_m * 0.4, 0.0)
        # Canal cells: rapid drying
        if self.canal_mask is not None and self.canal_mask.any():
            self.rain_water[self.canal_mask]  *= 0.30
            self.river_water[self.canal_mask] *= 0.30
        # Basin cells: aggressively capture surface water (stored in basin)
        # Water drains from surroundings into basin ~5× faster than canal.
        # Release slowly once storm passes (basin_storage decays each step).
        if self.basin_mask is not None and self.basin_mask.any():
            fill_ratio = self.basin_storage / max(self.basin_capacity, 0.01)
            # Intake: drain surrounding rain_water toward basin
            if fill_ratio < 1.0:
                intake_factor = (1.0 - fill_ratio) * 0.65
                self.rain_water[self.basin_mask]  *= max(0.05, 1.0 - intake_factor)
                self.river_water[self.basin_mask] *= max(0.05, 1.0 - intake_factor * 0.7)
            else:
                # Basin full: behaves like a pond, no additional intake
                self.rain_water[self.basin_mask]  *= 0.85
                self.river_water[self.basin_mask] *= 0.85
            # Slow release: basin storage gradually decreases after peak
            self.basin_storage = max(0.0, self.basin_storage - 0.012 * dt_h)
        # Road cells: rain drains off road surface quickly (steep effective slope)
        if self.road_mask is not None and self.road_mask.any():
            self.rain_water[self.road_mask]  *= 0.25
            self.river_water[self.road_mask] *= 0.20
        self.rain_water [self.rain_water  < 0.003] *= 0.60
        self.river_water[self.river_water < 0.004] *= 0.65

    def step(self, rate_mmhr, dt_h, intensity=1.0, wind_map=None):
        rain_mm = rate_mmhr * dt_h * intensity
        self.add_rainfall(rain_mm, wind_map)
        self.route_water(iters=12)
        self.apply_river_overflow(rate_mmhr, dt_h, intensity)
        self.apply_infiltration(dt_h)
        self.apply_drainage(dt_h)
        # Hard sanitize — clamp and remove any NaN/inf that leaked through physics
        np.clip(self.rain_water,  0.0, 10.0, out=self.rain_water)
        np.clip(self.river_water, 0.0, 10.0, out=self.river_water)
        np.nan_to_num(self.rain_water,  copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.river_water, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

# =============================================================================
# WIND MAP  (identical to 25%)
# =============================================================================

def wind_rainfall_map(dem, speed_kmh, direction_deg):
    if speed_kmh < 1.0:
        return np.ones_like(dem, dtype=float)
    wr    = np.radians(direction_deg)
    wx, wy = np.sin(wr), -np.cos(wr)
    dy, dx = np.gradient(dem)
    ww    = wx * dx + wy * dy
    wn    = ww / (np.abs(ww).max() + 1e-10)
    inten = min(speed_kmh / 60.0, 1.0) * 0.35
    return np.clip(1.0 + inten * wn, 0.40, 1.80).astype(float)

# =============================================================================
# MAP BACKGROUND  (identical to 25%)
# =============================================================================

def render_jpeg_background(dem):
    if JPEG_2D.exists() and PIL_OK:
        try:
            from PIL import Image as PILImage
            pil = PILImage.open(str(JPEG_2D)).convert("RGB")
            # Cap to 2048 px on longest side for performance, as in 25%
            max_side = max(pil.size)
            if max_side > 2048:
                scale = 2048 / max_side
                new_w = int(pil.size[0] * scale)
                new_h = int(pil.size[1] * scale)
                # Handle PILImage.Resampling for older Pillow versions
                if hasattr(PILImage, "Resampling"):
                    resample = PILImage.Resampling.LANCZOS
                else:
                    resample = getattr(PILImage, "LANCZOS", 1)
                pil = pil.resize((new_w, new_h), resample)
            img = np.array(pil, dtype=np.float32) / 255.0
            print(f"  JPEG background loaded at {img.shape[1]}×{img.shape[0]} px")
            return img
        except Exception as e:
            print(f"  JPEG load failed ({e})")
    return _elev_fallback(dem)


def _elev_fallback(dem):
    dem_n  = (dem - dem.min()) / (dem.max() - dem.min() + 1e-10)
    shade  = 0.80 + 0.20 * dem_n
    return np.stack([shade, shade, shade], axis=2).astype(np.float32)

# =============================================================================
# COLORMAPS  (same low-opacity scheme as 25%)
# =============================================================================

def _rain_cmap():
    colors = [
        (0.00, 0.00, 0.00, 0.00),
        (0.40, 0.95, 0.95, 0.30),
        (0.00, 0.78, 0.78, 0.42),
        (0.50, 0.90, 0.20, 0.55),
        (1.00, 1.00, 0.00, 0.60),
        (1.00, 0.55, 0.00, 0.65),
        (1.00, 0.20, 0.00, 0.70),
        (0.80, 0.00, 0.00, 0.70),
    ]
    cmap = LinearSegmentedColormap.from_list('rain', colors)
    cmap.set_bad(alpha=0)   # render NaN cells as fully transparent
    return cmap


def _river_cmap():
    # EXACTLY match the 25% simulation blue colormap
    cmap = LinearSegmentedColormap.from_list(
        'river', [
            (0.00, 0.00, 0.00, 0.00),
            (0.12, 0.38, 0.82, 0.13),
            (0.12, 0.38, 0.82, 0.28),
            (0.12, 0.38, 0.82, 0.55),
            (0.12, 0.38, 0.82, 0.85),
            (0.00, 0.00, 0.40, 1.00)
        ], N=256)
    cmap.set_bad(alpha=0)   # render NaN cells as fully transparent
    return cmap

# =============================================================================
# INTENSITY PATTERN  (identical to 25%)
# =============================================================================

def _intensity_factor(frame: int, total_frames: int, pattern: str) -> float:
    """
    Per-timestep rainfall intensity multiplier. All patterns integrate to
    approximately 1.0 over the full storm so the total rainfall matches the
    scenario value.

    uniform     : constant 1.0 — steady rain at the stated rate.
    progressive : 0.40 to 1.40 — ramps up as the storm develops.
    burst       : 0.30 to 1.80 — Gaussian peak centred at 45 percent of duration.
    decreasing  : 1.60 to 0.20 — heavy convective start that tapers off.
    """
    t = frame / max(total_frames - 1, 1)
    if pattern == 'progressive':
        return 0.40 + 1.00 * min(t / 0.75, 1.0)
    if pattern == 'burst':
        return 0.30 + 1.50 * float(np.exp(-((t - 0.45) ** 2) / 0.055))
    if pattern == 'decreasing':
        return max(1.60 - 1.40 * t, 0.20)
    return 1.0

# =============================================================================
#  RESULTS ANALYSIS MODULE  (drop-in addition for flood_simulation_75%.py)
# -----------------------------------------------------------------------------
#  Adds three academic-grade upgrades to the 75% build:
#    A. Quantitative Results Table  — peak flood metrics, baseline vs. prevention
#    B. Sensitivity Analysis        — sweeps a single parameter to show effect
#    C. Validation Hook             — compares simulation extent to a reference
#                                     event (e.g. Typhoon Pablo 2012)
# =============================================================================


# -----------------------------------------------------------------------------
# A.  QUANTITATIVE RESULTS TABLE
# -----------------------------------------------------------------------------
def compute_quantitative_results(scenario_name, rainfall_mm, stats, base_stats,
                                 prevention_str, num_frames, timestep_min,
                                 out_dir):
    """
    Compute peak flood metrics and write a quantitative results table.

    Produces:
      Results/data/quantitative_results_<scenario>.txt   (human-readable)
      Results/data/quantitative_results_<scenario>.json  (machine-readable)
    """
    CELL_HA = 30.64 ** 2 / 10_000.0
    GRID_HA = 326.44

    dt_h = timestep_min / 60.0

    def _peak(series):
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            return 0.0, 0, 0.0
        idx = int(np.argmax(arr))
        return float(arr[idx]), idx, idx * timestep_min

    def _hh_integral(flooded_pct_series):
        arr = np.asarray(flooded_pct_series, dtype=float)
        area_ha = (arr / 100.0) * GRID_HA
        return float(np.sum(area_ha) * dt_h)

    peak_flood_pct, peak_flood_fr, peak_flood_min = _peak(stats["flooded_pct"])
    peak_depth_mm,  peak_depth_fr, peak_depth_min = _peak(stats["max_depth_mm"])
    peak_river_pct, _, _ = _peak(stats["river_pct"])
    peak_flood_ha = peak_flood_pct / 100.0 * GRID_HA
    flood_hh = _hh_integral(stats["flooded_pct"])

    rows = {
        "scenario_name": scenario_name,
        "rainfall_mm": rainfall_mm,
        "duration_min": num_frames * timestep_min,
        "prevention": prevention_str,
        "prevention_run": {
            "peak_flooded_pct":        round(peak_flood_pct, 2),
            "peak_flooded_ha":         round(peak_flood_ha, 2),
            "peak_max_depth_mm":       round(peak_depth_mm, 1),
            "peak_river_overflow_pct": round(peak_river_pct, 2),
            "time_to_peak_flood_min":  peak_flood_min,
            "time_to_peak_depth_min":  peak_depth_min,
            "flood_exposure_ha_hours": round(flood_hh, 2),
        },
    }

    if base_stats is not None and len(base_stats.get("flooded_pct", [])) > 0:
        b_peak_flood_pct, _, b_peak_flood_min = _peak(base_stats["flooded_pct"])
        b_peak_depth_mm,  _, b_peak_depth_min = _peak(base_stats["max_depth_mm"])
        b_peak_flood_ha = b_peak_flood_pct / 100.0 * GRID_HA
        b_flood_hh = _hh_integral(base_stats["flooded_pct"])

        d_flood_pct  = b_peak_flood_pct - peak_flood_pct
        d_flood_ha   = b_peak_flood_ha  - peak_flood_ha
        d_depth_mm   = b_peak_depth_mm  - peak_depth_mm
        d_time_peak  = peak_flood_min   - b_peak_flood_min
        d_flood_hh   = b_flood_hh       - flood_hh

        d_flood_pct_rel = (d_flood_pct / b_peak_flood_pct * 100.0
                           if b_peak_flood_pct > 0 else 0.0)
        d_depth_rel     = (d_depth_mm / b_peak_depth_mm * 100.0
                           if b_peak_depth_mm > 0 else 0.0)

        rows["baseline_run"] = {
            "peak_flooded_pct":        round(b_peak_flood_pct, 2),
            "peak_flooded_ha":         round(b_peak_flood_ha, 2),
            "peak_max_depth_mm":       round(b_peak_depth_mm, 1),
            "time_to_peak_flood_min":  b_peak_flood_min,
            "flood_exposure_ha_hours": round(b_flood_hh, 2),
        }
        rows["improvement"] = {
            "extent_reduction_ha":         round(d_flood_ha, 2),
            "extent_reduction_pct":        round(d_flood_pct, 2),
            "extent_reduction_rel_pct":    round(d_flood_pct_rel, 1),
            "depth_reduction_mm":          round(d_depth_mm, 1),
            "depth_reduction_rel_pct":     round(d_depth_rel, 1),
            "time_to_peak_shift_min":      d_time_peak,
            "exposure_reduction_ha_hours": round(d_flood_hh, 2),
        }

    safe = scenario_name.replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    json_path = out_dir / f"quantitative_results_{safe}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, cls=_NumpyEncoder)

    txt_path = out_dir / f"quantitative_results_{safe}.txt"
    lines = []
    lines.append("=" * 76)
    lines.append("  QUANTITATIVE RESULTS — JADE VALLEY FLOOD SIMULATION")
    lines.append("=" * 76)
    lines.append(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Scenario   : {scenario_name}")
    lines.append(f"  Rainfall   : {rainfall_mm:.0f} mm over {num_frames * timestep_min} min")
    lines.append(f"  Prevention : {prevention_str}")
    lines.append("")
    lines.append("-" * 76)
    lines.append("  Peak flood metrics  (run with prevention applied)")
    lines.append("-" * 76)
    pr = rows["prevention_run"]
    lines.append(f"  Peak flood extent       : {pr['peak_flooded_ha']:>8.2f} ha   "
                 f"({pr['peak_flooded_pct']:.2f} %)")
    lines.append(f"  Peak maximum depth      : {pr['peak_max_depth_mm']:>8.1f} mm")
    lines.append(f"  Peak river overflow     : {pr['peak_river_overflow_pct']:>8.2f} %")
    lines.append(f"  Time to peak extent     : {pr['time_to_peak_flood_min']:>8.0f} min")
    lines.append(f"  Time to peak depth      : {pr['time_to_peak_depth_min']:>8.0f} min")
    lines.append(f"  Flood exposure integral : {pr['flood_exposure_ha_hours']:>8.2f} ha-hours")

    if "baseline_run" in rows:
        b = rows["baseline_run"]
        d = rows["improvement"]
        lines.append("")
        lines.append("-" * 76)
        lines.append("  Baseline metrics  (same scenario, no prevention)")
        lines.append("-" * 76)
        lines.append(f"  Peak flood extent       : {b['peak_flooded_ha']:>8.2f} ha   "
                     f"({b['peak_flooded_pct']:.2f} %)")
        lines.append(f"  Peak maximum depth      : {b['peak_max_depth_mm']:>8.1f} mm")
        lines.append(f"  Time to peak extent     : {b['time_to_peak_flood_min']:>8.0f} min")
        lines.append(f"  Flood exposure integral : {b['flood_exposure_ha_hours']:>8.2f} ha-hours")
        lines.append("")
        lines.append("-" * 76)
        lines.append("  Improvement from prevention measures")
        lines.append("-" * 76)
        lines.append(f"  Extent reduction        : {d['extent_reduction_ha']:>8.2f} ha   "
                     f"({d['extent_reduction_rel_pct']:+.1f} % relative)")
        lines.append(f"  Depth reduction         : {d['depth_reduction_mm']:>8.1f} mm  "
                     f"({d['depth_reduction_rel_pct']:+.1f} % relative)")
        lines.append(f"  Time-to-peak shift      : {d['time_to_peak_shift_min']:>+8.0f} min  "
                     f"(positive = peak delayed)")
        lines.append(f"  Exposure reduction      : {d['exposure_reduction_ha_hours']:>8.2f} ha-hours")

    lines.append("")
    lines.append("=" * 76)
    lines.append(f"  Files written:")
    lines.append(f"    {txt_path}")
    lines.append(f"    {json_path}")
    lines.append("=" * 76)

    txt_content = "\n".join(lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    print()
    print(txt_content)
    return rows


# -----------------------------------------------------------------------------
# B.  SENSITIVITY ANALYSIS
# -----------------------------------------------------------------------------
def run_sensitivity_analysis(parameter, values, base_kwargs, out_dir,
                             run_simulation_fn):
    """
    Run the simulation N times, varying one parameter, and tabulate how
    the output responds.

    Parameters
    ----------
    parameter : str
        Name of the run_simulation kwarg to sweep (e.g. 'soil_sat_pct').
    values : list[float]
        The values to test for that parameter.
    base_kwargs : dict
        All other run_simulation kwargs held constant.
    out_dir : Path
        Directory to write the results CSV (typically DATA_OUT).
    run_simulation_fn : callable
        The run_simulation function from this script.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"sensitivity_{parameter}.csv"
    sens_rows = []

    print()
    print("=" * 76)
    print(f"  SENSITIVITY ANALYSIS — varying {parameter}")
    print(f"  Values      : {values}")
    print(f"  Other params: {base_kwargs}")
    print("=" * 76)

    for v in values:
        kwargs = dict(base_kwargs)
        kwargs[parameter] = v
        print(f"\n  >>> {parameter} = {v}")
        result = run_simulation_fn(**kwargs)
        if isinstance(result, dict) and "flooded_pct" in result:
            peak_flooded  = max(result["flooded_pct"])
            peak_depth    = max(result["max_depth_mm"])
            peak_flood_ha = peak_flooded / 100.0 * 326.44
        else:
            peak_flooded = peak_depth = peak_flood_ha = float("nan")

        sens_rows.append({
            parameter:           v,
            "peak_flooded_pct":  round(peak_flooded, 2),
            "peak_flooded_ha":   round(peak_flood_ha, 2),
            "peak_depth_mm":     round(peak_depth, 1),
        })

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens_rows[0].keys()))
        w.writeheader()
        w.writerows(sens_rows)

    arr_in  = np.array([r[parameter]          for r in sens_rows], dtype=float)
    arr_pct = np.array([r["peak_flooded_pct"] for r in sens_rows], dtype=float)
    arr_dep = np.array([r["peak_depth_mm"]    for r in sens_rows], dtype=float)

    in_range          = arr_in.max() - arr_in.min() if arr_in.size > 1 else 1.0
    flood_sensitivity = (arr_pct.max() - arr_pct.min()) / in_range if in_range else 0.0
    depth_sensitivity = (arr_dep.max() - arr_dep.min()) / in_range if in_range else 0.0

    print()
    print("-" * 76)
    print(f"  Sensitivity summary")
    print("-" * 76)
    print(f"  Output range (peak flooded %)  : {arr_pct.min():.2f} → {arr_pct.max():.2f}")
    print(f"  Output range (peak depth mm)   : {arr_dep.min():.1f} → {arr_dep.max():.1f}")
    print(f"  Flood-pct sensitivity (Δ%/Δ{parameter}): {flood_sensitivity:.4f}")
    print(f"  Depth-mm  sensitivity (Δmm/Δ{parameter}): {depth_sensitivity:.4f}")
    print(f"  CSV saved to: {csv_path}")
    print("=" * 76)

    return sens_rows


# -----------------------------------------------------------------------------
# C.  VALIDATION HOOK — Compare to a reference event
# -----------------------------------------------------------------------------
REFERENCE_EVENTS = {
    "Typhoon Pablo 2012": {
        "rainfall_mm": 192,
        "duration_h": 18.0,
        "pattern": "burst",
        "observed_flooded_pct": None,
        "observed_max_depth_mm": None,
        "notes": "Severe Typhoon (Bopha). Catastrophic for Mindanao; estimated 200 mm "
                 "in 24 h over Davao Region. Use this for high-end validation.",
    },
    "Typhoon Odette 2021": {
        "rainfall_mm": 130,
        "duration_h": 12.0,
        "pattern": "burst",
        "observed_flooded_pct": None,
        "observed_max_depth_mm": None,
        "notes": "Typhoon Rai/Odette. Extensive flooding reported across Davao. "
                 "Substitute observed values from local barangay reports.",
    },
    "Habagat Heavy Rain Episode": {
        "rainfall_mm": 90,
        "duration_h": 4.0,
        "pattern": "progressive",
        "observed_flooded_pct": None,
        "observed_max_depth_mm": None,
        "notes": "Use this slot for any documented heavy rain event with good photos.",
    },
}


def validate_against_event(event_name, simulated_peak_flooded_pct,
                           simulated_peak_depth_mm, out_dir):
    """
    Compare the simulation's peak metrics against an observed reference event.
    Writes a validation report to Results/data/validation_<event>.txt.

    To use:
      1. Run the simulation with the rainfall/duration of the chosen event.
      2. Capture peak_flooded_pct and peak_max_depth_mm from the run.
      3. Call this function with those numbers.
    """
    if event_name not in REFERENCE_EVENTS:
        raise ValueError(f"Unknown event: {event_name}. "
                         f"Available: {list(REFERENCE_EVENTS.keys())}")
    ref = REFERENCE_EVENTS[event_name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    errors = {}
    rating = "Not rated (no observed data on file)"
    if ref["observed_flooded_pct"] is not None:
        err_pct = abs(simulated_peak_flooded_pct - ref["observed_flooded_pct"])
        errors["flooded_pct_abs_error"] = round(err_pct, 2)
        if err_pct < 5:    rating = "Strong agreement (< 5 % absolute)"
        elif err_pct < 10: rating = "Acceptable agreement (< 10 %)"
        elif err_pct < 20: rating = "Marginal agreement (< 20 %)"
        else:              rating = "Poor agreement (≥ 20 %)"

    if ref["observed_max_depth_mm"] is not None:
        err_dep = abs(simulated_peak_depth_mm - ref["observed_max_depth_mm"])
        errors["max_depth_mm_abs_error"] = round(err_dep, 1)

    safe = event_name.replace(' ', '_').replace('/', '-')
    txt_path = out_dir / f"validation_{safe}.txt"
    lines = []
    lines.append("=" * 76)
    lines.append(f"  VALIDATION REPORT — {event_name}")
    lines.append("=" * 76)
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Notes     : {ref['notes']}")
    lines.append("")
    lines.append("  Reference event parameters:")
    lines.append(f"    Rainfall : {ref['rainfall_mm']} mm")
    lines.append(f"    Duration : {ref['duration_h']} h")
    lines.append(f"    Pattern  : {ref['pattern']}")
    lines.append("")
    lines.append("  Observed (from reports / photos):")
    lines.append(f"    Peak flooded % : "
                 f"{ref['observed_flooded_pct'] if ref['observed_flooded_pct'] is not None else 'NOT YET RECORDED'}")
    lines.append(f"    Peak depth mm  : "
                 f"{ref['observed_max_depth_mm'] if ref['observed_max_depth_mm'] is not None else 'NOT YET RECORDED'}")
    lines.append("")
    lines.append("  Simulated:")
    lines.append(f"    Peak flooded % : {simulated_peak_flooded_pct:.2f}")
    lines.append(f"    Peak depth mm  : {simulated_peak_depth_mm:.1f}")
    lines.append("")
    if errors:
        lines.append("  Absolute errors:")
        for k, v in errors.items():
            lines.append(f"    {k:<30s}: {v}")
    lines.append("")
    lines.append(f"  Agreement rating: {rating}")
    lines.append("=" * 76)

    txt_content = "\n".join(lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print()
    print(txt_content)
    print(f"\n  Validation report saved: {txt_path}")
    return {"event": event_name, "errors": errors, "rating": rating}


# =============================================================================
# MAIN SIMULATION RUNNER
# =============================================================================
# Accepts prevention flags (use_floodwall, use_canal) and dimension parameters
# (wall_height, canal_depth) from the GUI. When prevention is active it runs
# a second baseline simulation on the original DEM so the stats panel can
# display improvement deltas frame by frame.

def run_simulation(dem: np.ndarray, cellsize: float,
                   rainfall_mm: float, duration_h: float,
                   timestep_min: int, start_time_str: str,
                   wind_speed: float, wind_dir: float,
                   soil_sat_pct: float, drain_cap: float,
                   pattern: str, scenario_name: str,
                   use_floodwall: bool = False,
                   use_canal: bool = False,
                   use_basin: bool = False,
                   use_road: bool = False,
                   wall_height: float = 1.5,
                   canal_depth: float = 2.0,
                   basin_depth: float = 6.0,
                   road_height: float = 1.5):

    any_prevention = use_floodwall or use_canal or use_basin or use_road

    # ── Apply prevention measures to DEM ─────────────────────────────────────
    dem_orig = dem.copy()
    if any_prevention:
        print("\n  Applying prevention measures to DEM…")
        sim_dem, wall_mask, canal_mask, basin_mask, road_mask = apply_prevention_measures(
            dem_orig, cellsize,
            use_floodwall, use_canal, use_basin, use_road,
            wall_height, canal_depth, basin_depth, road_height)
        prevention_label = []
        if use_floodwall:
            prevention_label.append(f"Floodwall +{wall_height:.1f}m")
        if use_canal:
            prevention_label.append("Drainage Canal")
        if use_basin:
            prevention_label.append(f"Retention Basin {basin_depth:.1f}m")
        if use_road:
            prevention_label.append(f"Elevated Road +{road_height:.1f}m")
        prevention_str = " + ".join(prevention_label)
    else:
        sim_dem     = dem_orig.copy()
        wall_mask   = np.zeros(dem_orig.shape, dtype=bool)
        canal_mask  = np.zeros(dem_orig.shape, dtype=bool)
        basin_mask  = np.zeros(dem_orig.shape, dtype=bool)
        road_mask   = np.zeros(dem_orig.shape, dtype=bool)
        prevention_str = "None (Baseline)"

    rate_mmhr  = rainfall_mm / duration_h
    dt_h       = timestep_min / 60.0
    num_frames = int(np.ceil(duration_h / dt_h))

    print(f"\n  Scenario    : {scenario_name}")
    print(f"  Rainfall    : {rainfall_mm:.0f} mm in {duration_h:.1f} h  ({rate_mmhr:.1f} mm/hr)")
    print(f"  Timestep    : {timestep_min} min  →  {num_frames} frames")
    print(f"  Prevention  : {prevention_str}")

    wmap = wind_rainfall_map(dem, wind_speed, wind_dir)

    # ── Run IMPROVED simulation (on modified DEM) ─────────────────────────----
    print("\n  Running improved simulation…")
    sim = FloodSimulation(sim_dem, cellsize,
                          soil_saturation_pct=soil_sat_pct,
                          drainage_capacity_mmhr=drain_cap,
                          canal_mask=canal_mask,
                          wall_mask=wall_mask,
                          basin_mask=basin_mask,
                          road_mask=road_mask,
                          rainfall_mm=rainfall_mm)
    rain_frames  = []
    river_frames = []
    times_list   = []
    stats = {"rain_mm": [], "flooded_pct": [], "river_pct": [],
             "max_depth_mm": [], "max_river_mm": []}

    sh, sm = map(int, start_time_str.split(':'))
    cur = datetime.now().replace(hour=sh, minute=sm, second=0)
    for fr in range(num_frames):
        inten = _intensity_factor(fr, num_frames, pattern)
        sim.step(rate_mmhr, dt_h, intensity=inten, wind_map=wmap)
        rain_frames .append(np.nan_to_num(sim.rain_water .copy(), nan=0.0, posinf=0.0, neginf=0.0))
        river_frames.append(np.nan_to_num(sim.river_water.copy(), nan=0.0, posinf=0.0, neginf=0.0))
        times_list  .append(cur.strftime("%H:%M"))
        total = sim.rain_water + sim.river_water
        stats["rain_mm"]     .append(float(sim.rainfall_accumulated * 1000))
        stats["flooded_pct"] .append(float(np.sum(total > 0.01) / total.size * 100))
        stats["river_pct"]   .append(float(np.sum(sim.river_water > 0.005) / total.size * 100))
        stats["max_depth_mm"].append(float(total.max() * 1000))
        stats["max_river_mm"].append(float(sim.river_water.max() * 1000))
        if (fr + 1) % max(1, num_frames // 8) == 0 or fr == 0:
            print(f"    [{fr+1:3d}/{num_frames}]  {times_list[-1]}  "
                  f"rain={stats['rain_mm'][-1]:.0f} mm  "
                  f"flooded={stats['flooded_pct'][-1]:.1f}%")
        cur += timedelta(minutes=timestep_min)

    # ── Run BASELINE simulation (for comparison stats) ─────────────────------
    if any_prevention:
        print("\n  Running baseline (no prevention) for comparison…")
        sim_base = FloodSimulation(dem_orig, cellsize,
                                   soil_saturation_pct=soil_sat_pct,
                                   drainage_capacity_mmhr=drain_cap,
                                   rainfall_mm=rainfall_mm)
        base_stats = {"flooded_pct": [], "max_depth_mm": []}
        for fr in range(num_frames):
            inten = _intensity_factor(fr, num_frames, pattern)
            sim_base.step(rate_mmhr, dt_h, intensity=inten, wind_map=wmap)
            total_b = sim_base.rain_water + sim_base.river_water
            base_stats["flooded_pct"].append(
                float(np.sum(total_b > 0.01) / total_b.size * 100))
            base_stats["max_depth_mm"].append(float(total_b.max() * 1000))
    else:
        base_stats = None

    # ── Quantitative results table ────────────────────────────────────────────
    compute_quantitative_results(
        scenario_name=scenario_name,
        rainfall_mm=rainfall_mm,
        stats=stats,
        base_stats=base_stats,
        prevention_str=prevention_str,
        num_frames=num_frames,
        timestep_min=timestep_min,
        out_dir=DATA_OUT,
    )

    # ── Build infrastructure overlay arrays (for drawing on the map) ─────────
    H, W = dem.shape
    ext  = (0, W, H, 0)

    # Floodwall overlay — lower opacity for better map visibility
    wall_rgba = np.zeros((H, W, 4), dtype=np.float32)
    if wall_mask.any():
        wall_rgba[wall_mask, 0] = 0.90
        wall_rgba[wall_mask, 1] = 0.15
        wall_rgba[wall_mask, 2] = 0.10
        wall_rgba[wall_mask, 3] = 0.32  # Lowered from 0.65 for less blocking

    # Canal overlay
    canal_rgba = np.zeros((H, W, 4), dtype=np.float32)
    if canal_mask.any():
        canal_rgba[canal_mask, 0] = 0.00
        canal_rgba[canal_mask, 1] = 0.88
        canal_rgba[canal_mask, 2] = 0.95
        canal_rgba[canal_mask, 3] = 0.28

    # Retention Basin overlay — dark green with medium opacity
    basin_rgba = np.zeros((H, W, 4), dtype=np.float32)
    if basin_mask.any():
        basin_rgba[basin_mask, 0] = 0.05
        basin_rgba[basin_mask, 1] = 0.72
        basin_rgba[basin_mask, 2] = 0.18
        basin_rgba[basin_mask, 3] = 0.38

    # Elevated Road overlay — amber/orange
    road_rgba = np.zeros((H, W, 4), dtype=np.float32)
    if road_mask.any():
        road_rgba[road_mask, 0] = 1.00
        road_rgba[road_mask, 1] = 0.60
        road_rgba[road_mask, 2] = 0.00
        road_rgba[road_mask, 3] = 0.40

    # ── Background + colormaps ──────────────────────────────────────────────
    print("\n  Rendering map background…")
    bg         = render_jpeg_background(dem)
    cmap_rain  = _rain_cmap()
    cmap_river = _river_cmap()

    # ── Figure layout ───────────────────────────────────────────────────────
    DARK  = '#0D1117'
    PANEL = '#161B22'
    TCLR  = '#E6EDF3'
    ACC   = '#4FC3F7'

    title_suffix = f"  |  Prevention: {prevention_str}" if any_prevention else ""

    # Determine figure size from screen dimensions at runtime.
    import matplotlib
    backend   = matplotlib.get_backend()
    screen_w, screen_h = 1600, 900
    fig_w, fig_h       = 1300, 800
    dpi = 100
    try:
        import tkinter as _tk
        _r = _tk.Tk(); _r.withdraw()
        screen_w = _r.winfo_screenwidth()
        screen_h = _r.winfo_screenheight()
        _r.destroy()
        fig_w = min(int(screen_w * 0.85), 1500)
        fig_h = min(int(screen_h * 0.85), 900)
    except Exception:
        pass
    figsize = (fig_w / dpi, fig_h / dpi)
    fig = plt.figure(figsize=figsize, facecolor=DARK, dpi=dpi)
    fig.suptitle(
        f"{scenario_name}{title_suffix}",
        fontsize=12, fontweight='bold', color=TCLR, y=0.979)
    # Move and resize the window if possible (TkAgg backend)
    try:
        if backend == 'TkAgg':
            mgr = plt.get_current_fig_manager()
            # Only set geometry if window attribute exists (TkAgg backend)
            if hasattr(mgr, 'window'):
                window = getattr(mgr, 'window', None)
                if window is not None and hasattr(window, 'wm_geometry'):
                    x = (screen_w - fig_w) // 2
                    y = (screen_h - fig_h) // 2
                    try:
                        window.wm_geometry(f"{fig_w}x{fig_h}+{x}+{y}")  # type: ignore
                    except Exception:
                        pass
                # If 'window' attribute does not exist, skip geometry setting
    except Exception:
        pass

    ax_map   = fig.add_axes((0.03, 0.18, 0.62, 0.76))
    ax_stats = fig.add_axes((0.69, 0.18, 0.29, 0.76))
    ax_map  .set_facecolor('black')
    ax_stats.set_facecolor(PANEL)
    ax_stats.axis('off')

    # ── Map layers ──────────────────────────────────────────────────────────
    # Layer 1: JPEG background — use 'bilinear' for smooth, photographic look (as in 25%)
    ax_map.imshow(bg, extent=ext, aspect='auto',
                  zorder=1, interpolation='bilinear')
    # Layer 2: Rain accumulation (updated each frame)
    im_rain  = ax_map.imshow(np.nan_to_num(rain_frames[0], nan=0.0, posinf=0.0, neginf=0.0) * 1000,
                              cmap=cmap_rain, vmin=0, vmax=600,
                              extent=ext, aspect='auto',
                              zorder=2, interpolation='bilinear',
                              alpha=0.45)
    # Blue cell shading: EXACTLY match 25% (total water depth, nonlinear colormap)
    total_water = np.nan_to_num(rain_frames[0], nan=0.0, posinf=0.0, neginf=0.0) + np.nan_to_num(river_frames[0], nan=0.0, posinf=0.0, neginf=0.0)
    im_river = ax_map.imshow(
        np.clip(np.power(total_water / 1.1, 0.65), 0, 1),
        cmap=cmap_river, vmin=0, vmax=1,
        extent=ext, aspect='auto', zorder=3,
        interpolation='nearest', alpha=0.62)

    # Layer 4: Floodwall overlay (static)
    if wall_mask.any():
        ax_map.imshow(wall_rgba, extent=ext, aspect='auto', zorder=5,
                      interpolation='nearest')

    # Layer 5: Drainage canal overlay (static)
    if canal_mask.any():
        ax_map.imshow(canal_rgba, extent=ext, aspect='auto', zorder=5,
                      interpolation='nearest')

    # Layer 6a: Retention basin overlay (static)
    if basin_mask.any():
        ax_map.imshow(basin_rgba, extent=ext, aspect='auto', zorder=5,
                      interpolation='nearest')

    # Layer 6b: Elevated road overlay (static)
    if road_mask.any():
        ax_map.imshow(road_rgba, extent=ext, aspect='auto', zorder=5,
                      interpolation='nearest')

    # Layer 6: Stream-network overlay (sky-blue tint on drainage channels)
    strm_rgba = np.zeros((H, W, 4), dtype=np.float32)
    strm_rgba[sim.streams, 0] = 0.18   # R
    strm_rgba[sim.streams, 1] = 0.72   # G  ← was 0 (missing!) — caused purple artefacts
    strm_rgba[sim.streams, 2] = 0.95   # B
    strm_rgba[sim.streams, 3] = 0.45   # A  (slightly reduced to keep map readable)
    ax_map.imshow(strm_rgba, extent=ext, aspect='auto', zorder=6,
                  interpolation='nearest')

    # Elevation contour lines (faint topographic reference)
    try:
        ax_map.contour(np.flipud(dem),
                       levels=np.linspace(dem.min(), dem.max(), 14),
                       colors="white", alpha=0.10, linewidths=0.4, zorder=4)
    except Exception:
        pass

    ax_map.set_xlim(0, W); ax_map.set_ylim(H, 0)
    ax_map.tick_params(colors=TCLR, labelsize=8)
    for sp in ax_map.spines.values():
        sp.set_edgecolor('#30363D')

    # Scale bar (~50 m physical length)
    sc_cells = max(3, int(round(50.0 / cellsize)))
    sc_m     = sc_cells * cellsize
    bx0, bx1 = W * 0.05, W * 0.05 + sc_cells
    by, bh   = H * 0.930, H * 0.007
    ax_map.fill_between([bx0, bx1], [by - bh] * 2, [by + bh] * 2,
                        color="white", zorder=15)
    ax_map.text((bx0 + bx1) / 2, by + bh * 3.0, f"{sc_m:.0f} m",
                color="white", fontsize=7, ha="center", va="bottom",
                fontweight="bold", zorder=15)

    # North arrow
    nx, ny0, ny1 = W * 0.938, H * 0.115, H * 0.060
    ax_map.annotate("", xy=(nx, ny1), xytext=(nx, ny0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2.0), zorder=15)
    ax_map.text(nx, ny1 - H * 0.014, "N", color="white",
                fontsize=9, ha="center", va="bottom", fontweight="bold", zorder=15)

    # Colorbar
    cbar = fig.colorbar(im_rain, ax=ax_map, orientation='vertical',
                        pad=0.01, shrink=0.80)
    cbar.set_label("Rain Water Depth (mm)", color=TCLR, fontsize=9)
    cbar.set_ticks([0, 30, 100, 200, 400, 600])
    cbar.ax.set_yticklabels(['Dry', '30', '100', '200', '400', '600+'],
                             color=TCLR, fontsize=8)
    cbar.ax.tick_params(colors=TCLR)

    # Map legend — water layers + prevention overlays
    legend_handles = [
        Patch(facecolor="#29B6F6", alpha=0.80, label="Stream channel"),
        Patch(facecolor="#1E90FF", alpha=0.75, label="River overflow"),
        Patch(facecolor="#00FFCC", alpha=0.55, label="Runoff ≤ 30 mm"),
        Patch(facecolor="#FFFF00", alpha=0.60, label="Runoff ≤ 100 mm"),
        Patch(facecolor="#FF6600", alpha=0.65, label="Runoff ≤ 300 mm"),
        Patch(facecolor="#CC4400", alpha=0.70, label="Runoff > 300 mm"),
    ]
    if wall_mask.any():
        legend_handles.append(Patch(facecolor=(0.9, 0.15, 0.1, 0.32), edgecolor='r',
                                    label=f'Floodwall (+{wall_height:.1f} m)'))
    if canal_mask.any():
        legend_handles.append(Patch(facecolor=(0, 0.88, 0.95, 0.28), edgecolor='c',
                                    label='Drainage Canal'))
    if basin_mask.any():
        legend_handles.append(Patch(facecolor=(0.05, 0.72, 0.18, 0.38), edgecolor='#3FB950',
                                    label=f'Retention Basin ({basin_depth:.1f}m deep)'))
    if road_mask.any():
        legend_handles.append(Patch(facecolor=(1.0, 0.60, 0.0, 0.40), edgecolor='#FF9800',
                                    label=f'Elevated Road (+{road_height:.1f} m)'))
    ax_map.legend(handles=legend_handles, loc='lower right',
                  facecolor="#0D1117", edgecolor=ACC,
                  labelcolor="#E6EDF3", fontsize=6.5,
                  framealpha=0.88, handlelength=1.2, borderpad=0.7)
    # Time badge
    time_txt = ax_map.text(
        0.015, 0.975, "", transform=ax_map.transAxes,
        fontsize=12, fontweight='bold', color='white', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL,
                  alpha=0.90, edgecolor=ACC, linewidth=1.5))

    # Stats text (effectiveness comparison shown live in draw() via improve_lines)
    # "Jade Valley Simulation" header label in the stats panel
    ax_stats.text(
        0.50, 0.997,
        "JADE VALLEY\nSUBDIVISION",
        fontsize=11, fontweight='bold', color=ACC,
        ha='center', va='top', transform=ax_stats.transAxes,
        family='monospace')
    ax_stats.text(
        0.50, 0.955,
        "FLOOD  SIMULATION",
        fontsize=8, fontweight='bold', color='#B0BEC5',
        ha='center', va='top', transform=ax_stats.transAxes,
        family='monospace')

    stats_txt = ax_stats.text(
        0.05, 0.91, "", fontsize=10, family='monospace',
        color=TCLR, va='top', transform=ax_stats.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#0D1117',
                  edgecolor=ACC, alpha=0.92, linewidth=1.2))

    # ── Widgets ──────────────────────────────────────────────────────────────
    ax_sl_frame = fig.add_axes((0.03, 0.098, 0.62, 0.024), facecolor='#21262D')
    ax_sl_speed = fig.add_axes((0.03, 0.048, 0.26, 0.024), facecolor='#21262D')
    ax_btn_play = fig.add_axes((0.335, 0.028, 0.09, 0.055))
    ax_btn_prev = fig.add_axes((0.430, 0.028, 0.055, 0.055))
    ax_btn_next = fig.add_axes((0.490, 0.028, 0.055, 0.055))
    ax_btn_gif  = fig.add_axes((0.555, 0.028, 0.10,  0.055))

    sl_frame = Slider(ax_sl_frame, 'Frame', 0, num_frames - 1,
                      valinit=0, valstep=1, color=ACC)
    sl_speed = Slider(ax_sl_speed, 'Speed ×', 0.25, 4.0,
                      valinit=1.0, color='#FFB74D')
    for sl in (sl_frame, sl_speed):
        sl.label.set_color(TCLR); sl.valtext.set_color(TCLR)
        sl.label.set_fontsize(8)

    btn_play = Button(ax_btn_play, 'Pause',    color='#1B5E20', hovercolor='#2E7D32')
    btn_prev = Button(ax_btn_prev, '◀◀', color='#0D47A1', hovercolor='#1565C0')
    btn_next = Button(ax_btn_next, '▶▶', color='#0D47A1', hovercolor='#1565C0')
    btn_gif  = Button(ax_btn_gif,  'Save GIF', color='#4A148C', hovercolor='#6A1B9A')
    for b in (btn_play, btn_prev, btn_next, btn_gif):
        b.label.set_color('white'); b.label.set_fontsize(10)

    player = {'playing': True, 'frame': 0}
    DIRS = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW']
    compass = DIRS[int((wind_dir + 11.25) / 22.5) % 16]
    wind_str = f"{wind_speed:.0f} km/h {compass}" if wind_speed >= 1 else "None"

    def _risk_str(max_depth_mm, river_pct):
        # Thresholds: 50 mm ankle-deep, 150 mm shin, 300 mm knee, 600 mm waist height.
        if river_pct > 30 or max_depth_mm > 600:
            return "EVACUATE NOW",               '#FF1744'
        if river_pct > 18 or max_depth_mm > 300:
            return "MANDATORY EVACUATION",       '#FF6D00'
        if river_pct > 8  or max_depth_mm > 150:
            return "PRE-EVACUATION ALERT",       '#FFD740'
        if river_pct > 3  or max_depth_mm > 50:
            return "STANDBY — prepare to move",  '#69F0AE'
        return     "NORMAL — monitoring",         '#B0BEC5'

    def draw(fi):
        fi = int(fi) % num_frames
        player['frame'] = fi
        rn = np.nan_to_num(rain_frames[fi], nan=0.0, posinf=0.0, neginf=0.0) * 1000
        im_rain .set_data(rn)
        im_rain .set_clim(0, min(max(float(rn.max()), 30) * 1.3, 600))
        # Blue cell shading: EXACTLY match 25% (total water depth, nonlinear colormap)
        total_water = np.nan_to_num(rain_frames[fi], nan=0.0, posinf=0.0, neginf=0.0) + np.nan_to_num(river_frames[fi], nan=0.0, posinf=0.0, neginf=0.0)
        im_river.set_data(np.clip(np.power(total_water / 1.1, 0.65), 0, 1))
        time_txt.set_text(
            f" Time: {times_list[fi]}\n"
            f" Rain: {stats['rain_mm'][fi]:.0f}/{rainfall_mm:.0f} mm\n"
            f" Wind: {wind_str}")

        # ── Stats panel: empty when no prevention; full detail when active ──
        if not any_prevention:
            # User has no prevention measures — show nothing
            stats_txt.set_text("")
            patch = stats_txt.get_bbox_patch()
            if patch is not None:
                patch.set_edgecolor(ACC)
        else:
            # ── Prevention is active — build rich stats panel ──────────────
            rsk, rsk_col = _risk_str(stats['max_depth_mm'][fi], stats['river_pct'][fi])

            # Basin fill level (live, from simulation object)
            basin_fill_pct = 0.0
            if use_basin and sim.basin_capacity > 0:
                basin_fill_pct = min(sim.basin_storage / sim.basin_capacity * 100, 100)

            # Prevention methods with sizes
            measure_lines = ""
            if use_floodwall:
                measure_lines += f"  • Floodwall        +{wall_height:.1f} m crest\n"
            if use_canal:
                measure_lines += f"  • Drainage Canal   -{canal_depth:.1f} m depth\n"
            if use_basin:
                measure_lines += f"  • Retention Basin  -{basin_depth:.1f} m depth  [{basin_fill_pct:.0f}% full]\n"
            if use_road:
                measure_lines += f"  • Elevated Road    +{road_height:.1f} m above terrain\n"

            # Improvement delta vs baseline
            if base_stats and fi < len(base_stats["flooded_pct"]):
                b_pct   = base_stats["flooded_pct"][fi]
                b_depth = base_stats["max_depth_mm"][fi]
                imp_pct  = b_pct   - stats["flooded_pct"][fi]
                imp_dep  = b_depth - stats["max_depth_mm"][fi]
                eff_rating = ("Excellent" if imp_pct > 15 else
                              "Good"      if imp_pct > 7  else
                              "Moderate"  if imp_pct > 2  else "Minimal")
                improve_lines = (
                    f"  vs BASELINE (No Prevention)\n"
                    f"  {'─'*28}\n"
                    f"  Baseline flooded : {b_pct:.1f}%\n"
                    f"  With prevention  : {stats['flooded_pct'][fi]:.1f}%\n"
                    f"  Area saved       : {imp_pct:+.1f}%\n"
                    f"  Depth saved      : {imp_dep:+.0f} mm\n"
                    f"  Effectiveness    : {eff_rating}\n"
                )
            else:
                improve_lines = ""

            stats_txt.set_text(
                f"  PREVENTION MEASURES ACTIVE\n"
                f"  {'─'*28}\n"
                f"{measure_lines}"
                f"\n"
                f"  SCENARIO\n"
                f"  {'─'*28}\n"
                f"  {scenario_name}\n"
                f"  Rainfall : {rainfall_mm:.0f} mm / {duration_h:.1f} hr\n"
                f"  Rate     : {rate_mmhr:.1f} mm/hr ({pattern})\n"
                f"\n"
                f"  LIVE STATUS  [{times_list[fi]}]\n"
                f"  {'─'*28}\n"
                f"  Elapsed  : {fi * timestep_min} min\n"
                f"  Fallen   : {stats['rain_mm'][fi]:.1f} mm\n"
                f"  Max depth: {stats['max_depth_mm'][fi]:.0f} mm\n"
                f"  Flooded  : {stats['flooded_pct'][fi]:.1f}%\n"
                f"  River area: {stats['river_pct'][fi]:.1f}%\n"
                f"\n"
                f"  RISK LEVEL:  {rsk}\n"
                f"\n"
                f"{improve_lines}"
            )
            patch = stats_txt.get_bbox_patch()
            if patch is not None:
                patch.set_edgecolor(rsk_col)

        if abs(sl_frame.val - fi) > 0.5:
            sl_frame.eventson = False
            sl_frame.set_val(fi)
            sl_frame.eventson = True
        fig.canvas.draw_idle()

    BASE_INTERVAL = 600

    def _anim_step(_):
        if player['playing']:
            draw(player['frame'] + 1)
        return []

    anim_obj = animation.FuncAnimation(
        fig, _anim_step, interval=BASE_INTERVAL,
        blit=False, cache_frame_data=False)

    def on_frame(val):
        draw(int(val))

    def on_speed(val):
        anim_obj.event_source.interval = max(50, int(BASE_INTERVAL / max(val, 0.01)))

    def on_play_pause(_):
        player['playing'] = not player['playing']
        if player['playing']:
            btn_play.label.set_text('Pause')
            btn_play.ax.set_facecolor('#1B5E20')
        else:
            btn_play.label.set_text('Play')
            btn_play.ax.set_facecolor('#BF360C')
        fig.canvas.draw_idle()

    def on_prev(_):
        player['playing'] = False
        btn_play.label.set_text('Play')
        btn_play.ax.set_facecolor('#BF360C')
        draw(player['frame'] - 1)

    def on_next(_):
        player['playing'] = False
        btn_play.label.set_text('Play')
        btn_play.ax.set_facecolor('#BF360C')
        draw(player['frame'] + 1)

    def on_save_gif(_):
        safe = (scenario_name.replace(' ', '_').replace('/', '-')
                .replace('(', '').replace(')', ''))
        prev_tag = "_prevention" if any_prevention else ""
        out = str(ANIM_DIR / f"flood_{safe}{prev_tag}.gif")
        print(f"\n  Saving GIF: {out}…")
        was = player['playing']
        player['playing'] = False
        if not PIL_OK:
            print("  [ERROR] pip install Pillow")
            player['playing'] = was
            return
        tmp_fig, (tm, ts) = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK)
        tm.set_facecolor('black'); ts.set_facecolor(PANEL); ts.axis('off')
        tm.imshow(bg, extent=ext, aspect='auto', zorder=1, interpolation='nearest')
        if wall_mask.any():
            tm.imshow(wall_rgba,  extent=ext, aspect='auto', zorder=5, interpolation='nearest')
        if canal_mask.any():
            tm.imshow(canal_rgba, extent=ext, aspect='auto', zorder=5, interpolation='nearest')
        if basin_mask.any():
            tm.imshow(basin_rgba, extent=ext, aspect='auto', zorder=5, interpolation='nearest')
        if road_mask.any():
            tm.imshow(road_rgba,  extent=ext, aspect='auto', zorder=5, interpolation='nearest')
        _irn = tm.imshow(np.nan_to_num(rain_frames[0], nan=0.0, posinf=0.0, neginf=0.0)*1000,  cmap=cmap_rain,  vmin=4, vmax=600,
                         extent=ext, aspect='auto', zorder=2, alpha=0.50,
                         interpolation='nearest')
        _irv = tm.imshow((np.nan_to_num(river_frames[0], nan=0.0, posinf=0.0, neginf=0.0)>0.015).astype(float), cmap=cmap_river,
                         vmin=0, vmax=1, extent=ext, aspect='auto', zorder=3, alpha=0.45,
                         interpolation='nearest')
        tm.set_xlim(0, W); tm.set_ylim(H, 0)
        _tt = tm.text(0.015, 0.975, "", transform=tm.transAxes,
                      fontsize=11, fontweight='bold', color='white', va='top',
                      bbox=dict(boxstyle='round', facecolor=PANEL, alpha=0.88, edgecolor=ACC))
        _st = ts.text(0.05, 0.97, "", fontsize=9, family='monospace',
                      color=TCLR, va='top', transform=ts.transAxes)
        frames_pil = []
        for fi in range(num_frames):
            rn = np.nan_to_num(rain_frames[fi], nan=0.0, posinf=0.0, neginf=0.0) * 1000
            _irn.set_data(rn)
            _irn.set_clim(4, min(max(float(rn.max()), 40)*1.3, 600))
            _irv.set_data((np.nan_to_num(river_frames[fi], nan=0.0, posinf=0.0, neginf=0.0) > 0.015).astype(float))
            _tt.set_text(f"Time: {times_list[fi]}\nRain: {stats['rain_mm'][fi]:.0f}/{rainfall_mm:.0f} mm")
            _st.set_text(f"{scenario_name}\nPrevention: {prevention_str}\n\n"
                         f"Time:    {times_list[fi]}\n"
                         f"Rain:    {stats['rain_mm'][fi]:.1f} mm\n"
                         f"Flooded: {stats['flooded_pct'][fi]:.1f}%\n"
                         f"MaxD:    {stats['max_depth_mm'][fi]:.0f} mm")
            tmp_fig.canvas.draw()
            buf = io.BytesIO()
            tmp_fig.savefig(buf, format='png', dpi=75,
                            bbox_inches='tight', facecolor=DARK)
            buf.seek(0)
            # FIX: convert to 'RGB' first, then quantize — avoids RGBA quantize crash
            pil_frame = PILImage.open(buf).copy().convert('RGB')   # type: ignore
            frames_pil.append(pil_frame.quantize(method=2))
            print(f"  GIF frame {fi + 1}/{num_frames} …", end='\r', flush=True)

        print()  # newline after progress overwrite
        plt.close(tmp_fig)
        frames_pil[0].save(out, save_all=True, append_images=frames_pil[1:],
                           loop=0, duration=int(1000 / 5))
        print(f"  GIF saved  ({len(frames_pil)} frames)")
        player['playing'] = was

    sl_frame.on_changed(on_frame)
    sl_speed.on_changed(on_speed)
    btn_play.on_clicked(on_play_pause)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_gif .on_clicked(on_save_gif)

    draw(0)
    print("\n  Interactive viewer ready.")
    print("  Controls: Pause/Play | Step Back/Forward | Speed x | Save GIF")
    plt.show()

# =============================================================================
# SIMULATION GUI
# =============================================================================
# Two-tab layout: Tab 1 (Storm Scenario) mirrors the 25% GUI controls.
# Tab 2 (Prevention Measures) adds checkboxes and sliders for the floodwall
# and drainage canal. Both tabs feed into a single run_simulation() call.

class SimulationGUI:
    """Tkinter GUI — storm tab + separate prevention measures tab."""
    BG     = '#0D1117'
    PANEL  = '#161B22'
    CARD   = '#1C2333'
    BORDER = '#30363D'
    TEXT   = '#E6EDF3'
    ACCENT = '#4FC3F7'
    GREEN  = '#2EA043'
    ORANGE = '#F0883E'
    RED    = '#DA3633'
    WHITE  = '#FFFFFF'
    WALL   = '#FF5555'
    CANAL  = '#00DDEE'
    BASIN  = '#3FB950'
    ROAD   = '#FF9800'

    def _on_preset_change(self, *args, **kwargs):
        """Update all sliders, description, and prevention checkboxes when a scenario preset is selected."""
        self._is_randomized = False
        key = self.scenario_var.get()
        sc = SCENARIOS.get(key)
        if sc and sc["rainfall_mm"] is not None:
            self.sliders['rainfall_mm'].set(sc["rainfall_mm"])
            self.sliders['duration_h'].set(sc["duration_h"])
            self.pattern_var.set(sc["pattern"])
            self.desc_var.set(sc.get("desc", ""))
            # Set prevention checkboxes to scenario-appropriate defaults
            rain = sc["rainfall_mm"]
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
                self.rb_var.set(True);  self.er_var.set(True)
        else:
            self.desc_var.set("")
            self.fw_var.set(False); self.cn_var.set(False)
            self.rb_var.set(False); self.er_var.set(False)
        # Optionally update preview/status
        if hasattr(self, '_update_measure_display'):
            self._update_measure_display()

    def _update_measure_display(self, *args, **kwargs):
        """Update prevention measure display ON/OFF badges."""
        if hasattr(self, 'fw_var') and hasattr(self, 'fw_badge'):
            if self.fw_var.get():
                self.fw_badge.config(text="ON", bg="#3FB950")
            else:
                self.fw_badge.config(text="OFF", bg="#AA2222")
        if hasattr(self, 'cn_var') and hasattr(self, 'cn_badge'):
            if self.cn_var.get():
                self.cn_badge.config(text="ON", bg="#3FB950")
            else:
                self.cn_badge.config(text="OFF", bg="#008899")
        if hasattr(self, 'rb_var') and hasattr(self, 'rb_badge'):
            if self.rb_var.get():
                self.rb_badge.config(text="ON", bg="#3FB950")
            else:
                self.rb_badge.config(text="OFF", bg="#1a5c28")
        if hasattr(self, 'er_var') and hasattr(self, 'er_badge'):
            if self.er_var.get():
                self.er_badge.config(text="ON", bg="#3FB950")
            else:
                self.er_badge.config(text="OFF", bg="#7a4800")

    def __init__(self, dem: np.ndarray, cellsize: float):
        self.dem      = dem
        self.cellsize = cellsize

        self.root = tk.Tk()
        self.root.title("Jade Valley Flood Simulator — Prevention Measures (75%)")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1200, int(screen_w * 0.78))
        win_h = min(780,  int(screen_h * 0.82))
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.update_idletasks()
        # Modernize style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame',        background=self.BG)
        style.configure('Card.TFrame',   background=self.CARD)
        style.configure('TNotebook',     background=self.BG, tabmargins=[2, 5, 0, 0])
        style.configure('TNotebook.Tab', background=self.PANEL,
                foreground=self.TEXT,
                font=('Segoe UI', 11, 'bold'),
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

        # ── Title ─────────────────────────────────────────────────────────
        ttk.Label(self.root, text="JADE VALLEY SUBDIVISION",
                  style='Title.TLabel').pack(pady=(14, 0))
        ttk.Label(self.root,
                  text="Flood Simulation with Prevention Measures  —  75% Build",
                  style='Desc.TLabel').pack()

        # ── Notebook (two tabs) ──────────────────────────────────────────
        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=16, pady=10)

        tab_storm = ttk.Frame(nb)
        tab_prev  = ttk.Frame(nb)
        nb.add(tab_storm, text='Storm Scenario')
        nb.add(tab_prev,  text='Prevention Measures')

        self._build_storm_tab(tab_storm)
        self._build_prevention_tab(tab_prev)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=16, pady=(4, 14))
        _BF = ('Segoe UI', 10, 'bold')

        tk.Button(btn_frame, text="RANDOMIZE",
              bg='#6E40C9', fg='white', activebackground='#8957E5',
              font=_BF, width=14, height=1, bd=0, cursor='hand2',
              command=self._randomize).pack(side='left', padx=4)
        tk.Button(btn_frame, text="▶  RUN SIMULATION",
              bg=self.GREEN, fg='white', activebackground='#3FB950',
              font=_BF, width=16, height=1, bd=0, cursor='hand2',
              command=self._run).pack(side='left', padx=4)
        tk.Button(btn_frame, text="✕  EXIT",
              bg=self.RED, fg='white', activebackground='#F85149',
              font=_BF, width=10, height=1, bd=0, cursor='hand2',
              command=self.root.destroy).pack(side='right', padx=4)

        # ── Info bar ──────────────────────────────────────────────────────
        info = (f"DEM: {dem.shape[0]}×{dem.shape[1]}  |  Cell: {cellsize:.1f} m  |  "
            f"Elev: {dem.min():.1f}–{dem.max():.1f} m  |  "
            f"Area: {dem.size * cellsize**2 / 10000:.1f} ha")
        tk.Label(self.root, text=info, bg=self.BORDER, fg='#8B949E',
             font=('Consolas', 9), pady=4, wraplength=win_w-40, justify='center').pack(fill='x', side='bottom')

        self._is_randomized = False
        self._on_preset_change()
        self.root.mainloop()

    # ── Tab 1: Storm Scenario  (identical to 25% GUI) ─────────────────────

    def _build_storm_tab(self, parent):
        main = ttk.Frame(parent)
        main.pack(fill='both', expand=True, padx=12, pady=10)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        left  = ttk.Frame(main); left .grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        right = ttk.Frame(main); right.grid(row=0, column=1, sticky='nsew', padx=(8, 0))

        # Scenario presets with colour coding by severity.
        ttk.Label(left, text="SCENARIO PRESET",
                  style='Header.TLabel').pack(anchor='w')
        self.scenario_var = tk.StringVar(value="3")
        sc_frame = ttk.Frame(left); sc_frame.pack(fill='x', pady=4)
        _SC_COLORS = {
            "1": "#3FB950",
            "2": "#79C0FF",
            "3": "#FFD740",
            "4": "#F0883E",
            "5": "#FF6B6B",
            "6": "#FF1744",
        }
        for key, sc in SCENARIOS.items():
            if sc["rainfall_mm"] is None:
                continue
            color = _SC_COLORS.get(key, self.TEXT)
            label = f"{sc['name']}  ({sc['rainfall_mm']} mm / {sc['duration_h']} h)"
            tk.Radiobutton(
                sc_frame, text=label,
                variable=self.scenario_var, value=key,
                bg=self.BG, fg=color, selectcolor=self.PANEL,
                activebackground=self.BG, activeforeground=self.ACCENT,
                font=('Segoe UI', 9), anchor='w',
                command=self._on_preset_change
            ).pack(fill='x', pady=1)
        self.desc_var = tk.StringVar()
        tk.Label(sc_frame, textvariable=self.desc_var, bg=self.CARD,
                 fg='#8B949E', font=('Segoe UI', 9, 'italic'),
                 wraplength=380, justify='left', padx=8, pady=6
                 ).pack(fill='x', pady=(6, 0))

        # PAGASA rainfall classification reference bands.
        ref_frame = tk.Frame(sc_frame, bg=self.CARD)
        ref_frame.pack(fill='x', padx=4, pady=(4, 2))
        tk.Label(ref_frame, text="PAGASA Classification (mm/hr):",
                 bg=self.CARD, fg='#8B949E',
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=4)
        band_row = tk.Frame(ref_frame, bg=self.CARD)
        band_row.pack(fill='x', padx=4, pady=2)
        for lbl, col in [("Light  0-7.5",   "#3FB950"),
                          ("Moderate 7.5-15", "#79C0FF"),
                          ("Heavy 15-30",    "#FFD740"),
                          ("Intense >30",    "#FF6B6B")]:
            tk.Label(band_row, text=f"  {lbl}  ",
                     bg=col, fg='#0D1117',
                     font=('Segoe UI', 7, 'bold'),
                     relief='flat', padx=2).pack(side='left', padx=2, pady=1)

        # Storm pattern
        sep = ttk.Frame(left); sep.pack(fill='x', pady=8)
        ttk.Label(sep, text="STORM PATTERN",
                  style='Header.TLabel').pack(anchor='w')
        self.pattern_var = tk.StringVar(value="burst")
        for val, desc in [("uniform",     "Constant rate"),
                          ("progressive", "Builds up → peaks"),
                          ("burst",       "Bell-curve peak mid-storm"),
                          ("decreasing",  "Heavy start → tapers off")]:
            tk.Radiobutton(sep, text=f"{val}  —  {desc}",
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

        add_slider(right, 'rainfall_mm', 'Rainfall Total (mm)       [5–500]',      5,   500, 130)
        add_slider(right, 'duration_h',  'Storm Duration (hours)    [0.5–24]',     0.5,  24,  4.0, res=0.5)
        add_slider(right, 'timestep_min','Timestep (minutes)        [5–60]',        5,   60,  10)
        add_slider(right, 'wind_speed',  'Wind Speed (km/h)         [0–200]',       0,  200,   0)
        add_slider(right, 'wind_dir',    'Wind Direction (°)        [0–360]',       0,  359, 270)
        add_slider(right, 'soil_sat',    'Soil Saturation (%)       [0–100]',       0,  100,  30)
        add_slider(right, 'drain_cap',   'Drainage Capacity (mm/hr) [0.5–50]',    0.5,  50,   5.0, res=0.5)

        tf = ttk.Frame(right); tf.pack(fill='x', pady=4)
        ttk.Label(tf, text='Start Time (HH:MM)').pack(anchor='w')
        self.start_time_var = tk.StringVar(value="14:00")
        tk.Entry(tf, textvariable=self.start_time_var, bg=self.PANEL, fg=self.TEXT,
                 insertbackground=self.TEXT, font=('Consolas', 11), width=8
                 ).pack(anchor='w')

    # ── Tab 2: Prevention Measures (4 cards, 2×2 grid) ────────────────────

    def _build_prevention_tab(self, parent):
        import tkinter as tk
        outer = ttk.Frame(parent)
        outer.pack(fill='both', expand=True, padx=12, pady=10)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.rowconfigure(2, weight=1)

        # How-to banner
        tk.Label(outer,
                 text="Enable any combination of measures, then click RUN on the Storm tab.  "
                      "All measures physically modify the DEM — water is genuinely blocked "
                      "or rerouted, not just drawn on top.",
                 bg=self.CARD, fg='#8B949E', font=('Segoe UI', 9, 'italic'),
                 wraplength=740, justify='left', padx=14, pady=8
                 ).grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))

        # ── Measure 1: Floodwall ──────────────────────────────────────────
        card1 = tk.Frame(outer, bg=self.CARD, bd=0,
                         highlightthickness=2, highlightbackground='#3a1a1a')
        card1.grid(row=1, column=0, sticky='nsew', padx=(0, 5), pady=(0, 5))

        hdr1 = tk.Frame(card1, bg=self.WALL)
        hdr1.pack(fill='x')
        self.fw_var = tk.BooleanVar(value=False)
        tk.Checkbutton(hdr1, text="  Riverbank Floodwall",
                       variable=self.fw_var,
                       bg=self.WALL, fg='white', selectcolor='#AA2222',
                       activebackground=self.WALL, activeforeground='white',
                       font=('Segoe UI', 11, 'bold'), anchor='w',
                       command=self._update_measure_display
                       ).pack(side='left', padx=6, pady=6)
        self.fw_badge = tk.Label(hdr1, text="OFF", bg='#AA2222', fg='white',
                                 font=('Segoe UI', 9, 'bold'), padx=8, pady=2)
        self.fw_badge.pack(side='right', padx=8)
        tk.Label(card1,
                 text="Raises the western river bank cells by the chosen height.\n"
                      "Creates a physical DEM barrier — flood routing cannot spill\n"
                      "until the river level exceeds the raised crest elevation.",
                 bg=self.CARD, fg='#8B949E', font=('Segoe UI', 9),
                 justify='left', padx=14, pady=6).pack(anchor='w')
        wh_frame = tk.Frame(card1, bg=self.CARD)
        wh_frame.pack(fill='x', padx=12, pady=(2, 2))
        tk.Label(wh_frame, text="Wall Height (m)  [1.0 – 3.0]",
                 bg=self.CARD, fg=self.TEXT, font=('Segoe UI', 10)).pack(anchor='w')
        self.wall_height_slider = tk.Scale(
            wh_frame, from_=1.0, to=3.0, orient='horizontal',
            resolution=0.25, length=260,
            bg=self.CARD, fg=self.WALL, troughcolor=self.PANEL,
            highlightthickness=0, font=('Consolas', 9),
            activebackground=self.WALL, command=self._update_measure_display)
        self.wall_height_slider.set(2.0)
        self.wall_height_slider.pack(fill='x')
        tk.Label(card1, text="Effect: delays 100-yr flood onset by 2–4 hrs",
                 bg=self.CARD, fg='#3FB950', font=('Segoe UI', 9, 'italic'),
                 padx=14, pady=4).pack(anchor='w', pady=(0, 8))

        # ── Measure 2: Drainage Canal ─────────────────────────────────────
        card2 = tk.Frame(outer, bg=self.CARD, bd=0,
                         highlightthickness=2, highlightbackground='#003a3a')
        card2.grid(row=1, column=1, sticky='nsew', padx=(5, 0), pady=(0, 5))

        hdr2 = tk.Frame(card2, bg=self.CANAL)
        hdr2.pack(fill='x')
        self.cn_var = tk.BooleanVar(value=False)
        tk.Checkbutton(hdr2, text="  Drainage Canal Network",
                       variable=self.cn_var,
                       bg=self.CANAL, fg='#003333', selectcolor='#008899',
                       activebackground=self.CANAL, activeforeground='#003333',
                       font=('Segoe UI', 11, 'bold'), anchor='w',
                       command=self._update_measure_display
                       ).pack(side='left', padx=6, pady=6)
        self.cn_badge = tk.Label(hdr2, text="OFF", bg='#008899', fg='white',
                                 font=('Segoe UI', 9, 'bold'), padx=8, pady=2)
        self.cn_badge.pack(side='right', padx=8)
        tk.Label(card2,
                 text="Lowers a corridor of cells (east outlet + south branch)\n"
                      "to create open channels in the DEM.  Runoff naturally\n"
                      "flows into the canals, draining away from residential core.",
                 bg=self.CARD, fg='#8B949E', font=('Segoe UI', 9),
                 justify='left', padx=14, pady=6).pack(anchor='w')
        cd_frame = tk.Frame(card2, bg=self.CARD)
        cd_frame.pack(fill='x', padx=12, pady=(2, 2))
        tk.Label(cd_frame, text="Canal Depth (m)  [1.0 – 4.0]",
                 bg=self.CARD, fg=self.TEXT, font=('Segoe UI', 10)).pack(anchor='w')
        self.canal_depth_slider = tk.Scale(
            cd_frame, from_=1.0, to=4.0, orient='horizontal',
            resolution=0.25, length=260,
            bg=self.CARD, fg=self.CANAL, troughcolor=self.PANEL,
            highlightthickness=0, font=('Consolas', 9),
            activebackground=self.CANAL, command=self._update_measure_display)
        self.canal_depth_slider.set(2.0)
        self.canal_depth_slider.pack(fill='x')
        tk.Label(card2, text="Effect: reduces inundation area by ~15–25%",
                 bg=self.CARD, fg='#3FB950', font=('Segoe UI', 9, 'italic'),
                 padx=14, pady=4).pack(anchor='w', pady=(0, 8))

        # ── Measure 3: Retention Basin ────────────────────────────────────
        card3 = tk.Frame(outer, bg=self.CARD, bd=0,
                         highlightthickness=2, highlightbackground='#1a3a1a')
        card3.grid(row=2, column=0, sticky='nsew', padx=(0, 5), pady=(5, 0))

        hdr3 = tk.Frame(card3, bg=self.BASIN)
        hdr3.pack(fill='x')
        self.rb_var = tk.BooleanVar(value=False)
        tk.Checkbutton(hdr3, text="  Retention Basin",
                       variable=self.rb_var,
                       bg=self.BASIN, fg='#001a00', selectcolor='#1a5c28',
                       activebackground=self.BASIN, activeforeground='#001a00',
                       font=('Segoe UI', 11, 'bold'), anchor='w',
                       command=self._update_measure_display
                       ).pack(side='left', padx=6, pady=6)
        self.rb_badge = tk.Label(hdr3, text="OFF", bg='#1a5c28', fg='white',
                                 font=('Segoe UI', 9, 'bold'), padx=8, pady=2)
        self.rb_badge.pack(side='right', padx=8)
        tk.Label(card3,
                 text="Excavates a stormwater pond on undeveloped perimeter land\n"
                      "(outer buffer zone — never inside the residential core).\n"
                      "Intercepts runoff draining off residential blocks, stores\n"
                      "it at peak rainfall, releases slowly — cuts inundation depth.",
                 bg=self.CARD, fg='#8B949E', font=('Segoe UI', 9),
                 justify='left', padx=14, pady=6).pack(anchor='w')
        bd_frame = tk.Frame(card3, bg=self.CARD)
        bd_frame.pack(fill='x', padx=12, pady=(2, 2))
        tk.Label(bd_frame, text="Basin Depth (m)  [3.0 – 10.0]",
                 bg=self.CARD, fg=self.TEXT, font=('Segoe UI', 10)).pack(anchor='w')
        self.basin_depth_slider = tk.Scale(
            bd_frame, from_=3.0, to=10.0, orient='horizontal',
            resolution=0.5, length=260,
            bg=self.CARD, fg=self.BASIN, troughcolor=self.PANEL,
            highlightthickness=0, font=('Consolas', 9),
            activebackground=self.BASIN, command=self._update_measure_display)
        self.basin_depth_slider.set(6.0)
        self.basin_depth_slider.pack(fill='x')
        tk.Label(card3, text="Effect: lowers peak depth by ~20–40%; basin fills live",
                 bg=self.CARD, fg='#3FB950', font=('Segoe UI', 9, 'italic'),
                 padx=14, pady=4).pack(anchor='w', pady=(0, 8))

        # ── Measure 4: Elevated Emergency Road ───────────────────────────
        card4 = tk.Frame(outer, bg=self.CARD, bd=0,
                         highlightthickness=2, highlightbackground='#3a2a00')
        card4.grid(row=2, column=1, sticky='nsew', padx=(5, 0), pady=(5, 0))

        hdr4 = tk.Frame(card4, bg=self.ROAD)
        hdr4.pack(fill='x')
        self.er_var = tk.BooleanVar(value=False)
        tk.Checkbutton(hdr4, text="  Elevated Emergency Road",
                       variable=self.er_var,
                       bg=self.ROAD, fg='#1a1000', selectcolor='#7a4800',
                       activebackground=self.ROAD, activeforeground='#1a1000',
                       font=('Segoe UI', 11, 'bold'), anchor='w',
                       command=self._update_measure_display
                       ).pack(side='left', padx=6, pady=6)
        self.er_badge = tk.Label(hdr4, text="OFF", bg='#7a4800', fg='white',
                                 font=('Segoe UI', 9, 'bold'), padx=8, pady=2)
        self.er_badge.pack(side='right', padx=8)
        tk.Label(card4,
                 text="Raises a cross-subdivision road corridor above flood level.\n"
                      "Acts as a berm that guides flow east/west while keeping\n"
                      "emergency vehicle access open throughout the flood event.",
                 bg=self.CARD, fg='#8B949E', font=('Segoe UI', 9),
                 justify='left', padx=14, pady=6).pack(anchor='w')
        rh_frame = tk.Frame(card4, bg=self.CARD)
        rh_frame.pack(fill='x', padx=12, pady=(2, 2))
        tk.Label(rh_frame, text="Road Height (m)  [0.5 – 3.0]",
                 bg=self.CARD, fg=self.TEXT, font=('Segoe UI', 10)).pack(anchor='w')
        self.road_height_slider = tk.Scale(
            rh_frame, from_=0.5, to=3.0, orient='horizontal',
            resolution=0.25, length=260,
            bg=self.CARD, fg=self.ROAD, troughcolor=self.PANEL,
            highlightthickness=0, font=('Consolas', 9),
            activebackground=self.ROAD, command=self._update_measure_display)
        self.road_height_slider.set(1.5)
        self.road_height_slider.pack(fill='x')
        tk.Label(card4, text="Effect: diverts runoff; road stays dry up to design flood",
                 bg=self.CARD, fg='#3FB950', font=('Segoe UI', 9, 'italic'),
                 padx=14, pady=4).pack(anchor='w', pady=(0, 8))

    def _randomize(self, *args, **kwargs):
        """Randomize rainfall, scenario parameters, and prevention checkboxes."""
        # Rainfall scenario classes
        classes = [
            ("Light Rain",        10,  50,  1.0,  3.0),
            ("Moderate Rain",     40, 100,  2.0,  5.0),
            ("Heavy Rain",        80, 180,  3.0,  6.0),
            ("Typhoon Signal 1", 140, 220,  4.0,  8.0),
            ("Typhoon Signal 2", 200, 300,  6.0, 10.0),
            ("Typhoon Signal 3", 280, 450,  8.0, 14.0),
        ]
        name, rlo, rhi, dlo, dhi = random.choice(classes)
        rain = random.randint(rlo, rhi)
        dur  = round(random.uniform(dlo, dhi) * 2) / 2
        wind = random.choice([0, 0, random.randint(20, 60),
                               random.randint(60, 120), random.randint(100, 180)])
        self.sliders['rainfall_mm'].set(rain)
        self.sliders['duration_h'] .set(dur)
        self.sliders['timestep_min'].set(10)
        self.sliders['wind_speed']  .set(wind)
        self.sliders['wind_dir']    .set(random.randint(0, 359))
        self.sliders['soil_sat']    .set(random.randint(10, 85))
        self.sliders['drain_cap']   .set(random.choice([1.5, 2.0, 3.0, 5.0, 8.0]))
        self.pattern_var.set(random.choice(["uniform","progressive","burst","decreasing"]))
        self.start_time_var.set(f"{random.randint(0,23):02d}:00")
        self._is_randomized = True
        self.desc_var.set(f"Randomized: {name}  |  {rain} mm / {dur} h")
        # Set prevention checkboxes based on randomized rain
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
            self.rb_var.set(random.choice([True, False]))
            self.er_var.set(random.choice([True, False]))
        if hasattr(self, '_update_measure_display'):
            self._update_measure_display()

    def _run(self):
        rainfall_mm = float(self.sliders['rainfall_mm'].get())
        duration_h  = float(self.sliders['duration_h'] .get())
        timestep_m  = int  (self.sliders['timestep_min'].get())
        wind_spd    = float(self.sliders['wind_speed']  .get())
        wind_dir    = float(self.sliders['wind_dir']    .get())
        soil_sat    = float(self.sliders['soil_sat']    .get())
        drain       = float(self.sliders['drain_cap']   .get())
        pattern     = self.pattern_var.get()
        start_str   = self.start_time_var.get().strip()
        try:
            h, m = map(int, start_str.split(':'))
            if not (0 <= h < 24 and 0 <= m < 60):
                raise ValueError
        except (ValueError, AttributeError):
            print(f"  [warn] Invalid start time '{start_str}' — defaulting to 14:00")
            start_str = "14:00"

        use_fw    = self.fw_var.get()
        use_cn    = self.cn_var.get()
        use_rb    = self.rb_var.get()
        use_er    = self.er_var.get()
        wall_h    = float(self.wall_height_slider.get())
        canal_d   = float(self.canal_depth_slider.get())
        basin_d   = float(self.basin_depth_slider.get())
        road_h    = float(self.road_height_slider.get())

        if self._is_randomized:
            scenario_name = f"Randomized ({rainfall_mm:.0f} mm / {duration_h:.1f} h)"
        else:
            key = self.scenario_var.get()
            sc  = SCENARIOS.get(key)
            scenario_name = (sc["name"] if sc and sc["rainfall_mm"] is not None
                             else f"Custom ({rainfall_mm:.0f} mm / {duration_h:.1f} h)")
        if wind_spd >= 100:
            scenario_name += f" + Wind {wind_spd:.0f} km/h"

        self.root.destroy()

        run_simulation(
            dem=self.dem, cellsize=self.cellsize,
            rainfall_mm=rainfall_mm, duration_h=duration_h,
            timestep_min=timestep_m, start_time_str=start_str,
            wind_speed=wind_spd, wind_dir=wind_dir,
            soil_sat_pct=soil_sat, drain_cap=drain,
            pattern=pattern, scenario_name=scenario_name,
            use_floodwall=use_fw, use_canal=use_cn,
            use_basin=use_rb, use_road=use_er,
            wall_height=wall_h, canal_depth=canal_d,
            basin_depth=basin_d, road_height=road_h,
        )

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  JADE VALLEY SUBDIVISION — FLOOD SIMULATION  (75% BUILD)")
    print("  Prevention Measures: Floodwall + Canal + Retention Basin")
    print("                       + Elevated Emergency Road")
    print("  Davao City, Philippines")
    print("=" * 68)

    print("\nLoading terrain data…")
    dem, cellsize = load_dem()

    print("\nLaunching GUI…")
    SimulationGUI(dem, cellsize)

    print("\n" + "=" * 68)
    print("  DONE — GIF saved to Results/animations/")
    print("=" * 68)