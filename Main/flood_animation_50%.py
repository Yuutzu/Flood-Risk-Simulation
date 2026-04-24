"""
=============================================================================
 JADE VALLEY SUBDIVISION — FLOOD SIMULATION WITH PREVENTION MEASURES  (50%)
 Davao City, Philippines
=============================================================================
 This is the 50% milestone build — an extension of flood_animation.py (25%).

 What's new in 50%:
   • Prevention Measures Panel in the GUI (separate tab, usable alongside
     any storm scenario)
   • Prevention Measure 1 — Riverbank Floodwall  (+1.5 m along west bank)
   • Prevention Measure 2 — Drainage Canal Network  (east outlet + south branch)
   • Both measures modify the DEM before the simulation runs, so the flood
     physics itself responds realistically — water is blocked / rerouted.
   • The animation shows three layers simultaneously:
       Layer 1  — JPEG map background (aligned to DEM grid)
       Layer 2  — River channel band (always-visible sky-blue)
       Layer 3  — Rain accumulation (green → yellow → red, low opacity)
       Layer 4  — River overflow spread (blue wash, low opacity)
       Layer 5  — Prevention infrastructure drawn as coloured overlays:
                    floodwall = red hatched line along west bank
                    drainage canals = bright cyan corridors
   • Stats panel shows both flood numbers AND which measures are active
   • A third prevention-only result panel appears after the animation ends
     showing the difference map: baseline depth minus improved depth.

 Run:  python Main/flood_animation_50.py
=============================================================================
"""

import heapq
import io
import os
import random
import sys
import warnings
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, Slider
from tkinter import ttk

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
    "1": {"name": "Light Rain",
          "rainfall_mm": 8,  "duration_h": 2.0,  "pattern": "uniform",
          "desc": "Avg 4 mm/hr — minor puddles, no flood risk"},
        "2": {"name": "Moderate Rain",
            "rainfall_mm": 36, "duration_h": 3.0,  "pattern": "progressive",
            "desc": "Avg 12 mm/hr — minor street flooding; low zones collect water"},
    "3": {"name": "Heavy Rain",
          "rainfall_mm": 35, "duration_h": 4.0,  "pattern": "burst",
          "desc": "Avg 8.75 mm/hr — low areas may collect water"},
    "4": {"name": "Typhoon Signal 1 (Tropical Depression)",
          "rainfall_mm": 100, "duration_h": 8.0, "pattern": "progressive",
          "desc": "Avg 12.5 mm/hr — river may rise; monitor advisories"},
    "5": {"name": "Typhoon Signal 2 (Tropical Storm)",
          "rainfall_mm": 180, "duration_h": 12.0, "pattern": "burst",
          "desc": "Avg 15 mm/hr — widespread flooding; prepare to evacuate"},
    "6": {"name": "Typhoon Signal 3 (Severe Typhoon)",
          "rainfall_mm": 300, "duration_h": 18.0, "pattern": "burst",
          "desc": "Avg 17 mm/hr — catastrophic flooding; evacuate"},
    "7": {"name": "Custom — I will enter my own values",
          "rainfall_mm": None, "duration_h": None, "pattern": None, "desc": ""},
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


def apply_prevention_measures(dem: np.ndarray, cellsize: float,
                               use_floodwall: bool,
                               use_canal: bool,
                               wall_height: float = 1.5,
                               canal_depth: float = 2.0) -> tuple:
    """
    Combine the requested prevention measures into a single modified DEM.
    Returns (modified_dem, wall_mask, canal_mask) — all three always
    returned so the animation layer code is simple even when a measure is off.
    """
    modified   = dem.copy()
    wall_mask  = np.zeros(dem.shape, dtype=bool)
    canal_mask = np.zeros(dem.shape, dtype=bool)

    if use_floodwall:
        modified, wall_mask = apply_floodwall(modified, cellsize, wall_height)
    if use_canal:
        modified, canal_mask = apply_drainage_canal(modified, cellsize, canal_depth)
    return modified, wall_mask, canal_mask

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

        self.canal_mask = canal_mask if canal_mask is not None else np.zeros_like(dem, dtype=bool)
        self.wall_mask  = wall_mask  if wall_mask  is not None else np.zeros_like(dem, dtype=bool)
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
                            be = self.dem[nr, nc]
                            if be < self.bank_elev[r, c]:
                                self.bank_elev[r, c] = be
        valid = self.river_mask & np.isfinite(self.bank_elev)
        if valid.any():
            self.bank_elev[~valid & self.river_mask] = self.dem[~valid & self.river_mask]
            self.river_level[valid]  = self.bank_elev[valid]
            self.flood_base_wse      = float(np.nanpercentile(self.bank_elev[valid], 50))
        else:
            self.flood_base_wse = float(self.dem.min())
        self.river_level_init = self.river_level.copy()
        self.bfs_front        = self.river_mask.copy()
        if not np.allclose(self.dem, dem):
            print("[DEBUG] DEM was modified for prevention measures.")
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
            blue_alpha = 0.25
        elif accum_mm < 60.0:
            # Moderate rain: further reduced overflow and blue shading
            ramp = np.clip((accum_mm - 36.0) / 24.0, 0.0, 1.0) * 0.18 + 0.10
            rise_mult = ramp * (1.5 + 2.5 * self.flow_weight[self.river_mask])
            hops = 1
            blue_alpha = 0.22
        elif accum_mm < 120.0:
            # Heavy rain: more pronounced overflow
            ramp = np.clip((accum_mm - 60.0) / 60.0, 0.0, 1.0) * 0.7 + 0.5
            rise_mult = ramp * (3.5 + 8.0 * self.flow_weight[self.river_mask])
            hops = 2
            blue_alpha = 0.52
        else:
            # Typhoon: severe overflow
            ramp = 1.0
            rise_mult = ramp * (4.0 + 10.0 * self.flow_weight[self.river_mask])
            hops = 4
            blue_alpha = 0.62
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
        # If canal is present, blue shading is reduced in canal cells
        if self.canal_mask is not None and self.canal_mask.any():
            blue_depth[self.canal_mask] *= 0.4
        # Make blue cell shading more unique per scenario
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
                # --- Major fix: block flow across floodwall unless overtopped ---
                if self.wall_mask is not None and self.wall_mask.any():
                    src_wall = self.wall_mask[r0:r1, c0:c1]
                    dst_wall = self.wall_mask[nr0:nr1, nc0:nc1]
                    src_elev = self.wall_elev[r0:r1, c0:c1]
                    dst_elev = self.wall_elev[nr0:nr1, nc0:nc1]
                    # Block flow from non-wall to wall unless overtopped
                    block = (src_wall != dst_wall)
                    # Only allow flow if water surface exceeds wall crest by >2cm
                    src_wse = wse[r0:r1, c0:c1]
                    dst_wse = wse[nr0:nr1, nc0:nc1]
                    crest   = np.maximum(src_elev, dst_elev)
                    allow   = (src_wse > crest + 0.02) | (dst_wse > crest + 0.02)
                    flow[block & ~allow] = 0.0
                dw[r0:r1,c0:c1]     -= flow
                dw[nr0:nr1,nc0:nc1] += flow
            self.rain_water = np.maximum(self.rain_water + dw, 0.0)
        np.clip(self.rain_water, 0.0, 5.0, out=self.rain_water)

    def apply_drainage(self, dt_h):
        # Scenario-adaptive drainage: less effective in severe scenarios
        scenario_factor = self._scenario_factor()
        ch_rate = 80.0 * self.flow_weight ** 2
        base_drain = self.drainage_capacity * (1.0 + 2.0 * self.elev_norm)
        # Reduce drainage in severe scenarios
        base_drain = base_drain * (1.0 - 0.45 * scenario_factor)
        rate = base_drain + ch_rate
        if self.canal_mask is not None and self.canal_mask.any():
            rate = rate + self.canal_drainage * 300.0  # 30x faster in canal
        drain_m = (rate / 1000.0) * dt_h
        self.rain_water  = np.maximum(self.rain_water  - drain_m,       0.0)
        self.river_water = np.maximum(self.river_water - drain_m * 0.4, 0.0)
        # Canal cells: force even more rapid drying for visual effect
        if self.canal_mask is not None and self.canal_mask.any():
            self.rain_water[self.canal_mask]  *= 0.30
            self.river_water[self.canal_mask] *= 0.30
        self.rain_water [self.rain_water  < 0.003] *= 0.60
        self.river_water[self.river_water < 0.004] *= 0.65

    def step(self, rate_mmhr, dt_h, intensity=1.0, wind_map=None):
        rain_mm = rate_mmhr * dt_h * intensity
        self.add_rainfall(rain_mm, wind_map)
        self.route_water(iters=12)
        self.apply_river_overflow(rate_mmhr, dt_h, intensity)
        self.apply_infiltration(dt_h)
        self.apply_drainage(dt_h)

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
        (0.55, 0.00, 1.00, 0.75),
    ]
    return LinearSegmentedColormap.from_list('rain', colors)


def _river_cmap():
    # EXACTLY match the 25% simulation blue colormap
    return LinearSegmentedColormap.from_list(
        'river', [
            (0.00, 0.00, 0.00, 0.00),
            (0.12, 0.38, 0.82, 0.13),
            (0.12, 0.38, 0.82, 0.28),
            (0.12, 0.38, 0.82, 0.55),
            (0.12, 0.38, 0.82, 0.85),
            (0.00, 0.00, 0.40, 1.00)
        ], N=256)

# =============================================================================
# INTENSITY PATTERN  (identical to 25%)
# =============================================================================

def _intensity_factor(frame, total_frames, pattern):
    t = frame / max(total_frames - 1, 1)
    if pattern == 'progressive':
        return 0.2 + 1.8 * min(t * 1.5, 1.0)
    if pattern == 'burst':
        return 0.15 + 3.0 * float(np.exp(-((t - 0.45) ** 2) / 0.040))
    if pattern == 'decreasing':
        return max(2.0 - 1.8 * t, 0.05)
    return 1.0

# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN SIMULATION RUNNER  ← UPDATED for 50%                             ║
# ║                                                                          ║
# ║  Accepts:                                                                ║
# ║    use_floodwall / use_canal  — whether to apply each measure           ║
# ║    wall_height / canal_depth  — configurable from GUI                  ║
# ║                                                                          ║
# ║  Behaviour:                                                              ║
# ║    1. Applies prevention measures to DEM (if any selected)              ║
# ║    2. Runs the simulation on the modified DEM                            ║
# ║    3. ALSO runs the same simulation on the original DEM (baseline)      ║
# ║       so the stats panel can show the improvement live                  ║
# ║    4. Draws prevention infrastructure as coloured overlays on the map   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

def run_simulation(dem: np.ndarray, cellsize: float,
                   rainfall_mm: float, duration_h: float,
                   timestep_min: int, start_time_str: str,
                   wind_speed: float, wind_dir: float,
                   soil_sat_pct: float, drain_cap: float,
                   pattern: str, scenario_name: str,
                   use_floodwall: bool = False,
                   use_canal: bool = False,
                   wall_height: float = 1.5,
                   canal_depth: float = 2.0):

    any_prevention = use_floodwall or use_canal

    # ── Apply prevention measures to DEM ─────────────────────────────────────
    # Always keep a pristine copy of the DEM for baseline
    dem_orig = dem.copy()
    if any_prevention:
        print("\n  Applying prevention measures to DEM…")
        sim_dem, wall_mask, canal_mask = apply_prevention_measures(
            dem_orig, cellsize, use_floodwall, use_canal, wall_height, canal_depth)
        prevention_label = []
        if use_floodwall:
            prevention_label.append(f"Floodwall +{wall_height:.1f}m")
        if use_canal:
            prevention_label.append("Drainage Canal")
        prevention_str = " + ".join(prevention_label)
    else:
        sim_dem     = dem_orig.copy()
        wall_mask   = np.zeros(dem_orig.shape, dtype=bool)
        canal_mask  = np.zeros(dem_orig.shape, dtype=bool)
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
                          wall_mask=wall_mask)
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
        rain_frames .append(sim.rain_water .copy())
        river_frames.append(sim.river_water.copy())
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
                                   drainage_capacity_mmhr=drain_cap)
        base_stats = {"flooded_pct": [], "max_depth_mm": []}
        for fr in range(num_frames):
            inten = _intensity_factor(fr, num_frames, pattern)
            sim_base.step(rate_mmhr, dt_h, intensity=inten, wind_map=wmap)
            total_b = sim_base.rain_water + sim_base.river_water
            base_stats["flooded_pct"].append(
                float(np.sum(total_b > 0.01) / total_b.size * 100))
            base_stats["max_depth_mm"].append(float(total_b.max() * 1000))
        baseline_last_depth  = (sim_base.rain_water + sim_base.river_water).copy()
    else:
        base_stats = None
        baseline_last_depth = None

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

    # Canal overlay — lower opacity for better map visibility
    canal_rgba = np.zeros((H, W, 4), dtype=np.float32)
    if canal_mask.any():
        canal_rgba[canal_mask, 0] = 0.00
        canal_rgba[canal_mask, 1] = 0.88
        canal_rgba[canal_mask, 2] = 0.95
        canal_rgba[canal_mask, 3] = 0.28  # Lowered from 0.72 for less blocking

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
    WALL_CLR   = '#FF4444'
    CANAL_CLR  = '#00DDEE'

    title_suffix = f"  |  Prevention: {prevention_str}" if any_prevention else ""
    # Dynamically size the figure window to fit the screen
    import matplotlib
    backend = matplotlib.get_backend()
    # Set safe defaults
    screen_w, screen_h = 1600, 900
    fig_w, fig_h = 1300, 800
    dpi = 100
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        fig_w = min(int(screen_w * 0.85), 1500)
        fig_h = min(int(screen_h * 0.85), 900)
    except Exception:
        pass
    figsize = (fig_w / dpi, fig_h / dpi)
    DARK  = '#0D1117'
    TCLR  = '#E6EDF3'
    title_suffix = f"  |  Prevention: {prevention_str}" if any_prevention else ""
    fig = plt.figure(figsize=figsize, facecolor=DARK, dpi=dpi)
    fig.suptitle(
        f"JADE VALLEY  —  FLOOD SIMULATION  |  {scenario_name}{title_suffix}",
        fontsize=13, fontweight='bold', color=TCLR, y=0.979)
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
    im_bg    = ax_map.imshow(bg, extent=ext, aspect='auto',
                              zorder=1, interpolation='bilinear')
    # Layer 2: Rain accumulation (updated each frame)
    im_rain  = ax_map.imshow(rain_frames[0] * 1000,
                              cmap=cmap_rain, vmin=0, vmax=600,
                              extent=ext, aspect='auto',
                              zorder=2, interpolation='bilinear',
                              alpha=0.45)
    # Blue cell shading: EXACTLY match 25% (total water depth, nonlinear colormap)
    total_water = rain_frames[0] + river_frames[0]
    im_river = ax_map.imshow(
        np.clip(np.power(total_water / 1.1, 0.65), 0, 1),
        cmap=cmap_river, vmin=0, vmax=1,
        extent=ext, aspect='auto', zorder=3,
        interpolation='nearest', alpha=0.62)

    # Layer 4: Floodwall overlay (static — drawn once on top)
    if wall_mask.any():
        ax_map.imshow(wall_rgba, extent=ext, aspect='auto', zorder=5,
                      interpolation='nearest')

    # Layer 5: Drainage canal overlay (static — drawn once on top)
    if canal_mask.any():
        ax_map.imshow(canal_rgba, extent=ext, aspect='auto', zorder=5,
                      interpolation='nearest')

    ax_map.set_xlim(0, W); ax_map.set_ylim(H, 0)
    ax_map.tick_params(colors=TCLR, labelsize=8)
    for sp in ax_map.spines.values():
        sp.set_edgecolor('#30363D')

    # Colorbar
    cbar = fig.colorbar(im_rain, ax=ax_map, orientation='vertical',
                        pad=0.01, shrink=0.80)
    cbar.set_label("Rain Water Depth (mm)", color=TCLR, fontsize=9)
    cbar.set_ticks([0, 30, 100, 200, 400, 600])
    cbar.ax.set_yticklabels(['Dry','30','100','200','400','600+'],
                             color=TCLR, fontsize=8)
    cbar.ax.tick_params(colors=TCLR)

    # Legend for prevention overlays (lower opacity to match overlays)
    from matplotlib.patches import Patch
    legend_handles = []
    if wall_mask.any():
        legend_handles.append(Patch(facecolor=(0.9,0.15,0.1,0.32), edgecolor='r', label=f'Floodwall (+{wall_height:.1f} m)'))
    if canal_mask.any():
        legend_handles.append(Patch(facecolor=(0,0.88,0.95,0.28), edgecolor='c', label='Drainage Canal'))
    if legend_handles:
        ax_map.legend(handles=legend_handles, loc='lower right', fontsize=10, framealpha=0.85)
    # Effectiveness stats (difference between baseline and improved)
    # (Moved after stats_txt is defined)
    # Effectiveness stats (difference between baseline and improved)
    # Time badge
    time_txt = ax_map.text(
        0.015, 0.975, "", transform=ax_map.transAxes,
        fontsize=12, fontweight='bold', color='white', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL,
                  alpha=0.90, edgecolor=ACC, linewidth=1.5))

    # Stats text
    stats_txt = ax_stats.text(
        0.05, 0.97, "", fontsize=10, family='monospace',
        color=TCLR, va='top', transform=ax_stats.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#0D1117',
                  edgecolor=ACC, alpha=0.92, linewidth=1.2))

    # Effectiveness stats (difference between baseline and improved)
    if any_prevention and base_stats is not None:
        flood_diff = np.array(base_stats['flooded_pct']) - np.array(stats['flooded_pct'])
        depth_diff = np.array(base_stats['max_depth_mm']) - np.array(stats['max_depth_mm'])
        eff_str = (f"\nEffectiveness vs Baseline:\n"
                  f"Flooded area reduced: {flood_diff[-1]:.2f}%\n"
                  f"Max depth reduced: {depth_diff[-1]:.1f} mm")
        stats_txt.set_text(stats_txt.get_text() + eff_str)

    # Time badge
    time_txt = ax_map.text(
        0.015, 0.975, "", transform=ax_map.transAxes,
        fontsize=12, fontweight='bold', color='white', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL,
                  alpha=0.90, edgecolor=ACC, linewidth=1.5))

    # Stats text
    stats_txt = ax_stats.text(
        0.05, 0.97, "", fontsize=10, family='monospace',
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

    btn_play = Button(ax_btn_play, '⏸ Pause', color='#1B5E20', hovercolor='#2E7D32')
    btn_prev = Button(ax_btn_prev, '◀◀',      color='#0D47A1', hovercolor='#1565C0')
    btn_next = Button(ax_btn_next, '▶▶',      color='#0D47A1', hovercolor='#1565C0')
    btn_gif  = Button(ax_btn_gif,  '💾 GIF',   color='#4A148C', hovercolor='#6A1B9A')
    for b in (btn_play, btn_prev, btn_next, btn_gif):
        b.label.set_color('white'); b.label.set_fontsize(10)

    player = {'playing': True, 'frame': 0}
    DIRS = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW']
    compass = DIRS[int((wind_dir + 11.25) / 22.5) % 16]
    wind_str = f"{wind_speed:.0f} km/h {compass}" if wind_speed >= 1 else "None"

    def _risk_str(max_depth_mm, river_pct):
        # PAGASA-aligned: use max water depth and river overflow area
        # 50mm = ankle-deep (STANDBY), 150mm = shin (PRE-EVAC), 300mm = knee (MANDATORY), 600mm = waist (EVACUATE)
        if river_pct > 30 or max_depth_mm > 600:
            return "⚠ EVACUATE NOW",             '#FF1744'
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
        rn = rain_frames[fi] * 1000
        rv = river_frames[fi]
        im_rain .set_data(rn)
        im_rain .set_clim(0, min(max(float(rn.max()), 30) * 1.3, 600))
        # Blue cell shading: EXACTLY match 25% (total water depth, nonlinear colormap)
        total_water = rain_frames[fi] + river_frames[fi]
        im_river.set_data(np.clip(np.power(total_water / 1.1, 0.65), 0, 1))
        time_txt.set_text(
            f" Time: {times_list[fi]}\n"
            f" Rain: {stats['rain_mm'][fi]:.0f}/{rainfall_mm:.0f} mm\n"
            f" Wind: {wind_str}")

        # Use max water depth and river overflow area for risk level
        rsk, rsk_col = _risk_str(stats['max_depth_mm'][fi], stats['river_pct'][fi])

        # Improvement delta lines (only shown when prevention is active)
        if base_stats and fi < len(base_stats["flooded_pct"]):
            b_pct   = base_stats["flooded_pct"][fi]
            b_depth = base_stats["max_depth_mm"][fi]
            imp_pct  = b_pct   - stats["flooded_pct"][fi]
            imp_dep  = b_depth - stats["max_depth_mm"][fi]
            improve_lines = (
                f"\n"
                f"  vs BASELINE\n"
                f"  {'─'*26}\n"
                f"  Base flood  : {b_pct:.1f}%\n"
                f"  Now flooded : {stats['flooded_pct'][fi]:.1f}%\n"
                f"  Improvement : {imp_pct:+.1f}%  area\n"
                f"  Depth saved : {imp_dep:+.0f} mm\n"
            )
        else:
            improve_lines = ""

        stats_txt.set_text(
            f"  SCENARIO\n"
            f"  {'─'*26}\n"
            f"  {scenario_name}\n"
            f"\n"
            f"  PARAMETERS\n"
            f"  {'─'*26}\n"
            f"  Rainfall : {rainfall_mm:.0f} mm\n"
            f"  Rate     : {rate_mmhr:.1f} mm/hr ({pattern})\n"
            f"  Duration : {duration_h:.1f} hr\n"
            f"  Timestep : {timestep_min} min\n"
            f"  Wind     : {wind_str}\n"
            f"  Soil sat : {soil_sat_pct:.0f}%\n"
            f"  Drainage : {drain_cap:.1f} mm/hr\n"
            f"\n"
            f"  PREVENTION\n"
            f"  {'─'*26}\n"
            f"  {prevention_str}\n"
            f"\n"
            f"  LIVE STATUS  [{times_list[fi]}]\n"
            f"  {'─'*26}\n"
            f"  Elapsed  : {fi * timestep_min} min\n"
            f"  Fallen   : {stats['rain_mm'][fi]:.1f} mm\n"
            f"  Max depth: {stats['max_depth_mm'][fi]:.0f} mm\n"
            f"  Flooded  : {stats['flooded_pct'][fi]:.1f}%\n"
            f"\n"
            f"  RIVER OVERFLOW\n"
            f"  {'─'*26}\n"
            f"  Area     : {stats['river_pct'][fi]:.1f}%\n"
            f"  Max depth: {stats['max_river_mm'][fi]:.0f} mm\n"
            f"\n"
            f"  RISK LEVEL\n"
            f"  {rsk}"
            f"{improve_lines}"
        )
        stats_txt.get_bbox_patch().set_edgecolor(rsk_col)   # type: ignore

        if abs(sl_frame.val - fi) > 0.5:
            sl_frame.eventson = False
            sl_frame.set_val(fi)
            sl_frame.eventson = True
        fig.canvas.draw_idle()

    BASE_INTERVAL = 600

    def _anim_step(_):
        if player['playing']:
            draw(player['frame'] + 1)
        return []   # FIX: FuncAnimation requires an iterable return value;
                    # without this the animation timer fires but visuals don't update

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
            btn_play.label.set_text('⏸ Pause')
            btn_play.ax.set_facecolor('#1B5E20')
        else:
            btn_play.label.set_text('▶ Play')
            btn_play.ax.set_facecolor('#BF360C')
        fig.canvas.draw_idle()

    def on_prev(_):
        player['playing'] = False
        btn_play.label.set_text('▶ Play')
        btn_play.ax.set_facecolor('#BF360C')
        draw(player['frame'] - 1)

    def on_next(_):
        player['playing'] = False
        btn_play.label.set_text('▶ Play')
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
        _irn = tm.imshow(rain_frames[0]*1000,  cmap=cmap_rain,  vmin=4, vmax=600,
                         extent=ext, aspect='auto', zorder=2, alpha=0.50,
                         interpolation='nearest')
        _irv = tm.imshow((river_frames[0]>0.015).astype(float), cmap=cmap_river,
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
            rn = rain_frames[fi]*1000
            _irn.set_data(rn)
            _irn.set_clim(4, min(max(float(rn.max()), 40)*1.3, 600))
            _irv.set_data((river_frames[fi]>0.015).astype(float))
            _tt.set_text(f"⏱ {times_list[fi]}\nRain: {stats['rain_mm'][fi]:.0f}/{rainfall_mm:.0f} mm")
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
        plt.close(tmp_fig)
        frames_pil[0].save(out, save_all=True, append_images=frames_pil[1:],
                           loop=0, duration=int(1000 / 5))
        print(f"  ✓ GIF saved  ({len(frames_pil)} frames)")
        player['playing'] = was

    sl_frame.on_changed(on_frame)
    sl_speed.on_changed(on_speed)
    btn_play.on_clicked(on_play_pause)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_gif .on_clicked(on_save_gif)

    draw(0)
    print("\n  ✓ Interactive viewer ready.")
    print("  Controls: ⏸/▶ Play-Pause | ◀◀/▶▶ Step | Speed ×  |  💾 GIF")
    plt.show()

# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIMULATION GUI  — 50% VERSION                                          ║
# ║                                                                          ║
# ║  Two-tab layout:                                                         ║
# ║    Tab 1 "Storm Scenario" — identical to 25% GUI                        ║
# ║    Tab 2 "Prevention Measures" — NEW controls for floodwall + canal     ║
# ║                                                                          ║
# ║  The two tabs are completely independent: you can pick any storm        ║
# ║  scenario AND any combination of prevention measures.  Both are         ║
# ║  passed together to run_simulation() so one animation shows both.      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

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

    # --- Add missing methods if not present ---

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
            # Light/Moderate: off, Heavy and above: on floodwall, Typhoon: both
            rain = sc["rainfall_mm"]
            if rain <= 36:
                self.fw_var.set(False)
                self.cn_var.set(False)
            elif rain <= 90:
                self.fw_var.set(True)
                self.cn_var.set(False)
            else:
                self.fw_var.set(True)
                self.cn_var.set(True)
        else:
            self.desc_var.set("")
            self.fw_var.set(False)
            self.cn_var.set(False)
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

    def __init__(self, dem: np.ndarray, cellsize: float):
        self.dem      = dem
        self.cellsize = cellsize

        self.root = tk.Tk()
        self.root.title("Jade Valley Flood Simulator 50% — Prevention Measures")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)
        # More compact and modern window size
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1000, int(screen_w * 0.6))
        win_h = min(600, int(screen_h * 0.6))
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
                  text="Flood Simulation with Prevention Measures  —  50% Build",
                  style='Desc.TLabel').pack()

        # ── Notebook (two tabs) ──────────────────────────────────────────
        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=16, pady=10)

        tab_storm = ttk.Frame(nb)
        tab_prev  = ttk.Frame(nb)
        nb.add(tab_storm, text='🌧  Storm Scenario')
        nb.add(tab_prev,  text='🛡  Prevention Measures')

        self._build_storm_tab(tab_storm)
        self._build_prevention_tab(tab_prev)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=16, pady=(4, 14))
        _BF = ('Segoe UI', 10, 'bold')

        tk.Button(btn_frame, text="🎲  RANDOMIZE",
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

        # Scenario presets
        ttk.Label(left, text="SCENARIO PRESET",
                  style='Header.TLabel').pack(anchor='w')
        self.scenario_var = tk.StringVar(value="3")
        sc_frame = ttk.Frame(left); sc_frame.pack(fill='x', pady=4)
        for key, sc in SCENARIOS.items():
            if sc["rainfall_mm"] is None:
                continue
            tk.Radiobutton(
                sc_frame, text=sc['name'],
                variable=self.scenario_var, value=key,
                bg=self.BG, fg=self.TEXT, selectcolor=self.PANEL,
                activebackground=self.BG, activeforeground=self.ACCENT,
                font=('Segoe UI', 9), anchor='w',
                command=self._on_preset_change
            ).pack(fill='x', pady=1)
        self.desc_var = tk.StringVar()
        tk.Label(sc_frame, textvariable=self.desc_var, bg=self.CARD,
                 fg='#8B949E', font=('Segoe UI', 9, 'italic'),
                 wraplength=340, justify='left', padx=8, pady=6
                 ).pack(fill='x', pady=(6, 0))

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

    # ── Tab 2: Prevention Measures  ← NEW ─────────────────────────────────

    def _build_prevention_tab(self, parent):
        # Ensure tkinter is always available as tk
        import tkinter as tk
        # Use tk and ttk from global imports
        outer = ttk.Frame(parent)
        outer.pack(fill='both', expand=True, padx=12, pady=10)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        # ── How-to banner ─────────────────────────────────────────────────
        tk.Label(outer,
                 text="Enable one or both measures below, then click RUN on the Storm tab.  "
                      "Water is physically blocked/rerouted — not just drawn on top.",
                 bg=self.CARD, fg='#8B949E', font=('Segoe UI', 9, 'italic'),
                 wraplength=700, justify='left', padx=14, pady=8
                 ).grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 12))

        # ── Measure 1: Floodwall ──────────────────────────────────────────
        card1 = tk.Frame(outer, bg=self.CARD, bd=0,
                         highlightthickness=2, highlightbackground='#3a1a1a')
        card1.grid(row=1, column=0, sticky='nsew', padx=(0, 6), pady=4)

        hdr1 = tk.Frame(card1, bg=self.WALL)
        hdr1.pack(fill='x')
        self.fw_var = tk.BooleanVar(value=False)
        tk.Checkbutton(hdr1,
                       text="  Riverbank Floodwall",
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
        wh_frame.pack(fill='x', padx=12, pady=(4, 4))
        tk.Label(wh_frame, text="Wall Height (m)  [1.0 – 3.0]",
                 bg=self.CARD, fg=self.TEXT, font=('Segoe UI', 10)).pack(anchor='w')
        self.wall_height_slider = tk.Scale(
            wh_frame, from_=1.0, to=3.0, orient='horizontal',
            resolution=0.25, length=280,
            bg=self.CARD, fg=self.WALL, troughcolor=self.PANEL,
            highlightthickness=0, font=('Consolas', 9),
            activebackground=self.WALL, command=self._update_measure_display)
        self.wall_height_slider.set(2.0)
        self.wall_height_slider.pack(fill='x')

        self.fw_preview = tk.Label(card1,
                 text="Effect: delays 100-yr flood onset by 2–4 hrs",
                 bg=self.CARD, fg='#3FB950', font=('Segoe UI', 9, 'italic'),
                 padx=14, pady=4)
        self.fw_preview.pack(anchor='w', pady=(0, 10))

        # ── Measure 2: Drainage Canal ─────────────────────────────────────
        card2 = tk.Frame(outer, bg=self.CARD, bd=0,
                         highlightthickness=2, highlightbackground='#003a3a')
        card2.grid(row=1, column=1, sticky='nsew', padx=(6, 0), pady=4)

        hdr2 = tk.Frame(card2, bg=self.CANAL)
        hdr2.pack(fill='x')
        self.cn_var = tk.BooleanVar(value=False)
        tk.Checkbutton(hdr2,
                       text="  Drainage Canal Network",
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
        cd_frame.pack(fill='x', padx=12, pady=(4, 4))
        tk.Label(cd_frame, text="Canal Depth (m)  [1.0 – 4.0]",
                 bg=self.CARD, fg=self.TEXT, font=('Segoe UI', 10)).pack(anchor='w')
        self.canal_depth_slider = tk.Scale(
            cd_frame, from_=1.0, to=4.0, orient='horizontal',
            resolution=0.25, length=280,
            bg=self.CARD, fg=self.CANAL, troughcolor=self.PANEL,
            highlightthickness=0, font=('Consolas', 9),
            activebackground=self.CANAL, command=self._update_measure_display)
        self.canal_depth_slider.set(2.0)
        self.canal_depth_slider.pack(fill='x')

        self.cn_preview = tk.Label(card2,
                 text="Effect: reduces inundation area by ~15–25%",
                 bg=self.CARD, fg='#3FB950', font=('Segoe UI', 9, 'italic'),
                 padx=14, pady=4)
        self.cn_preview.pack(anchor='w', pady=(0, 10))

        # ── Combined status bar ────────────────────────────────────────────
        self.combined_status = tk.Label(
            card2,
            text="",
            bg=self.BG,
            fg=self.TEXT,
            font=('Segoe UI', 10)
        )

    def _randomize(self, *args, **kwargs):
        """Randomize rainfall, scenario parameters, and prevention checkboxes."""
        import matplotlib
        import random
        backend = matplotlib.get_backend() if hasattr(matplotlib, 'get_backend') else 'TkAgg'
        screen_w, screen_h = 1600, 900
        fig_w, fig_h = 1300, 800
        dpi = 100
        TCLR = '#E6EDF3'
        DARK = '#0D1117'
        title_suffix = ""
        scenario_name = "Randomized"
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy()
            fig_w = min(int(screen_w * 0.85), 1500)
            fig_h = min(int(screen_h * 0.85), 900)
        except Exception:
            pass
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
        self.desc_var.set(f"🎲 Randomized: {name}  |  {rain} mm / {dur} h")
        # Set prevention checkboxes based on randomized rain
        if rain <= 36:
            self.fw_var.set(False)
            self.cn_var.set(False)
        elif rain <= 90:
            self.fw_var.set(True)
            self.cn_var.set(False)
        else:
            self.fw_var.set(True)
            self.cn_var.set(True)
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
            start_str = "14:00"

        use_fw    = self.fw_var.get()
        use_cn    = self.cn_var.get()
        wall_h    = float(self.wall_height_slider.get())
        canal_d   = float(self.canal_depth_slider.get())

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
            wall_height=wall_h, canal_depth=canal_d,
        )

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  JADE VALLEY SUBDIVISION — FLOOD SIMULATION  (50% BUILD)")
    print("  Prevention Measures: Riverbank Floodwall + Drainage Canal")
    print("  Davao City, Philippines")
    print("=" * 68)

    print("\nLoading terrain data…")
    dem, cellsize = load_dem()

    print("\nLaunching GUI…")
    SimulationGUI(dem, cellsize)

    print("\n" + "=" * 68)
    print("  DONE — GIF saved to Results/animations/")
    print("=" * 68)