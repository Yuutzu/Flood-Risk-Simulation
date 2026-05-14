"""
=============================================================================
 JADE VALLEY SUBDIVISION — ANIMATED FLOOD SIMULATION
 Davao City, Philippines
=============================================================================
 Interactive animated simulation showing real-time flood progression over
 the Jade Valley terrain using D8 flow routing, river overflow via BFS
 dilation, and Green-Ampt infiltration.

 Inputs: rainfall amount, storm scenario, duration, timestep, wind speed
 and direction, soil saturation, drainage capacity.

 Playback controls: Play/Pause, frame scrubber, speed slider, step buttons.
 Export: Save GIF to Results/animations/, Save CSV time-series data.
=============================================================================
"""

import csv
import heapq
import io
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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Button, Slider

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252 and crash on print() with characters like
# ≈ → that appear in this module's progress messages. Reconfigure stdout and
# stderr to UTF-8 once at import time so every print works on every host.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# ── Optional libraries ───────────────────────────────────────────────────────
try:
    import rasterio
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

try:
    import ezdxf
    EZDXF_OK = True
except ImportError:
    EZDXF_OK = False

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
    import contextily as ctx
    CTX_OK = True
except ImportError:
    CTX_OK = False

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Map Topography"
TIF_FILE    = DATA_DIR / "3D" / "JVS_Simulation.tif"
DXF_2D      = DATA_DIR / "2D" / "Jade_Valley_Subdivision_2D_vectorial.dxf"
JPEG_2D     = DATA_DIR / "2D" / "JVS_2D.jpg"
ANIM_DIR    = BASE_DIR / "Results" / "animations"
ANIM_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PREDEFINED TYPHOON / STORM SCENARIOS
# =============================================================================

# Rainfall values calibrated to PAGASA classification for Mindanao:
#   Light    :  0–7.5 mm/hr       Moderate : 7.5–15 mm/hr
#   Heavy    : 15–30 mm/hr        Intense  : 30–60 mm/hr (typhoon range)
# Duration matches how long each event typically persists in Davao City.
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
# DEM LOADING
# =============================================================================

def load_dem() -> tuple:
    """Load JVS_Simulation.tif. Returns (dem_array, cellsize_m)."""
    if not TIF_FILE.exists():
        sys.exit(f"\n[ERROR] DEM file not found:\n  {TIF_FILE}\n"
                 "  Export the GeoTIFF from QGIS first.")
    if not RASTERIO_OK:
        sys.exit("\n[ERROR] rasterio is required.  Run:  pip install rasterio")

    print(f"  Loading DEM: {TIF_FILE.name}")
    with rasterio.open(str(TIF_FILE)) as src:  # type: ignore[possibly-undefined]
        dem = src.read(1).astype(np.float64)
        nodata = src.nodata if src.nodata is not None else -9999.0
        dem[dem == nodata] = np.nan
        transform = src.transform
        cellsize_x = abs(float(transform.a))
        crs = src.crs
        if crs and crs.is_geographic:
            lat = float(src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2)
            cellsize = cellsize_x * 111320 * abs(np.cos(np.radians(lat)))
        else:
            cellsize = cellsize_x

    # Fill NaN with nearest valid elevation
    nan_mask = np.isnan(dem)
    if nan_mask.any() and SCIPY_OK:
        indices = distance_transform_edt(  # type: ignore[possibly-undefined]
            nan_mask, return_distances=False, return_indices=True)
        rows = np.asarray(indices[0], dtype=int)[nan_mask]   # type: ignore[index]
        cols = np.asarray(indices[1], dtype=int)[nan_mask]   # type: ignore[index]
        dem[nan_mask] = dem[rows, cols]
    elif nan_mask.any():
        dem[nan_mask] = float(np.nanmean(dem))

    print(f"  Grid    : {dem.shape[0]} rows × {dem.shape[1]} cols")
    print(f"  Cell    : ≈{cellsize:.1f} m")
    print(f"  Elev.   : {dem.min():.1f} – {dem.max():.1f} m")
    return dem, round(cellsize, 2)


# =============================================================================
# HYDROLOGICAL PREPROCESSING
# =============================================================================

def _fill_depressions(dem: np.ndarray) -> np.ndarray:
    """Priority-queue sink filling (Wang & Liu 2006)."""
    rows, cols = dem.shape
    filled   = dem.copy()
    visited  = np.zeros((rows, cols), dtype=bool)
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


def _d8_flow_direction(dem: np.ndarray, cell_w: float, cell_h: float) -> np.ndarray:
    """D8 steepest-descent flow direction (returns index 0–7)."""
    rows, cols = dem.shape
    d8 = [(-1,-1, float(np.sqrt(cell_w**2+cell_h**2))),
          (-1, 0, cell_h), (-1, 1, float(np.sqrt(cell_w**2+cell_h**2))),
          ( 0,-1, cell_w), ( 0, 1, cell_w),
          ( 1,-1, float(np.sqrt(cell_w**2+cell_h**2))),
          ( 1, 0, cell_h), ( 1, 1, float(np.sqrt(cell_w**2+cell_h**2)))]
    fdir = np.zeros((rows, cols), dtype=np.int8)
    for flat_idx in np.argsort(-dem.ravel()):
        r, c = divmod(int(flat_idx), cols)
        best_grad = 0.0
        best_d = 0
        for d, (dr, dc, dist) in enumerate(d8):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                g = (dem[r, c] - dem[nr, nc]) / dist
                if g > best_grad:
                    best_grad = g
                    best_d = d
        fdir[r, c] = best_d
    return fdir


def _flow_accumulation(fdir: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """Flow accumulation via topological sort."""
    rows, cols = dem.shape
    accum = np.ones((rows, cols), dtype=np.float32)
    d8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for flat_idx in np.argsort(dem.ravel())[::-1]:
        r, c = divmod(int(flat_idx), cols)
        dr, dc = d8[int(fdir[r, c])]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            accum[nr, nc] += accum[r, c]
    return accum


def build_stream_mask(accum: np.ndarray, pct: float = 92.0) -> np.ndarray:
    """Stream cells = top X% of flow accumulation."""
    thresh = float(np.percentile(accum, pct))
    return accum >= thresh


# =============================================================================
# FLOOD SIMULATION ENGINE
# =============================================================================

class FloodSimulation:
    """
    Two-layer terrain-aware flood simulation.

    rain_water  — water from direct rainfall runoff (cyan → orange colormap)
    river_water — water from river overflow (solid blue mask)

    Physics:
      • D8 flow routing concentrates rain in natural drainage channels
      • Saturation model: soil absorbs less as it wets
      • River level rises with rain; once above bank, BFS dilation floods land
      • Infiltration (Green-Ampt inspired) and channel drainage
    """

    def __init__(self, dem: np.ndarray, cellsize: float,
                 soil_saturation_pct: float = 30.0,
                 drainage_capacity_mmhr: float = 5.0):
        self.dem_raw = dem.copy()
        self.dem     = _fill_depressions(dem)        # routing surface
        self.rows, self.cols = dem.shape
        self.cell   = cellsize

        self.rain_water  = np.zeros_like(dem)
        self.river_water = np.zeros_like(dem)
        self.river_level = self.dem.copy()
        self.rainfall_accumulated = 0.0

        self.drainage_capacity = drainage_capacity_mmhr
        init_sat = float(np.clip(soil_saturation_pct / 100.0, 0.0, 1.0))
        self.saturation = np.full_like(dem, init_sat)

        print("  Preprocessing terrain (slope / flow accumulation)…")
        fdir = _d8_flow_direction(self.dem, cellsize, cellsize)
        self.fdir  = fdir
        self.accum = _flow_accumulation(fdir, self.dem)
        self.streams = build_stream_mask(self.accum)

        # Normalised elevation and flow-weight arrays used in routing and infiltration.
        e = self.dem
        self.elev_norm  = (e - e.min()) / (e.max() - e.min() + 1e-10)
        fa_log = np.log1p(self.accum)
        self.flow_weight = fa_log / (fa_log.max() + 1e-10)

        self.slope = np.hypot(*np.gradient(self.dem, cellsize, cellsize))
        slope_n = self.slope / (self.slope.max() + 1e-10)

        # Runoff coefficient: fraction of rain that becomes surface flow.
        # Flat areas (slope~0): 0.30 — urban mix, some infiltration possible
        # Steep slopes (slope_n=1): 0.80 — impermeable runoff on steep hillsides
        # This prevents flat low cells from routing 45% of every raindrop
        # straight into drainage channels, which caused over-flooding under light rain.
        self.runoff_coeff = np.clip(0.30 + 0.50 * slope_n, 0.25, 0.82)

        # Infiltration capacity (Green-Ampt style):
        # Increased base from 0.3 to 1.5 mm so dry soil genuinely absorbs
        # light rain before it becomes runoff. Remains depth-scaled by elev_norm
        # so low-lying areas (already wet) infiltrate less.
        self.max_inf = 1.5 + 1.8 * self.elev_norm * (1.0 - init_sat)

        # River channel is defined as cells in the top 8 percent of flow accumulation.
        self.river_mask = self.accum >= float(np.percentile(self.accum, 92))

        # For each river cell, find the minimum elevation among non-river neighbours.
        # This value is used as the bank crest height for overflow calculation.
        nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        self.bank_elev = np.full((self.rows, self.cols), np.inf)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.river_mask[r, c]:
                    for dr, dc in nbrs:
                        nr, nc = r+dr, c+dc
                        if (0 <= nr < self.rows and 0 <= nc < self.cols
                                and not self.river_mask[nr, nc]):
                            be = self.dem_raw[nr, nc]
                            if be < self.bank_elev[r, c]:
                                self.bank_elev[r, c] = be
        valid = self.river_mask & np.isfinite(self.bank_elev)
        if valid.any():
            self.bank_elev[~valid & self.river_mask] = \
                self.dem_raw[~valid & self.river_mask]
            self.river_level[valid] = self.bank_elev[valid]
            self.flood_base_wse = float(np.nanpercentile(self.bank_elev[valid], 50))
        else:
            self.flood_base_wse = float(self.dem.min())
        self.river_level_init = self.river_level.copy()

        # The BFS flood front is initialised at the river channel and expands outward each timestep.
        self.bfs_front = self.river_mask.copy()

        if SCIPY_OK:
            self.river_dist = distance_transform_edt(~self.river_mask)  # type: ignore[possibly-undefined]
        else:
            self.river_dist = np.zeros_like(dem)

    @property
    def water_depth(self):
        return self.rain_water + self.river_water

    # ── Rainfall ─────────────────────────────────────────────────────────────

    def add_rainfall(self, total_mm: float, wind_map=None):
        mod = self.runoff_coeff.copy()
        if wind_map is not None:
            mod *= wind_map
        self.rain_water += (total_mm / 1000.0) * mod
        self.rainfall_accumulated += total_mm / 1000.0

    # ── River overflow ────────────────────────────────────────────────────────

    def apply_river_overflow(self, rate_mmhr: float, dt_h: float,
                             intensity: float = 1.0):
        """
        Improved: Overflow and blue cell shading are now highly scenario-sensitive.
        Each scenario (light, moderate, heavy, typhoon) produces a distinct overflow pattern.
        """
        if not self.river_mask.any():
            return

        rain_m   = (rate_mmhr / 1000.0) * dt_h * intensity
        accum_mm = self.rainfall_accumulated * 1000.0

        # Overflow is suppressed below the accumulated rainfall threshold for each PAGASA category.
        if accum_mm < 30.0:
            return
        if accum_mm < 50.0 and rate_mmhr < 10:
            ramp = np.clip((accum_mm - 30.0) / 20.0, 0.0, 1.0) * 0.3
        elif accum_mm < 80.0 and rate_mmhr < 18:
            ramp = np.clip((accum_mm - 50.0) / 30.0, 0.0, 1.0) * 0.6 + 0.3
        elif accum_mm < 120.0:
            ramp = np.clip((accum_mm - 80.0) / 40.0, 0.0, 1.0) * 0.7 + 0.9
        else:
            ramp = 1.0

        # The overflow multiplier scales with scenario severity and local flow concentration.
        rise_mult = ramp * (4.0 + 10.0 * self.flow_weight[self.river_mask])
        if rate_mmhr > 20:
            rise_mult *= 1.5
        self.river_level[self.river_mask] += rain_m * rise_mult

        rise = self.river_level[self.river_mask] - self.river_level_init[self.river_mask]
        flood_rise = float(np.percentile(rise, 90))
        if flood_rise <= 0:
            return

        # Water surface elevation is capped at a level appropriate for each scenario severity.
        if accum_mm < 50:
            wse_cap = 0.5
        elif accum_mm < 80:
            wse_cap = 1.2
        elif accum_mm < 120:
            wse_cap = 2.5
        else:
            wse_cap = 4.0
        eff_wse = self.flood_base_wse + min(flood_rise, wse_cap)

        # BFS dilation hop count increases with scenario severity.
        if accum_mm < 50:
            HOPS = 1
        elif accum_mm < 80:
            HOPS = 2
        elif accum_mm < 120:
            HOPS = 4
        else:
            HOPS = 6
        can_flood = (self.dem_raw < eff_wse) & ~self.river_mask
        struct    = np.ones((3, 3), dtype=bool)
        if SCIPY_OK:
            for _ in range(HOPS):
                exp = binary_dilation(self.bfs_front, structure=struct)  # type: ignore[possibly-undefined]
                new = exp & can_flood & ~self.bfs_front  # type: ignore[operator]
                if not new.any():
                    break
                self.bfs_front |= new
        else:
            for _ in range(HOPS):
                f = self.bfs_front
                shifted = (np.roll(f,1,0)|np.roll(f,-1,0)|
                           np.roll(f,1,1)|np.roll(f,-1,1)|
                           np.roll(np.roll(f,1,0),1,1)|
                           np.roll(np.roll(f,1,0),-1,1)|
                           np.roll(np.roll(f,-1,0),1,1)|
                           np.roll(np.roll(f,-1,0),-1,1))
                new = shifted & can_flood & ~f
                if not new.any():
                    break
                self.bfs_front = f | new

        land = self.bfs_front & ~self.river_mask
        # Blue cell shading depth is proportional to scenario severity.
        blue_depth = np.clip(eff_wse - self.dem_raw, 0.0, wse_cap)
        blue_depth[self.river_mask] = 0.0
        delta = (blue_depth[land] - self.river_water[land]) * (0.35 + 0.25 * ramp)
        self.river_water[land] = np.maximum(self.river_water[land] + delta, 0.0)

        draining = ~self.bfs_front & ~self.river_mask & (self.river_water > 0)
        self.river_water[draining] *= 0.85
        np.clip(self.river_water, 0.0, 4.0, out=self.river_water)

    # ── Infiltration ──────────────────────────────────────────────────────────

    def apply_infiltration(self, dt_h: float):
        sat_inc = np.minimum(0.12 * dt_h, 1.0 - self.saturation)
        sat_inc *= (0.5 + 0.5 * (1.0 - self.elev_norm))
        self.saturation = np.clip(self.saturation + sat_inc, 0.0, 1.0)
        inf_m = (self.max_inf * (1.0 - self.saturation) / 1000.0) * dt_h
        self.rain_water  = np.maximum(self.rain_water  - inf_m,       0.0)
        self.river_water = np.maximum(self.river_water - inf_m * 0.5, 0.0)

    # ── Flow routing (rain layer only) ────────────────────────────────────────

    def route_water(self, iters: int = 12):
        d8_dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for _ in range(iters):
            wse = self.dem + self.rain_water
            dw  = np.zeros_like(self.rain_water)
            for dr, dc in d8_dirs:
                r0 = max(0,-dr); r1 = self.rows - max(0,dr)
                c0 = max(0,-dc); c1 = self.cols - max(0,dc)
                nr0, nr1 = r0+dr, r1+dr
                nc0, nc1 = c0+dc, c1+dc
                diff = wse[r0:r1,c0:c1] - wse[nr0:nr1,nc0:nc1]
                slope_n = self.slope[r0:r1,c0:c1] / (self.slope.max()+1e-10)
                tf   = np.clip(0.05 + 0.45*slope_n, 0.05, 0.50)
                flow = np.clip(diff * tf, 0,
                               self.rain_water[r0:r1,c0:c1] * 0.35)
                dw[r0:r1,c0:c1]    -= flow
                dw[nr0:nr1,nc0:nc1] += flow
            self.rain_water = np.maximum(self.rain_water + dw, 0.0)
        np.clip(self.rain_water, 0.0, 5.0, out=self.rain_water)

    # ── Drainage ──────────────────────────────────────────────────────────────

    def apply_drainage(self, dt_h: float):
        """
        Physically calibrated drainage.

        Flat low-elevation cells (where Jade Valley floods):
          - Urban stormwater drain floor = 8 mm/hr minimum
          - Scales up with slope so sloped cells drain faster
          - Scales up with drainage_capacity parameter (user control)

        Channel cells get a large bonus so routed water exits quickly —
        the river channel should not permanently hold surface runoff.

        This ensures:
          Light rain  → drains fully within 2–3 hrs after rain stops
          Moderate    → drains within 4–6 hrs (some residual in low spots)
          Heavy       → significant standing water persists 6–12 hrs
          Typhoon     → widespread flooding, very slow to drain
        """
        slope_n      = self.slope / (self.slope.max() + 1e-10)
        channel_rate = 60.0 * self.flow_weight ** 2   # channel bonus (mm/hr)

        # Base rate: minimum 8 mm/hr (urban drain floor) + slope contribution
        # + user-configured drainage_capacity
        base_rate = (8.0
                     + self.drainage_capacity * (0.5 + 1.5 * slope_n)
                     + channel_rate)

        drain_m = (base_rate / 1000.0) * dt_h
        self.rain_water  = np.maximum(self.rain_water  - drain_m,       0.0)
        self.river_water = np.maximum(self.river_water - drain_m * 0.3, 0.0)

    # ── Full timestep ─────────────────────────────────────────────────────────

    def step(self, rate_mmhr: float, dt_h: float,
             intensity: float = 1.0, wind_map=None):
        rain_mm = rate_mmhr * dt_h * intensity
        # Rain falls FIRST → routes into channels → channels rise → overflow
        self.add_rainfall(rain_mm, wind_map)
        self.route_water(iters=12)
        self.apply_river_overflow(rate_mmhr, dt_h, intensity)
        self.apply_infiltration(dt_h)
        self.apply_drainage(dt_h)


# =============================================================================
# WIND MAP
# =============================================================================

def wind_rainfall_map(dem: np.ndarray, speed_kmh: float,
                      direction_deg: float) -> np.ndarray:
    if speed_kmh < 1.0:
        return np.ones_like(dem, dtype=float)
    wind_rad = np.radians(direction_deg)
    wx, wy   =  np.sin(wind_rad), -np.cos(wind_rad)
    dy, dx   = np.gradient(dem)
    windward = wx*dx + wy*dy
    wn = windward / (np.abs(windward).max() + 1e-10)
    intensity = min(speed_kmh / 60.0, 1.0) * 0.35
    return np.clip(1.0 + intensity * wn, 0.40, 1.80).astype(float)


# =============================================================================
# OSM MAP BACKGROUND
# =============================================================================


def render_jpeg_background(dem: np.ndarray) -> np.ndarray:
    """
    Load JVS_2D.jpg at high resolution for a sharp background.
    Returns RGB float32 array in [0, 1].
    Falls back to DXF or elevation shading if unavailable.
    """
    if JPEG_2D.exists() and PIL_OK:
        try:
            pil = PILImage.open(str(JPEG_2D)).convert("RGB")  # type: ignore[possibly-undefined]
            # Keep original resolution (or cap at 2048 px on longest side)
            max_side = max(pil.size)
            if max_side > 2048:
                scale = 2048 / max_side
                new_w = int(pil.size[0] * scale)
                new_h = int(pil.size[1] * scale)
                pil = pil.resize((new_w, new_h), PILImage.Resampling.LANCZOS)  # type: ignore[possibly-undefined]
            img = np.array(pil, dtype=np.float32) / 255.0
            print(f"  JPEG background loaded at {img.shape[1]}×{img.shape[0]} px")
            return img
        except Exception as e:
            print(f"  JPEG load failed ({e}) — trying DXF fallback")
    return render_dxf_background(dem)


def render_osm_background(dem: np.ndarray) -> np.ndarray:
    """
    Fetch OpenStreetMap tiles aligned to TIF bounds and resize to DEM pixel
    dimensions. Returns RGB float32 array (H×W×3) in [0, 1].
    Falls back to DXF or elevation shading if unavailable.
    """
    rows, cols = dem.shape
    if CTX_OK and RASTERIO_OK:
        try:
            from rasterio.warp import transform_bounds
            with rasterio.open(str(TIF_FILE)) as src:  # type: ignore[possibly-undefined]
                crs = src.crs
                b   = src.bounds
                if crs and not crs.is_geographic:
                    west, south, east, north = transform_bounds(
                        crs, "EPSG:4326",
                        b.left, b.bottom, b.right, b.top)
                else:
                    west, south, east, north = b.left, b.bottom, b.right, b.top
            img, _ = ctx.bounds2img(  # type: ignore[possibly-undefined, arg-type, call-arg]
                west, south, east, north,
                zoom=15,  # type: ignore[arg-type]
                ll=True,
                source=ctx.providers.OpenStreetMap.Mapnik)  # type: ignore[attr-defined]
            if PIL_OK:
                pil = PILImage.fromarray(img).convert("RGB")  # type: ignore[possibly-undefined]
                pil = pil.resize((cols, rows), PILImage.Resampling.LANCZOS)  # type: ignore[possibly-undefined]
                osm = np.array(pil, dtype=np.float32) / 255.0
            else:
                # manual resize via numpy
                osm_raw = img[:, :, :3].astype(np.float32) / 255.0
                from scipy.ndimage import zoom as nd_zoom
                zy = rows / osm_raw.shape[0]
                zx = cols / osm_raw.shape[1]
                osm_raw_list: list = [nd_zoom(osm_raw[:, :, c], (zy, zx)) for c in range(3)]
                osm = np.stack(osm_raw_list, axis=2).astype(np.float32)  # type: ignore[arg-type]
            print("  OSM map background fetched and aligned to TIF extent")
            return osm
        except Exception as e:
            print(f"  OSM fetch failed ({e}) — trying DXF fallback")
    return render_dxf_background(dem)


# =============================================================================
# DXF MAP BACKGROUND  (fallback)
# =============================================================================

_LAYER_STYLE = {
    'TPX_BUILDINGS'                   : ('#444444', None,      1.4, 0.95, 8),
    'TPX_BUILDINGS_HATCH'             : ('#555555', '#D8D8D8', 0.6, 0.95, 7),
    'TPX_BUILDINGS_SHADOWS_HATCH'     : (None,      '#B0B0B0', 0.0, 0.45, 6),
    'TPX_BUILDINGS_SHADOWS_OUTLINES'  : ('#999999', None,      0.5, 0.45, 6),
    'TPX_ROADS_AXES'                  : ('#777777', None,      1.8, 0.92, 4),
    'TPX_ROADS_CONTOURS'              : ('#666666', None,      0.9, 0.92, 4),
    'TPX_ROADS_HATCH'                 : ('#AAAAAA', '#F2F2F2', 0.5, 0.88, 2),
    'TPX_WATERWAYS'                   : ('#1A6EA8', None,      1.4, 0.96, 9),
    'TPX_WATERWAYS_HATCH'             : ('#1A6EA8', '#7EC8E3', 0.6, 0.93, 8),
    'TPX_VEGETATION_GREEN_SPACES'     : ('#1E8449', None,      1.0, 0.93, 5),
    'TPX_VEGETATION_GREEN_SPACES_HATCH': ('#1E8449','#A9DFBF', 0.4, 0.90, 4),
    'TPX_RELIEF_CONTOUR_LINES'        : ('#BBBBBB', None,      0.7, 0.75, 1),
    'TPX_FRAME_PROJECT'               : ('#CCCCCC', None,      0.5, 0.40, 1),
}


def render_dxf_background(dem: np.ndarray) -> np.ndarray:
    """Render the 2D DXF as a clean map image (RGB float32 H×W×3)."""
    fallback_fn = _elevation_fallback

    if not EZDXF_OK or not PIL_OK:
        return fallback_fn(dem)
    if not DXF_2D.exists():
        return fallback_fn(dem)

    try:
        import io as _io
        from matplotlib.patches import Polygon as MplPoly

        doc = ezdxf.readfile(str(DXF_2D))        # type: ignore[attr-defined]
        msp = doc.modelspace()
        all_x, all_y = [], []
        for ent in msp:
            for pt in _dxf_points(ent):
                all_x.append(pt[0]); all_y.append(pt[1])
        if not all_x:
            return fallback_fn(dem)

        pad = max(max(all_x)-min(all_x), max(all_y)-min(all_y)) * 0.01
        x0, x1 = min(all_x)-pad, max(all_x)+pad
        y0, y1 = min(all_y)-pad, max(all_y)+pad

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect('equal'); ax.axis('off')
        fig.subplots_adjust(0, 0, 1, 1)

        for layer_name, style in sorted(_LAYER_STYLE.items(),
                                         key=lambda kv: kv[1][4]):
            ec, fc, lw, alpha, zo = style
            for ent in msp:
                if ent.dxf.layer != layer_name:
                    continue
                if ent.dxftype() == 'LWPOLYLINE':
                    pts = list(ent.get_points())  # type: ignore[attr-defined]
                    if len(pts) < 2:
                        continue
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    closed = ent.closed  # type: ignore[attr-defined]
                    if fc and closed and len(pts) >= 3:
                        ax.add_patch(MplPoly(
                            list(zip(xs, ys)), closed=True,
                            facecolor=fc, edgecolor=ec or 'none',
                            linewidth=lw, alpha=alpha, zorder=zo))
                    else:
                        if closed:
                            xs.append(xs[0]); ys.append(ys[0])
                        if ec:
                            ax.plot(xs, ys, color=ec, linewidth=lw,
                                    alpha=alpha, zorder=zo)
                elif ent.dxftype() == 'HATCH' and fc:
                    for path in ent.paths:  # type: ignore[attr-defined]
                        try:
                            verts = getattr(path, 'vertices', None)
                            if verts is not None:
                                pts2 = [(v[0], v[1]) for v in verts]
                                if len(pts2) >= 3:
                                    ax.add_patch(MplPoly(
                                        pts2, closed=True,
                                        facecolor=fc, edgecolor=ec or 'none',
                                        linewidth=lw, alpha=alpha, zorder=zo))
                        except Exception:
                            pass

        buf = _io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    pad_inches=0, facecolor='white')
        buf.seek(0)
        pil = PILImage.open(buf).convert('RGB').resize((800, 800), PILImage.Resampling.LANCZOS)  # type: ignore[possibly-undefined]
        img = np.array(pil, dtype=np.float32) / 255.0
        plt.close(fig)
        print("  DXF map background rendered  (800×800 px)")
        return img
    except Exception as e:
        print(f"  DXF render skipped ({e}) — using elevation background")
        return fallback_fn(dem)


def _dxf_points(ent) -> list:
    pts = []
    try:
        if ent.dxftype() == 'LWPOLYLINE':
            for p in ent.get_points():
                pts.append((p[0], p[1]))
        elif ent.dxftype() == 'LINE':
            pts += [(ent.dxf.start.x, ent.dxf.start.y),
                    (ent.dxf.end.x,   ent.dxf.end.y)]
        elif ent.dxftype() == 'HATCH':
            for path in ent.paths:
                if hasattr(path, 'vertices'):
                    for v in path.vertices:
                        pts.append((v[0], v[1]))
    except Exception:
        pass
    return pts


def _elevation_fallback(dem: np.ndarray) -> np.ndarray:
    dem_n = (dem - dem.min()) / (dem.max() - dem.min() + 1e-10)
    shade = 0.80 + 0.20 * dem_n
    return np.stack([shade, shade, shade], axis=2).astype(np.float32)


# =============================================================================
# COLORMAPS
# =============================================================================

def _rain_cmap():
    colors = [
        (0.00, 0.00, 0.00, 0.00),
        (0.40, 0.95, 0.95, 0.30),
        (0.00, 0.78, 0.78, 0.42),
        (0.00, 0.60, 0.60, 0.50),
        (0.50, 0.90, 0.20, 0.55),
        (1.00, 1.00, 0.00, 0.60),
        (1.00, 0.55, 0.00, 0.65),
        (1.00, 0.20, 0.00, 0.70),
        (0.80, 0.00, 0.00, 0.70),
    ]
    return LinearSegmentedColormap.from_list('rain', colors)


def _river_cmap():
    # Continuous, nonlinear blue colormap for water depth
    # More sensitive to shallow/moderate flooding
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
# STORM INTENSITY PATTERN
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
    elif pattern == 'burst':
        return 0.30 + 1.50 * float(np.exp(-((t - 0.45) ** 2) / 0.055))
    elif pattern == 'decreasing':
        return max(1.60 - 1.40 * t, 0.20)
    return 1.0


# =============================================================================
# ACADEMIC HELPERS
# =============================================================================

def _estimate_return_period(total_mm: float, duration_h: float) -> str:
    """Rough return-period classification based on PAGASA IDF data for Davao City.
    Intensity thresholds derived from Bureau of Soils and Water Management (BSWM)
    regional frequency analysis for Southern Mindanao."""
    rate = total_mm / max(duration_h, 0.1)
    if rate < 7.5:   return "< 2-year event"
    if rate < 15.0:  return "2 – 5 year event"
    if rate < 25.0:  return "5 – 10 year event"
    if rate < 40.0:  return "10 – 25 year event"
    if rate < 60.0:  return "25 – 50 year event"
    return               "50 – 100 year event"


def _export_sim_csv(out_path: str, times_list: list, elapsed_list: list,
                    rate_frames: list, stats: dict, timestep_min: int) -> None:
    """Write full time-series simulation data to CSV for academic reporting."""
    fields = [
        'time', 'elapsed_min', 'rain_intensity_mmhr',
        'rain_fallen_mm', 'flooded_pct', 'flooded_ha',
        'max_depth_mm', 'river_pct', 'max_river_mm', 'runoff_vol_m3',
    ]
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t in enumerate(times_list):
            w.writerow({
                'time':                 t,
                'elapsed_min':          elapsed_list[i],
                'rain_intensity_mmhr':  f"{rate_frames[i]:.2f}",
                'rain_fallen_mm':       f"{stats['rain_mm'][i]:.2f}",
                'flooded_pct':          f"{stats['flooded_pct'][i]:.2f}",
                'flooded_ha':           f"{stats['flooded_ha'][i]:.3f}",
                'max_depth_mm':         f"{stats['max_depth_mm'][i]:.1f}",
                'river_pct':            f"{stats['river_pct'][i]:.2f}",
                'max_river_mm':         f"{stats['max_river_mm'][i]:.1f}",
                'runoff_vol_m3':        f"{stats['runoff_vol_m3'][i]:.1f}",
            })


# =============================================================================
# MAIN ANIMATION FUNCTION
# =============================================================================

def run_simulation(dem, cellsize, rainfall_mm, duration_h,
                   timestep_min, start_time_str, wind_speed, wind_dir,
                   soil_sat_pct, drain_cap, pattern, scenario_name):
    """Run simulation; build 3-panel interactive viewer (map + stats + hydrograph); export CSV."""

    rate_mmhr     = rainfall_mm / duration_h
    dt_h          = timestep_min / 60.0
    num_frames    = int(np.ceil(duration_h / dt_h))
    total_area_m2 = dem.shape[0] * dem.shape[1] * cellsize ** 2

    print(f"\n  Scenario  : {scenario_name}")
    print(f"  Rainfall  : {rainfall_mm:.0f} mm in {duration_h:.1f} h  "
          f"({rate_mmhr:.1f} mm/hr, {pattern})")
    print(f"  Timestep  : {timestep_min} min  \u2192  {num_frames} frames")
    print(f"  Wind      : {wind_speed:.0f} km/h from {wind_dir:.0f}\u00b0")
    print(f"  Soil      : {soil_sat_pct:.0f}%  Drain: {drain_cap:.1f} mm/hr")

    sim  = FloodSimulation(dem, cellsize,
                           soil_saturation_pct    = soil_sat_pct,
                           drainage_capacity_mmhr = drain_cap)
    wmap = wind_rainfall_map(dem, wind_speed, wind_dir)

    # Pre-compute per-frame rainfall intensity for the hydrograph panel
    rate_frames = [rate_mmhr * _intensity_factor(i, num_frames, pattern)
                   for i in range(num_frames)]

    # ── Run all timesteps ──────────────────────────────────────────────────────
    rain_frames  = []
    river_frames = []
    times_list   = []
    elapsed_list = []
    stats: dict  = {
        "rain_mm":       [],
        "flooded_pct":   [],
        "flooded_ha":    [],
        "river_pct":     [],
        "max_depth_mm":  [],
        "max_river_mm":  [],
        "runoff_vol_m3": [],
    }

    sh, sm = map(int, start_time_str.split(":"))
    cur    = datetime.now().replace(hour=sh, minute=sm, second=0)
    print("\n  Simulating\u2026")

    for fr in range(num_frames):
        inten = _intensity_factor(fr, num_frames, pattern)
        sim.step(rate_mmhr, dt_h, intensity=inten, wind_map=wmap)

        rain_frames .append(sim.rain_water .copy())
        river_frames.append(sim.river_water.copy())
        times_list  .append(cur.strftime("%H:%M"))
        elapsed_list.append(fr * timestep_min)

        total        = sim.rain_water + sim.river_water
        rain_acc_mm  = float(sim.rainfall_accumulated * 1000)
        flooded_pct  = float(np.sum(total > 0.01) / total.size * 100)
        flooded_ha   = flooded_pct / 100.0 * total_area_m2 / 10000.0
        river_pct    = float(np.sum(sim.river_water > 0.005) / total.size * 100)
        max_depth_mm = float(total.max() * 1000)
        max_river_mm = float(sim.river_water.max() * 1000)
        runoff_vol   = rain_acc_mm / 1000.0 * float(sim.runoff_coeff.mean()) * total_area_m2

        stats["rain_mm"]      .append(rain_acc_mm)
        stats["flooded_pct"]  .append(flooded_pct)
        stats["flooded_ha"]   .append(flooded_ha)
        stats["river_pct"]    .append(river_pct)
        stats["max_depth_mm"] .append(max_depth_mm)
        stats["max_river_mm"] .append(max_river_mm)
        stats["runoff_vol_m3"].append(runoff_vol)

        if (fr + 1) % max(1, num_frames // 8) == 0 or fr == 0:
            print(f"    [{fr+1:3d}/{num_frames}]  {times_list[-1]}  "
                  f"rain={rain_acc_mm:.0f} mm  "
                  f"flooded={flooded_pct:.1f}%  "
                  f"maxdepth={max_depth_mm:.0f} mm")
        cur += timedelta(minutes=timestep_min)

    # ── Academic metrics (computed once) ───────────────────────────────────────
    ret_period        = _estimate_return_period(rainfall_mm, duration_h)
    peak_intensity_ms = max(rate_frames) / (1000.0 * 3600.0)
    peak_Q_m3s        = 0.55 * peak_intensity_ms * total_area_m2   # Rational method C=0.55

    # ── Background and colormaps ───────────────────────────────────────────────
    print("\n  Rendering map background\u2026")
    bg         = render_jpeg_background(dem)
    cmap_rain  = _rain_cmap()
    cmap_river = _river_cmap()

    DARK     = "#0D1117"
    PANEL    = "#161B22"
    TCLR     = "#E6EDF3"
    ACC      = "#4FC3F7"
    CHART_BG = "#0A0F18"

    # ── Figure: 3-panel layout (map | stats | hydrograph) ─────────────────────
    import matplotlib as _mpl
    _backend = _mpl.get_backend()
    _screen_w, _screen_h = 1600, 900
    _fig_w,    _fig_h    = 1300, 750
    _dpi = 100
    try:
        import tkinter as _tk
        _r = _tk.Tk(); _r.withdraw()
        _screen_w = _r.winfo_screenwidth()
        _screen_h = _r.winfo_screenheight()
        _r.destroy()
        _fig_w = min(int(_screen_w * 0.88), 1500)
        _fig_h = min(int(_screen_h * 0.88), 860)
    except Exception:
        pass
    fig = plt.figure(figsize=(_fig_w / _dpi, _fig_h / _dpi), facecolor=DARK, dpi=_dpi)
    fig.suptitle(
        f"JADE VALLEY SUBDIVISION  \u2014  FLOOD SIMULATION  |  {scenario_name}",
        fontsize=14, fontweight="bold", color=TCLR, y=0.983)

    ax_map   = fig.add_axes((0.03, 0.13, 0.61, 0.83))
    ax_stats = fig.add_axes((0.67, 0.38, 0.31, 0.58))
    ax_chart = fig.add_axes((0.67, 0.13, 0.31, 0.21), facecolor=CHART_BG)

    ax_map  .set_facecolor("black")
    ax_stats.set_facecolor(PANEL)
    ax_stats.axis("off")

    H, W = dem.shape
    ext  = (0, W, H, 0)

    # Map: base layers
    ax_map.imshow(bg, extent=ext, aspect="auto", zorder=1, interpolation="bilinear")
    im_rain = ax_map.imshow(
        rain_frames[0] * 1000,
        cmap=cmap_rain, vmin=0, vmax=600,
        extent=ext, aspect="auto", zorder=2, interpolation="bilinear", alpha=0.45)
    total_water0 = rain_frames[0] + river_frames[0]
    im_river = ax_map.imshow(
        np.clip(np.power(total_water0 / 1.1, 0.65), 0, 1),
        cmap=cmap_river, vmin=0, vmax=1,
        extent=ext, aspect="auto", zorder=3, interpolation="nearest", alpha=0.62)

    # Elevation contour lines (faint topographic reference)
    try:
        ax_map.contour(np.flipud(dem),
                       levels=np.linspace(dem.min(), dem.max(), 14),
                       colors="white", alpha=0.10, linewidths=0.4, zorder=4)
    except Exception:
        pass

    # Stream-network overlay (permanent sky-blue band)
    strm_rgba = np.zeros((H, W, 4), dtype=np.float32)
    strm_rgba[sim.streams, 0] = 0.15
    strm_rgba[sim.streams, 2] = 0.90
    strm_rgba[sim.streams, 3] = 0.65
    ax_map.imshow(strm_rgba, extent=ext, aspect="auto", zorder=5, interpolation="nearest")

    # Scale bar (targets ~50 m physical length)
    sc_cells = max(3, int(round(50.0 / cellsize)))
    sc_m     = sc_cells * cellsize
    bx0, bx1 = W * 0.05, W * 0.05 + sc_cells
    by, bh   = H * 0.930, H * 0.007
    ax_map.fill_between([bx0, bx1], [by - bh] * 2, [by + bh] * 2, color="white", zorder=15)
    ax_map.text((bx0 + bx1) / 2, by + bh * 3.0, f"{sc_m:.0f} m",
                color="white", fontsize=7, ha="center", va="bottom",
                fontweight="bold", zorder=15)

    # North arrow
    nx, ny0, ny1 = W * 0.938, H * 0.115, H * 0.060
    ax_map.annotate("", xy=(nx, ny1), xytext=(nx, ny0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2.0), zorder=15)
    ax_map.text(nx, ny1 - H * 0.014, "N", color="white",
                fontsize=9, ha="center", va="bottom", fontweight="bold", zorder=15)

    # Map legend
    leg_h = [
        Patch(facecolor="#29B6F6", alpha=0.80, label="Stream channel"),
        Patch(facecolor="#1E90FF", alpha=0.75, label="River overflow"),
        Patch(facecolor="#00FFCC", alpha=0.55, label="Runoff \u2264 30 mm"),
        Patch(facecolor="#FFFF00", alpha=0.60, label="Runoff \u2264 100 mm"),
        Patch(facecolor="#FF6600", alpha=0.65, label="Runoff \u2264 300 mm"),
        Patch(facecolor="#CC00FF", alpha=0.70, label="Runoff > 300 mm"),
    ]
    ax_map.legend(handles=leg_h, loc="lower right",
                  facecolor="#0D1117", edgecolor=ACC,
                  labelcolor="#E6EDF3", fontsize=6.5,
                  framealpha=0.88, handlelength=1.2, borderpad=0.7)

    ax_map.set_xlim(0, W); ax_map.set_ylim(H, 0)
    ax_map.tick_params(colors=TCLR, labelsize=7)
    for sp in ax_map.spines.values():
        sp.set_edgecolor("#30363D")

    cbar = fig.colorbar(im_rain, ax=ax_map, orientation="vertical", pad=0.01, shrink=0.74)
    cbar.set_label("Rainfall Runoff Depth (mm)", color=TCLR, fontsize=8)
    cbar.set_ticks([0, 30, 100, 200, 400, 600])
    cbar.ax.set_yticklabels(["Dry", "30", "100", "200", "400", "600+"],
                             color=TCLR, fontsize=7)
    cbar.ax.tick_params(colors=TCLR)

    time_txt = ax_map.text(
        0.015, 0.975, "", transform=ax_map.transAxes,
        fontsize=11, fontweight="bold", color="white", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL,
                  alpha=0.90, edgecolor=ACC, linewidth=1.5))

    stats_txt = ax_stats.text(
        0.04, 0.98, "", fontsize=8.5, family="monospace",
        color=TCLR, va="top", transform=ax_stats.transAxes,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#0D1117",
                  edgecolor=ACC, alpha=0.92, linewidth=1.2))

    # ── Hydrograph panel ───────────────────────────────────────────────────────
    time_axis = [i * timestep_min for i in range(num_frames)]

    ax_chart.bar(time_axis, rate_frames, width=timestep_min * 0.85,
                 color="#29B6F6", alpha=0.55, zorder=2)
    ax_chart.set_xlim(-timestep_min * 0.5, time_axis[-1] + timestep_min)
    ax_chart.set_ylim(0, max(rate_frames) * 1.40 + 1)
    ax_chart.set_xlabel("Elapsed Time (min)", color=TCLR, fontsize=7)
    ax_chart.set_ylabel("Intensity (mm/hr)", color="#29B6F6", fontsize=7)
    ax_chart.tick_params(colors=TCLR, labelsize=6.5)
    for sp in ax_chart.spines.values():
        sp.set_edgecolor("#30363D")
    ax_chart.text(0.015, 0.91, "HYDROGRAPH", transform=ax_chart.transAxes,
                  fontsize=6.5, color="#B0BEC5", fontweight="bold", va="top")

    ax_chart2 = ax_chart.twinx()
    ax_chart2.tick_params(colors="#F97316", labelsize=6.5)
    ax_chart2.set_ylabel("Max Depth (mm)", color="#F97316", fontsize=7)
    ax_chart2.set_ylim(0, max(max(stats["max_depth_mm"]) * 1.35, 10))
    ax_chart2.set_xlim(-timestep_min * 0.5, time_axis[-1] + timestep_min)
    depth_line, = ax_chart2.plot([], [], "-", color="#F97316", linewidth=1.8, zorder=3)
    chart_vline  = ax_chart.axvline(0, color="#FFD740", linewidth=1.3, linestyle="--", zorder=4)

    # ── Widgets ────────────────────────────────────────────────────────────────
    ax_sl_frame = fig.add_axes((0.03, 0.084, 0.61, 0.022), facecolor="#21262D")
    ax_sl_speed = fig.add_axes((0.03, 0.040, 0.23, 0.022), facecolor="#21262D")
    ax_btn_play = fig.add_axes((0.280, 0.018, 0.080, 0.052))
    ax_btn_prev = fig.add_axes((0.366, 0.018, 0.048, 0.052))
    ax_btn_next = fig.add_axes((0.420, 0.018, 0.048, 0.052))
    ax_btn_gif  = fig.add_axes((0.475, 0.018, 0.083, 0.052))
    ax_btn_csv  = fig.add_axes((0.565, 0.018, 0.083, 0.052))

    sl_frame = Slider(ax_sl_frame, "Frame", 0, num_frames - 1,
                      valinit=0, valstep=1, color=ACC)
    sl_speed = Slider(ax_sl_speed, "Speed \u00d7", 0.25, 4.0,
                      valinit=1.0, color="#FFB74D")
    for sl in (sl_frame, sl_speed):
        sl.label.set_color(TCLR); sl.valtext.set_color(TCLR)
        sl.label.set_fontsize(7.5)

    btn_play = Button(ax_btn_play, "Pause",    color="#1B5E20", hovercolor="#2E7D32")
    btn_prev = Button(ax_btn_prev, "\u25c4\u25c4", color="#0D47A1", hovercolor="#1565C0")
    btn_next = Button(ax_btn_next, "\u25ba\u25ba", color="#0D47A1", hovercolor="#1565C0")
    btn_gif  = Button(ax_btn_gif,  "Save GIF", color="#4A148C", hovercolor="#6A1B9A")
    btn_csv  = Button(ax_btn_csv,  "Save CSV", color="#1A3A2A", hovercolor="#1B5E20")
    for b in (btn_play, btn_prev, btn_next, btn_gif, btn_csv):
        b.label.set_color("white"); b.label.set_fontsize(9)

    player = {"playing": True, "frame": 0}
    DIRS = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    compass_lbl = DIRS[int((wind_dir + 11.25) / 22.5) % 16]
    wind_str    = (f"{wind_speed:.0f} km/h from {compass_lbl}"
                   if wind_speed >= 1.0 else "None")

    def _risk_str(max_depth_mm: float, river_pct: float):
        """PAGASA-aligned depth + overflow thresholds."""
        if river_pct > 30 or max_depth_mm > 600:
            return "EVACUATE NOW",                          "#FF1744"
        if river_pct > 18 or max_depth_mm > 300:
            return "MANDATORY EVACUATION",                  "#FF6D00"
        if river_pct > 8  or max_depth_mm > 150:
            return "PRE-EVACUATION ALERT",                  "#FFD740"
        if river_pct > 3  or max_depth_mm > 50:
            return "STANDBY \u2014 prepare to move",       "#69F0AE"
        return             "NORMAL \u2014 monitoring",      "#B0BEC5"

    sep = "─────────────────────────────"

    def draw(fi: int):
        fi = int(fi) % num_frames
        player["frame"] = fi

        rn = rain_frames[fi] * 1000
        im_rain.set_data(rn)
        im_rain.set_clim(0, min(max(float(rn.max()), 30) * 1.3, 600))
        total_w = rain_frames[fi] + river_frames[fi]
        im_river.set_data(np.clip(np.power(total_w / 1.1, 0.65), 0, 1))

        time_txt.set_text(
            f" Time : {times_list[fi]}\n"
            f" Rain : {stats['rain_mm'][fi]:.0f} / {rainfall_mm:.0f} mm\n"
            f" Wind : {wind_str}")

        rsk, rsk_col = _risk_str(stats["max_depth_mm"][fi], stats["river_pct"][fi])

        stats_txt.set_text(
            f"  SCENARIO\n"
            f"  {sep}\n"
            f"  {scenario_name}\n"
            f"\n"
            f"  STORM PARAMETERS\n"
            f"  {sep}\n"
            f"  Total Rainfall  : {rainfall_mm:.0f} mm\n"
            f"  Avg Rate        : {rate_mmhr:.1f} mm/hr\n"
            f"  Peak Rate       : {max(rate_frames):.1f} mm/hr\n"
            f"  Duration        : {duration_h:.1f} hr\n"
            f"  Pattern         : {pattern}\n"
            f"  Timestep        : {timestep_min} min\n"
            f"  Wind            : {wind_str}\n"
            f"  Soil Saturation : {soil_sat_pct:.0f}%\n"
            f"  Drain Capacity  : {drain_cap:.1f} mm/hr\n"
            f"\n"
            f"  HYDROLOGICAL ANALYSIS\n"
            f"  {sep}\n"
            f"  Return Period   : {ret_period}\n"
            f"  Peak Discharge  : {peak_Q_m3s:.3f} m\u00b3/s\n"
            f"  Runoff Volume   : {stats['runoff_vol_m3'][fi]/1000:.1f} \u00d7 10\u00b3 m\u00b3\n"
            f"  Watershed Area  : {total_area_m2/10000:.2f} ha\n"
            f"\n"
            f"  LIVE STATUS  [{times_list[fi]}]\n"
            f"  {sep}\n"
            f"  Elapsed Time    : {fi * timestep_min} min\n"
            f"  Rain Fallen     : {stats['rain_mm'][fi]:.1f} mm\n"
            f"  Max Depth       : {stats['max_depth_mm'][fi]:.0f} mm\n"
            f"  Flooded Area    : {stats['flooded_pct'][fi]:.1f}%"
            f" ({stats['flooded_ha'][fi]:.2f} ha)\n"
            f"\n"
            f"  RIVER OVERFLOW\n"
            f"  {sep}\n"
            f"  Area Affected   : {stats['river_pct'][fi]:.1f}% of grid\n"
            f"  Max Depth       : {stats['max_river_mm'][fi]:.0f} mm\n"
            f"\n"
            f"  RISK ASSESSMENT (PAGASA)\n"
            f"  {sep}\n"
            f"  {rsk}\n"
        )
        patch = stats_txt.get_bbox_patch()
        if patch is not None:
            patch.set_edgecolor(rsk_col)

        x_so_far = time_axis[:fi + 1]
        depth_line.set_data(x_so_far, stats["max_depth_mm"][:fi + 1])
        chart_vline.set_xdata([elapsed_list[fi], elapsed_list[fi]])

        if abs(sl_frame.val - fi) > 0.5:
            sl_frame.eventson = False
            sl_frame.set_val(fi)
            sl_frame.eventson = True
        fig.canvas.draw_idle()

    BASE_INTERVAL = 600

    def _anim_step(_) -> list:
        if player["playing"]:
            draw(player["frame"] + 1)
        return []

    anim_obj = animation.FuncAnimation(
        fig, _anim_step, interval=BASE_INTERVAL,
        blit=False, cache_frame_data=False)

    def on_frame(val):
        draw(int(val))

    def on_speed(val):
        anim_obj.event_source.interval = max(50, int(BASE_INTERVAL / max(val, 0.01)))

    def on_play_pause(_):
        player["playing"] = not player["playing"]
        if player["playing"]:
            btn_play.label.set_text("Pause")
            btn_play.ax.set_facecolor("#1B5E20")
        else:
            btn_play.label.set_text("Play")
            btn_play.ax.set_facecolor("#BF360C")
        fig.canvas.draw_idle()

    def on_prev(_):
        player["playing"] = False
        btn_play.label.set_text("Play")
        btn_play.ax.set_facecolor("#BF360C")
        draw(player["frame"] - 1)

    def on_next(_):
        player["playing"] = False
        btn_play.label.set_text("Play")
        btn_play.ax.set_facecolor("#BF360C")
        draw(player["frame"] + 1)

    def _do_export_csv():
        safe = (scenario_name.replace(" ", "_").replace("/", "-")
                             .replace("(", "").replace(")", ""))
        p = str(ANIM_DIR / f"flood_{safe}_stats.csv")
        _export_sim_csv(p, times_list, elapsed_list, rate_frames, stats, timestep_min)
        print(f"  \u2713 CSV exported  \u2192  {p}")

    def on_save_gif(_):
        safe_name = (scenario_name.replace(" ", "_").replace("/", "-")
                                   .replace("(", "").replace(")", ""))
        out_path = str(ANIM_DIR / f"flood_{safe_name}.gif")
        print(f"\n  Saving GIF: {out_path}  (may take 30\u201360 s)\u2026")
        was_playing = player["playing"]
        player["playing"] = False
        if not PIL_OK:
            print("  [ERROR] Pillow not installed. Run: pip install Pillow")
            player["playing"] = was_playing
            return

        tmp_fig, (tmp_ax_m, tmp_ax_s) = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK)
        tmp_ax_m.set_facecolor("black"); tmp_ax_s.set_facecolor(PANEL); tmp_ax_s.axis("off")
        tmp_ax_m.imshow(bg, extent=ext, aspect="auto", zorder=1, interpolation="bilinear")
        tmp_ax_m.imshow(strm_rgba, extent=ext, aspect="auto", zorder=4, interpolation="nearest")
        _irn = tmp_ax_m.imshow(rain_frames[0] * 1000, cmap=cmap_rain, vmin=0, vmax=600,
                                extent=ext, aspect="auto", zorder=2)
        _irv = tmp_ax_m.imshow((river_frames[0] > 0.005).astype(float),
                                cmap=cmap_river, vmin=0, vmax=1,
                                extent=ext, aspect="auto", zorder=3)
        tmp_ax_m.set_xlim(0, W); tmp_ax_m.set_ylim(H, 0)
        _ttxt = tmp_ax_m.text(
            0.015, 0.975, "", transform=tmp_ax_m.transAxes,
            fontsize=11, fontweight="bold", color="white", va="top",
            bbox=dict(boxstyle="round", facecolor=PANEL, alpha=0.88, edgecolor=ACC))
        _stxt = tmp_ax_s.text(
            0.05, 0.97, "", fontsize=8.5, family="monospace",
            color=TCLR, va="top", transform=tmp_ax_s.transAxes)

        frames_pil = []
        for fi in range(num_frames):
            rn = rain_frames[fi] * 1000
            _irn.set_data(rn)
            _irn.set_clim(0, min(max(float(rn.max()), 30) * 1.3, 600))
            _irv.set_data((river_frames[fi] > 0.005).astype(float))
            rsk_g, _ = _risk_str(stats["max_depth_mm"][fi], stats["river_pct"][fi])
            _ttxt.set_text(
                f"Time: {times_list[fi]}\n"
                f"Rain: {stats['rain_mm'][fi]:.0f}/{rainfall_mm:.0f} mm")
            _stxt.set_text(
                f"{scenario_name}\n\n"
                f"Return Period : {ret_period}\n"
                f"Peak Discharge: {peak_Q_m3s:.3f} m\u00b3/s\n"
                f"Watershed Area: {total_area_m2/10000:.2f} ha\n\n"
                f"Time     : {times_list[fi]}\n"
                f"Rain     : {stats['rain_mm'][fi]:.1f} mm\n"
                f"Flooded  : {stats['flooded_pct'][fi]:.1f}%"
                f" ({stats['flooded_ha'][fi]:.2f} ha)\n"
                f"Max Depth: {stats['max_depth_mm'][fi]:.0f} mm\n"
                f"River Fld: {stats['river_pct'][fi]:.1f}%\n"
                f"Risk     : {rsk_g}")
            tmp_fig.canvas.draw()
            buf = io.BytesIO()
            tmp_fig.savefig(buf, format="png", dpi=75, bbox_inches="tight", facecolor=DARK)
            buf.seek(0)
            frames_pil.append(PILImage.open(buf).copy().convert("P"))  # type: ignore[possibly-undefined]
            print(f"  GIF frame {fi + 1}/{num_frames} \u2026", end="\r", flush=True)

        print()
        plt.close(tmp_fig)
        frames_pil[0].save(out_path, save_all=True, append_images=frames_pil[1:],
                           loop=0, duration=int(1000 / 5))
        print(f"  \u2713 Saved GIF  ({len(frames_pil)} frames)  \u2192  {out_path}")
        player["playing"] = was_playing

    def on_save_csv(_):
        _do_export_csv()

    sl_frame.on_changed(on_frame)
    sl_speed.on_changed(on_speed)
    btn_play.on_clicked(on_play_pause)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_gif .on_clicked(on_save_gif)
    btn_csv .on_clicked(on_save_csv)

    draw(0)
    try:
        if _backend == 'TkAgg':
            mgr = plt.get_current_fig_manager()
            _win = getattr(mgr, 'window', None)
            if _win is not None and hasattr(_win, 'wm_geometry'):
                _x = max(0, (_screen_w - _fig_w) // 2)
                _y = max(0, (_screen_h - _fig_h) // 2)
                try:
                    _win.wm_geometry(f"{_fig_w}x{_fig_h}+{_x}+{_y}")
                except Exception:
                    pass
    except Exception:
        pass
    print("\n  Interactive viewer ready.")
    print("  Controls: Pause/Play | Step | Speed x | Save GIF | Save CSV")
    plt.show()


# =============================================================================
# GUI LAUNCHER  (tkinter)
# =============================================================================


class SimulationGUI:
    """Tkinter GUI for configuring and launching the flood simulation."""

    # Color scheme
    BG      = '#0D1117'
    PANEL   = '#161B22'
    CARD    = '#1C2333'
    BORDER  = '#30363D'
    TEXT    = '#E6EDF3'
    ACCENT  = '#4FC3F7'
    GREEN   = '#2EA043'
    ORANGE  = '#F0883E'
    RED     = '#DA3633'
    WHITE   = '#FFFFFF'

    def __init__(self, dem: np.ndarray, cellsize: float):
        self.dem      = dem
        self.cellsize = cellsize

        self.root = tk.Tk()
        self.root.title("Jade Valley Flood Simulator — Davao City, Philippines")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1200, int(screen_w * 0.78))
        win_h = min(800,  int(screen_h * 0.84))
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.minsize(900, 600)

        # ── Styling ──────────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame',      background=self.BG)
        style.configure('Card.TFrame', background=self.CARD)
        style.configure('TLabel',      background=self.BG,
                        foreground=self.TEXT, font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=self.BG,
                        foreground=self.ACCENT, font=('Segoe UI', 12, 'bold'))
        style.configure('Title.TLabel', background=self.BG,
                        foreground=self.WHITE,
                        font=('Segoe UI', 16, 'bold'))
        style.configure('Desc.TLabel', background=self.CARD,
                        foreground='#8B949E', font=('Segoe UI', 9))
        style.configure('TCombobox', fieldbackground=self.PANEL,
                        foreground=self.TEXT, font=('Segoe UI', 10))
        style.configure('Val.TLabel', background=self.BG,
                        foreground=self.ORANGE,
                        font=('Consolas', 10, 'bold'))

        # ── Title banner ─────────────────────────────────────────────────
        ttk.Label(self.root, text="JADE VALLEY SUBDIVISION",
                  style='Title.TLabel').pack(pady=(14, 0))
        ttk.Label(self.root,
                  text="Hydrological Flood Simulation  —  Davao City, Philippines",
                  style='Desc.TLabel').pack()
        ttk.Label(self.root,
                  text="D8 Flow Routing  •  Green-Ampt Infiltration  •  BFS River Overflow  •  PAGASA Risk Levels",
                  background=self.BG, foreground='#484F58',
                  font=('Segoe UI', 8)).pack(pady=(0, 4))

        # ── Terrain stats card ───────────────────────────────────────────
        tc = tk.Frame(self.root, bg='#1C2333', relief='flat')
        tc.pack(fill='x', padx=16, pady=(2, 6))
        total_area_ha  = dem.size * cellsize ** 2 / 10000
        elev_range_m   = dem.max() - dem.min()
        slope_arr      = np.hypot(*np.gradient(dem, cellsize, cellsize))
        mean_slope_pct = float(np.mean(slope_arr)) / cellsize * 100
        terrain_info   = (
            f"  │  Grid: {dem.shape[0]}×{dem.shape[1]} cells"
            f"  │  Cell size: {cellsize:.1f} m"
            f"  │  Elevation: {dem.min():.1f} – {dem.max():.1f} m"
            f"  │  Relief: {elev_range_m:.1f} m"
            f"  │  Area: {total_area_ha:.1f} ha"
            f"  │"
        )
        tk.Label(tc, text="TERRAIN", bg='#161B22', fg=self.ACCENT,
                 font=('Segoe UI', 8, 'bold'), padx=10, pady=5).pack(side='left')
        tk.Label(tc, text=terrain_info, bg='#1C2333', fg='#8B949E',
                 font=('Consolas', 8), pady=5).pack(side='left', fill='x')

        # ── Main two-column frame ────────────────────────────────────────
        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True, padx=16, pady=10)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        # ────────────────────────────────────────────────────────────────
        # LEFT COLUMN  —  Scenario Presets
        # ────────────────────────────────────────────────────────────────
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 8))

        ttk.Label(left, text="SCENARIO PRESET",
                  style='Header.TLabel').pack(anchor='w')

        self.scenario_var = tk.StringVar(value="3")
        sc_frame = ttk.Frame(left)
        sc_frame.pack(fill='x', pady=4)

        # Colour each scenario radio button by severity to aid quick visual selection.
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
            rb = tk.Radiobutton(
                sc_frame, text=label,
                variable=self.scenario_var, value=key,
                bg=self.BG, fg=color, selectcolor=self.PANEL,
                activebackground=self.BG, activeforeground=self.ACCENT,
                font=('Segoe UI', 9), anchor='w',
                command=self._on_preset_change)
            rb.pack(fill='x', pady=1)

        # Scenario description
        self.desc_var = tk.StringVar()
        desc_lbl = tk.Label(sc_frame, textvariable=self.desc_var,
                            bg=self.CARD, fg='#8B949E',
                            font=('Segoe UI', 9, 'italic'),
                            wraplength=380, justify='left', padx=8, pady=6)
        desc_lbl.pack(fill='x', pady=(6, 0))

        # PAGASA rainfall classification reference (color-banded)
        ref_frame = tk.Frame(sc_frame, bg=self.CARD)
        ref_frame.pack(fill='x', padx=4, pady=(4, 2))
        tk.Label(ref_frame, text="PAGASA Classification (mm/hr):",
                 bg=self.CARD, fg='#8B949E', font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=4)
        pagasa_bands = [
            ("Light  0–7.5",   "#3FB950"),
            ("Moderate 7.5–15", "#79C0FF"),
            ("Heavy 15–30",    "#FFD740"),
            ("Intense >30",    "#FF6B6B"),
        ]
        band_row = tk.Frame(ref_frame, bg=self.CARD)
        band_row.pack(fill='x', padx=4, pady=2)
        for label_text, col in pagasa_bands:
            tk.Label(band_row, text=f"  {label_text}  ",
                     bg=col, fg='#0D1117',
                     font=('Segoe UI', 7, 'bold'), relief='flat', padx=2
                     ).pack(side='left', padx=2, pady=1)

        # ── Pattern selector ─────────────────────────────────────────────
        sep1 = ttk.Frame(left)
        sep1.pack(fill='x', pady=8)
        ttk.Label(sep1, text="STORM PATTERN",
                  style='Header.TLabel').pack(anchor='w')
        self.pattern_var = tk.StringVar(value="burst")
        pat_frame = ttk.Frame(sep1)
        pat_frame.pack(fill='x')
        patterns = [("uniform", "Constant rate"),
                    ("progressive", "Builds up → peaks"),
                    ("burst", "Bell-curve peak mid-storm"),
                    ("decreasing", "Heavy start → tapers off")]
        for val, desc in patterns:
            rb = tk.Radiobutton(
                pat_frame, text=f"{val}  — {desc}",
                variable=self.pattern_var, value=val,
                bg=self.BG, fg=self.TEXT, selectcolor=self.PANEL,
                activebackground=self.BG, activeforeground=self.ACCENT,
                font=('Segoe UI', 9), anchor='w')
            rb.pack(fill='x', pady=1)

        # ────────────────────────────────────────────────────────────────
        # RIGHT COLUMN  —  Manual Sliders
        # ────────────────────────────────────────────────────────────────
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky='nsew', padx=(8, 0))

        ttk.Label(right, text="SIMULATION PARAMETERS",
                  style='Header.TLabel').pack(anchor='w')

        # Helper to create labelled scales
        self.sliders: dict[str, tk.Scale] = {}

        def add_slider(parent, key, label, from_, to, default,
                       resolution=1.0, fmt="{:.0f}"):
            f = ttk.Frame(parent)
            f.pack(fill='x', pady=2)
            ttk.Label(f, text=label).pack(anchor='w')
            s = tk.Scale(f, from_=from_, to=to, orient='horizontal',
                         resolution=resolution, length=300,
                         bg=self.BG, fg=self.TEXT, troughcolor=self.PANEL,
                         highlightthickness=0, font=('Consolas', 9),
                         activebackground=self.ACCENT)
            s.set(default)
            s.pack(fill='x')
            self.sliders[key] = s

        add_slider(right, 'rainfall_mm',
                   'Rainfall Total (mm)       [5 – 500]',
                   5, 500, 130)
        add_slider(right, 'duration_h',
                   'Storm Duration (hours)    [0.5 – 24]',
                   0.5, 24, 4.0, resolution=0.5)
        add_slider(right, 'timestep_min',
                   'Timestep (minutes)        [5 – 60]',
                   5, 60, 10)
        add_slider(right, 'wind_speed',
                   'Wind Speed (km/h)         [0 – 200]',
                   0, 200, 0)
        add_slider(right, 'wind_dir',
                   'Wind Direction (°)        [0 – 360]',
                   0, 359, 270)
        add_slider(right, 'soil_sat',
                   'Soil Saturation (%)       [0 – 100]',
                   0, 100, 30)
        add_slider(right, 'drain_cap',
                   'Drainage Capacity (mm/hr) [0.5 – 50]',
                   0.5, 50, 5.0, resolution=0.5)

        # ── Start time ───────────────────────────────────────────────────
        tf = ttk.Frame(right)
        tf.pack(fill='x', pady=4)
        ttk.Label(tf, text='Start Time (HH:MM)').pack(anchor='w')
        self.start_time_var = tk.StringVar(value="14:00")
        te = tk.Entry(tf, textvariable=self.start_time_var,
                      bg=self.PANEL, fg=self.TEXT, insertbackground=self.TEXT,
                      font=('Consolas', 11), width=8)
        te.pack(anchor='w')

        # ────────────────────────────────────────────────────────────────
        # BOTTOM BUTTONS
        # ────────────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=16, pady=(4, 14))

        _BFONT = ('Segoe UI', 11, 'bold')

        tk.Button(btn_frame, text="RANDOMIZE",
                  bg='#6E40C9', fg='white', activebackground='#8957E5',
                  font=_BFONT, width=20, height=2, bd=0, cursor='hand2',
                  command=self._randomize
                  ).pack(side='left', padx=4)

        tk.Button(btn_frame, text="\u25B6  RUN SIMULATION",
                  bg=self.GREEN, fg='white', activebackground='#3FB950',
                  font=_BFONT, width=20, height=2, bd=0, cursor='hand2',
                  command=self._run
                  ).pack(side='left', padx=4)

        tk.Button(btn_frame, text="\u2715  EXIT",
                  bg=self.RED, fg='white', activebackground='#F85149',
                  font=_BFONT, width=20, height=2, bd=0, cursor='hand2',
                  command=self.root.destroy
                  ).pack(side='right', padx=4)

        # ── Status / info bar ─────────────────────────────────────────────
        info = (
            f"  Model: D8 Flow Routing + Green-Ampt Infiltration + BFS Overflow"
            f"  │  DEM: {dem.shape[0]}×{dem.shape[1]} ({cellsize:.1f} m/cell)"
            f"  │  Elevation: {dem.min():.1f}–{dem.max():.1f} m"
            f"  │  Area: {dem.size * cellsize**2 / 10000:.1f} ha"
            f"  │  Outputs: GIF + CSV saved to Results/animations/"
        )
        tk.Label(self.root, text=info, bg='#161B22', fg='#484F58',
                 font=('Consolas', 8), pady=5, anchor='w').pack(fill='x', side='bottom')

        self._is_randomized = False
        self._on_preset_change()
        self.root.mainloop()

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_preset_change(self):
        self._is_randomized = False
        key = self.scenario_var.get()
        sc  = SCENARIOS.get(key, SCENARIOS["3"])
        if sc["rainfall_mm"] is not None:
            self.sliders['rainfall_mm'].set(sc["rainfall_mm"])
            self.sliders['duration_h'].set(sc["duration_h"])
            self.pattern_var.set(sc["pattern"])
            self.desc_var.set(sc.get("desc", ""))

    def _randomize(self):
        """Fill all sliders with random but realistic values."""
        # Pick a random intensity class
        # Rainfall ranges calibrated to PAGASA thresholds for Mindanao
        classes = [
            ("Light Rain",               10,  25,  1.5,  3.0),
            ("Moderate Rain",            25,  60,  2.5,  5.0),
            ("Heavy Rain",               60, 120,  3.5,  6.0),
            ("Typhoon Signal 1",        120, 200,  6.0, 10.0),
            ("Typhoon Signal 2",        190, 300, 10.0, 14.0),
            ("Typhoon Signal 3 (Severe)",280, 450, 14.0, 20.0),
        ]
        name, rain_lo, rain_hi, dur_lo, dur_hi = random.choice(classes)

        rain = random.randint(rain_lo, rain_hi)
        dur  = round(random.uniform(dur_lo, dur_hi) * 2) / 2  # snap 0.5
        wind = random.choice([0, 0, 0,
                              random.randint(20, 60),
                              random.randint(60, 120),
                              random.randint(100, 180)])
        wdir = random.randint(0, 359)
        soil = random.randint(10, 85)
        drain = random.choice([1.5, 2.0, 3.0, 5.0, 5.0, 5.0, 8.0, 12.0])
        pat  = random.choice(["uniform", "progressive", "burst", "decreasing"])

        hours = [f"{h:02d}:00" for h in range(0, 24)]
        start = random.choice(hours)

        self.sliders['rainfall_mm'].set(rain)
        self.sliders['duration_h'].set(dur)
        self.sliders['timestep_min'].set(10)
        self.sliders['wind_speed'].set(wind)
        self.sliders['wind_dir'].set(wdir)
        self.sliders['soil_sat'].set(soil)
        self.sliders['drain_cap'].set(drain)
        self.pattern_var.set(pat)
        self.start_time_var.set(start)

        self._is_randomized = True
        self.desc_var.set(
            f"Randomized:  {name}  |  {rain} mm  /  {dur} h  /  "
            f"wind {wind} km/h  /  soil {soil}%  /  drain {drain} mm/hr"
        )

    def _run(self):
        """Gather all values and launch the simulation."""
        rainfall_mm = float(self.sliders['rainfall_mm'].get())
        duration_h  = float(self.sliders['duration_h'].get())
        timestep_m  = int(self.sliders['timestep_min'].get())
        wind_spd    = float(self.sliders['wind_speed'].get())
        wind_dir    = float(self.sliders['wind_dir'].get())
        soil_sat    = float(self.sliders['soil_sat'].get())
        drain       = float(self.sliders['drain_cap'].get())
        pattern     = self.pattern_var.get()
        start_str   = self.start_time_var.get().strip()

        # Validate start time
        try:
            h, m = map(int, start_str.split(':'))
            if not (0 <= h < 24 and 0 <= m < 60):
                raise ValueError()
        except (ValueError, AttributeError):
            print(f"  [warn] Invalid start time '{start_str}' — defaulting to 14:00")
            start_str = "14:00"

        # Build scenario name from the actual selection
        if self._is_randomized:
            scenario_name = f"Randomized ({rainfall_mm:.0f} mm / {duration_h:.1f} h)"
        else:
            key = self.scenario_var.get()
            sc  = SCENARIOS.get(key)
            if sc and sc["name"] and sc["rainfall_mm"] is not None:
                scenario_name = sc["name"]
            else:
                scenario_name = f"Custom ({rainfall_mm:.0f} mm / {duration_h:.1f} h)"

        if wind_spd >= 100:
            scenario_name += f" + Wind {wind_spd:.0f} km/h"

        # Close GUI window before running matplotlib animation
        self.root.destroy()

        print("\n" + "=" * 68)
        print("  STARTING SIMULATION")
        print("=" * 68)
        print(f"  Scenario  : {scenario_name}")
        print(f"  Rainfall  : {rainfall_mm:.0f} mm in {duration_h:.1f} h")
        print(f"  Pattern   : {pattern}")
        print(f"  Wind      : {wind_spd:.0f} km/h from {wind_dir:.0f}°")
        print(f"  Soil      : {soil_sat:.0f}%   Drain: {drain:.1f} mm/hr")
        print(f"  Start     : {start_str}")
        print("=" * 68)

        run_simulation(
            dem            = self.dem,
            cellsize       = self.cellsize,
            rainfall_mm    = rainfall_mm,
            duration_h     = duration_h,
            timestep_min   = timestep_m,
            start_time_str = start_str,
            wind_speed     = wind_spd,
            wind_dir       = wind_dir,
            soil_sat_pct   = soil_sat,
            drain_cap      = drain,
            pattern        = pattern,
            scenario_name  = scenario_name,
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  JADE VALLEY SUBDIVISION — FLOOD ANIMATION SIMULATOR")
    print("  Davao City, Philippines")
    print("=" * 68)

    print("\nLoading terrain data…")
    dem, cellsize = load_dem()

    print("\nLaunching GUI…")
    SimulationGUI(dem, cellsize)

    print("\n" + "=" * 68)
    print("  DONE — GIF saved to Results/animations/")
    print("=" * 68)