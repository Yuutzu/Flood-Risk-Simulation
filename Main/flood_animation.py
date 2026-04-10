"""
=============================================================================
 JADE VALLEY SUBDIVISION — ANIMATED FLOOD SIMULATION
 Davao City, Philippines
=============================================================================
 Interactive animated simulation showing:
   • Real-time flood progression over your terrain (JVS_Simulation.tif)
   • River overflow when rain volume causes the river to rise
   • D8 flow routing — water follows natural valleys and channels
   • Separate rain runoff layer vs river overflow layer
   • Manual inputs: rainfall amount, typhoon/storm scenario, duration,
     timestep, wind, soil saturation, drainage capacity
   • Playback controls: Play/Pause, scrubber, speed slider, ◀◀/▶▶
   • Save GIF button — exports animation to Results/animations/
=============================================================================
"""

import heapq
import io
import os
import sys
import warnings
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, Slider

warnings.filterwarnings("ignore")

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

SCENARIOS = {
    "1": {
        "name"       : "Light Rain",
        "rainfall_mm": 8,
        "duration_h" : 2.0,
        "pattern"    : "uniform",
        "desc"       : "Avg 4 mm/hr — minor puddles, no flood risk",
    },
    "2": {
        "name"       : "Moderate Rain",
        "rainfall_mm": 15,
        "duration_h" : 3.0,
        "pattern"    : "progressive",
        "desc"       : "Avg 5 mm/hr — typical Davao afternoon rain",
    },
    "3": {
        "name"       : "Heavy Rain",
        "rainfall_mm": 35,
        "duration_h" : 4.0,
        "pattern"    : "burst",
        "desc"       : "Avg 8.75 mm/hr — low areas may collect water",
    },
    "4": {
        "name"       : "Typhoon Signal 1 (Tropical Depression)",
        "rainfall_mm": 100,
        "duration_h" : 8.0,
        "pattern"    : "progressive",
        "desc"       : "Avg 12.5 mm/hr — river may rise; monitor advisories",
    },
    "5": {
        "name"       : "Typhoon Signal 2 (Tropical Storm)",
        "rainfall_mm": 180,
        "duration_h" : 12.0,
        "pattern"    : "burst",
        "desc"       : "Avg 15 mm/hr — widespread flooding; prepare to evacuate",
    },
    "6": {
        "name"       : "Typhoon Signal 3 (Severe Typhoon)",
        "rainfall_mm": 300,
        "duration_h" : 18.0,
        "pattern"    : "burst",
        "desc"       : "Avg 17 mm/hr — catastrophic flooding; evacuate",
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

        # Normalised helpers
        e = self.dem
        self.elev_norm  = (e - e.min()) / (e.max() - e.min() + 1e-10)
        fa_log = np.log1p(self.accum)
        self.flow_weight = fa_log / (fa_log.max() + 1e-10)

        self.slope = np.hypot(*np.gradient(self.dem, cellsize, cellsize))
        slope_n = self.slope / (self.slope.max() + 1e-10)
        self.runoff_coeff = np.clip(0.45 + 0.45 * slope_n, 0.30, 0.95)
        self.max_inf = 0.3 + 0.9 * self.elev_norm * (1.0 - init_sat)

        # River channel = top 8% flow accumulation
        self.river_mask = self.accum >= float(np.percentile(self.accum, 92))

        # Bank elevation per river cell
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

        # BFS flood front starts at the river channel
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
        if not self.river_mask.any():
            return
        rain_m = (rate_mmhr / 1000.0) * dt_h * intensity

        # River only starts rising after 15% of rain has accumulated on
        # the surface — water needs time to flow into channels first.
        accum_frac = self.rainfall_accumulated / (self.rainfall_accumulated + 0.05)
        if accum_frac < 0.15:
            return

        # Scale river rise with how much rain has already fallen
        # (gradual ramp: 0 at start → full strength once 40%+ accumulated)
        ramp = np.clip((accum_frac - 0.15) / 0.25, 0.0, 1.0)
        rise_mult = ramp * (8.0 + 30.0 * self.flow_weight[self.river_mask])

        self.river_level[self.river_mask] += rain_m * rise_mult

        rise = self.river_level[self.river_mask] - \
               self.river_level_init[self.river_mask]
        flood_rise = float(np.percentile(rise, 90))
        if flood_rise <= 0:
            return

        eff_wse = self.flood_base_wse + min(flood_rise, 3.0)

        # BFS dilation — hops scale with flood progression (1→6)
        can_flood = (self.dem_raw < eff_wse) & ~self.river_mask
        struct    = np.ones((3, 3), dtype=bool)
        HOPS = max(1, int(ramp * 6))
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
        target = np.clip(eff_wse - self.dem_raw, 0.0, 4.0)
        target[self.river_mask] = 0.0
        delta = (target[land] - self.river_water[land]) * 0.50
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
        channel_rate = 80.0 * self.flow_weight ** 2
        rate = (self.drainage_capacity * (1.0 + 2.0*self.elev_norm) + channel_rate)
        drain_m = (rate / 1000.0) * dt_h
        self.rain_water  = np.maximum(self.rain_water  - drain_m,       0.0)
        self.river_water = np.maximum(self.river_water - drain_m * 0.4, 0.0)

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
        (0.55, 0.00, 1.00, 0.75),
    ]
    return LinearSegmentedColormap.from_list('rain', colors)


def _river_cmap():
    return LinearSegmentedColormap.from_list(
        'river',
        [(0, 0, 0, 0), (0.12, 0.38, 0.82, 0.50)], N=2)


# =============================================================================
# STORM INTENSITY PATTERN
# =============================================================================

def _intensity_factor(frame: int, total_frames: int, pattern: str) -> float:
    t = frame / max(total_frames - 1, 1)
    if pattern == 'progressive':
        return 0.2 + 1.8 * min(t * 1.5, 1.0)
    elif pattern == 'burst':
        return 0.15 + 3.0 * float(np.exp(-((t - 0.45) ** 2) / 0.040))
    elif pattern == 'decreasing':
        return max(2.0 - 1.8 * t, 0.05)
    return 1.0   # uniform


# =============================================================================
# MAIN ANIMATION FUNCTION
# =============================================================================

def run_simulation(dem: np.ndarray, cellsize: float,
                   rainfall_mm: float, duration_h: float,
                   timestep_min: int, start_time_str: str,
                   wind_speed: float, wind_dir: float,
                   soil_sat_pct: float, drain_cap: float,
                   pattern: str, scenario_name: str):
    """Run simulation, build interactive animated viewer, auto-save GIF."""

    rate_mmhr   = rainfall_mm / duration_h
    dt_h        = timestep_min / 60.0
    num_frames  = int(np.ceil(duration_h / dt_h))

    print(f"\n  Scenario  : {scenario_name}")
    print(f"  Rainfall  : {rainfall_mm:.0f} mm in {duration_h:.1f} h  "
          f"({rate_mmhr:.1f} mm/hr, {pattern})")
    print(f"  Timestep  : {timestep_min} min  →  {num_frames} frames")
    print(f"  Wind      : {wind_speed:.0f} km/h from {wind_dir:.0f}°")
    print(f"  Soil      : {soil_sat_pct:.0f}%  Drain: {drain_cap:.1f} mm/hr")

    sim = FloodSimulation(dem, cellsize,
                          soil_saturation_pct   = soil_sat_pct,
                          drainage_capacity_mmhr= drain_cap)

    wmap = wind_rainfall_map(dem, wind_speed, wind_dir)

    # ── Run all timesteps ─────────────────────────────────────────────────────
    rain_frames  = []
    river_frames = []
    times_list   = []
    stats        = {"rain_mm": [], "flooded_pct": [], "river_pct": [],
                    "max_depth_mm": [], "max_river_mm": []}

    sh, sm = map(int, start_time_str.split(':'))
    cur = datetime.now().replace(hour=sh, minute=sm, second=0)
    print("\n  Simulating…")

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
                  f"flooded={stats['flooded_pct'][-1]:.1f}%  "
                  f"maxdepth={stats['max_depth_mm'][-1]:.0f} mm")
        cur += timedelta(minutes=timestep_min)

    # ── Background and colormaps ──────────────────────────────────────────────
    print("\n  Rendering map background…")
    bg         = render_jpeg_background(dem)
    cmap_rain  = _rain_cmap()
    cmap_river = _river_cmap()

    # ── Figure layout ─────────────────────────────────────────────────────────
    DARK  = '#0D1117'
    PANEL = '#161B22'
    TCLR  = '#E6EDF3'
    ACC   = '#4FC3F7'

    fig = plt.figure(figsize=(24, 12), facecolor=DARK)
    fig.suptitle(
        f"JADE VALLEY SUBDIVISION  —  FLOOD SIMULATION  |  {scenario_name}",
        fontsize=14, fontweight='bold', color=TCLR, y=0.978)

    ax_map   = fig.add_axes((0.03, 0.18, 0.62, 0.76))
    ax_stats = fig.add_axes((0.69, 0.18, 0.29, 0.76))
    ax_map  .set_facecolor('black')
    ax_stats.set_facecolor(PANEL)
    ax_stats.axis('off')

    H, W = dem.shape
    ext  = (0, W, H, 0)

    im_bg    = ax_map.imshow(bg, extent=ext, aspect='auto',
                              zorder=1, interpolation='bilinear')
    im_rain  = ax_map.imshow(rain_frames[0] * 1000,
                              cmap=cmap_rain, vmin=0, vmax=600,
                              extent=ext, aspect='auto',
                              zorder=2, interpolation='bilinear',
                              alpha=0.45)
    im_river = ax_map.imshow(
        (river_frames[0] > 0.005).astype(float),
        cmap=cmap_river, vmin=0, vmax=1,
        extent=ext, aspect='auto',
        zorder=3, interpolation='nearest',
        alpha=0.40)

    ax_map.set_xlim(0, W); ax_map.set_ylim(H, 0)
    ax_map.tick_params(colors=TCLR, labelsize=8)
    for sp in ax_map.spines.values():
        sp.set_edgecolor('#30363D')

    cbar = fig.colorbar(im_rain, ax=ax_map, orientation='vertical',
                        pad=0.01, shrink=0.80)
    cbar.set_label("Rain Water Depth (mm)", color=TCLR, fontsize=9)
    cbar.set_ticks([0, 30, 100, 200, 400, 600])
    cbar.ax.set_yticklabels(['Dry','30','100','200','400','600+'],
                             color=TCLR, fontsize=8)
    cbar.ax.tick_params(colors=TCLR)

    time_txt = ax_map.text(
        0.015, 0.975, "", transform=ax_map.transAxes,
        fontsize=12, fontweight='bold', color='white', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL,
                  alpha=0.90, edgecolor=ACC, linewidth=1.5))

    stats_txt = ax_stats.text(
        0.05, 0.97, "", fontsize=10, family='monospace',
        color=TCLR, va='top', transform=ax_stats.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#0D1117',
                  edgecolor=ACC, alpha=0.92, linewidth=1.2))

    # ── Widgets ───────────────────────────────────────────────────────────────
    ax_sl_frame = fig.add_axes((0.03, 0.098, 0.62, 0.024), facecolor='#21262D')
    ax_sl_speed = fig.add_axes((0.03, 0.048, 0.26, 0.024), facecolor='#21262D')
    ax_btn_play = fig.add_axes((0.335, 0.028, 0.09, 0.055))
    ax_btn_prev = fig.add_axes((0.430, 0.028, 0.055, 0.055))
    ax_btn_next = fig.add_axes((0.490, 0.028, 0.055, 0.055))
    ax_btn_gif  = fig.add_axes((0.555, 0.028, 0.10, 0.055))

    sl_frame = Slider(ax_sl_frame, 'Frame', 0, num_frames-1,
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
    compass_lbl = DIRS[int((wind_dir + 11.25) / 22.5) % 16]
    wind_str    = (f"{wind_speed:.0f} km/h from {compass_lbl}"
                   if wind_speed >= 1.0 else "None")

    def _risk_str(pct, river_pct):
        if river_pct > 25 or pct > 70:
            return "⚠ EVACUATE NOW",           '#FF1744'
        if river_pct > 12 or pct > 45:
            return "MANDATORY EVACUATION",     '#FF6D00'
        if pct > 25:
            return "PRE-EVACUATION ALERT",     '#FFD740'
        if pct > 10:
            return "STANDBY — prepare to move", '#69F0AE'
        return     "NORMAL — monitoring",       '#B0BEC5'

    def draw(fi):
        fi = int(fi) % num_frames
        player['frame'] = fi
        rn = rain_frames[fi] * 1000
        rv = river_frames[fi]

        im_rain .set_data(rn)
        im_rain .set_clim(0, min(max(float(rn.max()), 30) * 1.3, 600))
        im_river.set_data((rv > 0.005).astype(float))

        time_txt.set_text(
            f" Time: {times_list[fi]}\n"
            f" Rain: {stats['rain_mm'][fi]:.0f}/{rainfall_mm:.0f} mm\n"
            f" Wind: {wind_str}")

        rsk, rsk_col = _risk_str(stats['flooded_pct'][fi],
                                    stats['river_pct'][fi])
        stats_txt.set_text(
            f"  SCENARIO\n"
            f"  {'─'*26}\n"
            f"  {scenario_name}\n"
            f"\n"
            f"  PARAMETERS\n"
            f"  {'─'*26}\n"
            f"  Rainfall : {rainfall_mm:.0f} mm total\n"
            f"  Rate     : {rate_mmhr:.1f} mm/hr ({pattern})\n"
            f"  Duration : {duration_h:.1f} hr\n"
            f"  Timestep : {timestep_min} min\n"
            f"  Wind     : {wind_str}\n"
            f"  Soil sat : {soil_sat_pct:.0f}%\n"
            f"  Drainage : {drain_cap:.1f} mm/hr\n"
            f"\n"
            f"  LIVE STATUS  [{times_list[fi]}]\n"
            f"  {'─'*26}\n"
            f"  Elapsed  : {fi * timestep_min} min\n"
            f"  Fallen   : {stats['rain_mm'][fi]:.1f} mm\n"
            f"  Max depth: {stats['max_depth_mm'][fi]:.0f} mm\n"
            f"  Flooded  : {stats['flooded_pct'][fi]:.1f}% of area\n"
            f"\n"
            f"  RIVER OVERFLOW\n"
            f"  {'─'*26}\n"
            f"  Area     : {stats['river_pct'][fi]:.1f}% flooded\n"
            f"  Max depth: {stats['max_river_mm'][fi]:.0f} mm\n"
            f"\n"
            f"  RISK LEVEL\n"
            f"  {rsk}\n"
        )
        stats_txt.get_bbox_patch().set_edgecolor(rsk_col)  # type: ignore[union-attr]

        if abs(sl_frame.val - fi) > 0.5:
            sl_frame.eventson = False
            sl_frame.set_val(fi)
            sl_frame.eventson = True
        fig.canvas.draw_idle()

    BASE_INTERVAL = 600   # ms at speed ×1

    def _anim_step(_):
        if player['playing']:
            draw(player['frame'] + 1)

    def _anim_step_wrapper(_) -> list:  # type: ignore[return]
        _anim_step(_)
        return []

    anim_obj = animation.FuncAnimation(
        fig, _anim_step_wrapper, interval=BASE_INTERVAL,
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
        safe_name = (scenario_name.replace(' ', '_')
                                   .replace('/', '-')
                                   .replace('(', '').replace(')', ''))
        out_path = str(ANIM_DIR / f"flood_{safe_name}.gif")
        print(f"\n  Saving GIF: {out_path}  (may take 30–60 s)…")
        was_playing = player['playing']
        player['playing'] = False

        if not PIL_OK:
            print("  [ERROR] Pillow not installed. Run: pip install Pillow")
            player['playing'] = was_playing
            return

        tmp_fig, (tmp_ax_m, tmp_ax_s) = plt.subplots(1, 2, figsize=(16, 7),
                                                       facecolor=DARK)
        tmp_ax_m.set_facecolor('black'); tmp_ax_s.set_facecolor(PANEL)
        tmp_ax_s.axis('off')
        _ibg  = tmp_ax_m.imshow(bg, extent=ext, aspect='auto',
                                  zorder=1, interpolation='bilinear')
        _irn  = tmp_ax_m.imshow(rain_frames[0]*1000,
                                 cmap=cmap_rain, vmin=0, vmax=600,
                                 extent=ext, aspect='auto', zorder=2)
        _irv  = tmp_ax_m.imshow((river_frames[0]>0.005).astype(float),
                                 cmap=cmap_river, vmin=0, vmax=1,
                                 extent=ext, aspect='auto', zorder=3)
        tmp_ax_m.set_xlim(0, W); tmp_ax_m.set_ylim(H, 0)
        _ttxt = tmp_ax_m.text(
            0.015, 0.975, "", transform=tmp_ax_m.transAxes,
            fontsize=11, fontweight='bold', color='white', va='top',
            bbox=dict(boxstyle='round', facecolor=PANEL, alpha=0.88,
                      edgecolor=ACC))
        _stxt = tmp_ax_s.text(
            0.05, 0.97, "", fontsize=9, family='monospace',
            color=TCLR, va='top', transform=tmp_ax_s.transAxes)

        frames_pil = []
        for fi in range(num_frames):
            rn = rain_frames[fi] * 1000
            _irn.set_data(rn)
            _irn.set_clim(0, min(max(float(rn.max()), 30)*1.3, 600))
            _irv.set_data((river_frames[fi] > 0.005).astype(float))
            _ttxt.set_text(
                f"⏱ {times_list[fi]}\n"
                f"Rain: {stats['rain_mm'][fi]:.0f}/{rainfall_mm:.0f} mm")
            _stxt.set_text(
                f"{scenario_name}\n\n"
                f"Time:     {times_list[fi]}\n"
                f"Rain:     {stats['rain_mm'][fi]:.1f} mm\n"
                f"Flooded:  {stats['flooded_pct'][fi]:.1f}%\n"
                f"RiverFld: {stats['river_pct'][fi]:.1f}%\n"
                f"MaxDepth: {stats['max_depth_mm'][fi]:.0f} mm\n"
                f"RiverMax: {stats['max_river_mm'][fi]:.0f} mm")
            tmp_fig.canvas.draw()
            buf = io.BytesIO()
            tmp_fig.savefig(buf, format='png', dpi=75,
                             bbox_inches='tight', facecolor=DARK)
            buf.seek(0)
            frames_pil.append(
                PILImage.open(buf).copy().convert('P'))  # type: ignore[possibly-undefined]

        plt.close(tmp_fig)
        frames_pil[0].save(
            out_path, save_all=True, append_images=frames_pil[1:],
            loop=0, duration=int(1000 / 5))
        print(f"  ✓ Saved GIF  ({len(frames_pil)} frames)  →  {out_path}")
        player['playing'] = was_playing

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
# GUI LAUNCHER  (tkinter)
# =============================================================================

import random
import tkinter as tk
from tkinter import ttk


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
        ttk.Label(self.root, text="Flood Animation Simulator  —  Davao City, Philippines",
                  style='Desc.TLabel').pack()

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

        for key, sc in SCENARIOS.items():
            if sc["rainfall_mm"] is None:
                continue
            rb = tk.Radiobutton(
                sc_frame, text=f"{sc['name']}",
                variable=self.scenario_var, value=key,
                bg=self.BG, fg=self.TEXT, selectcolor=self.PANEL,
                activebackground=self.BG, activeforeground=self.ACCENT,
                font=('Segoe UI', 9), anchor='w',
                command=self._on_preset_change)
            rb.pack(fill='x', pady=1)

        # Scenario description
        self.desc_var = tk.StringVar()
        desc_lbl = tk.Label(sc_frame, textvariable=self.desc_var,
                            bg=self.CARD, fg='#8B949E',
                            font=('Segoe UI', 9, 'italic'),
                            wraplength=340, justify='left', padx=8, pady=6)
        desc_lbl.pack(fill='x', pady=(6, 0))

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

        tk.Button(btn_frame, text="\U0001F3B2  RANDOMIZE",
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

        # ── Info bar ─────────────────────────────────────────────────────
        info = (f"DEM: {dem.shape[0]}×{dem.shape[1]}  |  "
                f"Cell: {cellsize:.1f} m  |  "
                f"Elev: {dem.min():.1f}–{dem.max():.1f} m  |  "
                f"Area: {dem.size * cellsize**2 / 10000:.1f} ha")
        tk.Label(self.root, text=info, bg=self.BORDER, fg='#8B949E',
                 font=('Consolas', 9), pady=4).pack(fill='x', side='bottom')

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
        classes = [
            ("Light Rain",                  10,  50,  1.0,  3.0),
            ("Moderate Rain",               40, 100,  2.0,  5.0),
            ("Heavy Rain",                  80, 180,  3.0,  6.0),
            ("Typhoon Signal 1",           140, 220,  4.0,  8.0),
            ("Typhoon Signal 2",           200, 300,  6.0, 10.0),
            ("Typhoon Signal 3 (Severe)",  280, 450,  8.0, 14.0),
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
            f"🎲 Randomized:  {name}  |  {rain} mm  /  {dur} h  /  "
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
