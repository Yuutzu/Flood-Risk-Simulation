"""
=============================================================================
 JADE VALLEY SUBDIVISION — FLOOD RISK SIMULATION
 Davao City, Philippines
=============================================================================
 Method  : HAND (Height Above Nearest Drainage) Model
 Engine  : pysheds (primary) / numpy+scipy (fallback)
 Inputs  : ASC Digital Elevation Model, DXF 2D/3D topographic models
 Outputs : Results/maps/  — PNG flood inundation and risk maps
           Results/data/  — JSON statistics, TXT report, NPY arrays
=============================================================================
"""

import heapq
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.axes as mpl_axes
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS & CONFIGURATION
# =============================================================================

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Map Topography"
RESULTS_DIR = BASE_DIR / "Results"
MAPS_DIR   = RESULTS_DIR / "maps"
DATA_OUT   = RESULTS_DIR / "data"

TIF_FILE   = DATA_DIR / "3D" / "JVS_Simulation.tif"
DXF_2D     = DATA_DIR / "2D" / "Jade_Valley_Subdivision_2D_vectorial.dxf"
DXF_3D     = DATA_DIR / "3D" / "Jade_Valley_Subdivision_3D_modeling.dxf"

# Create output directories
for d in (MAPS_DIR, DATA_OUT):
    d.mkdir(parents=True, exist_ok=True)

PARAMS = {
    "location"              : "Jade Valley Subdivision, Davao City, Philippines",
    "analysis_date"         : datetime.now().strftime("%Y-%m-%d"),
    "stream_threshold_pct"  : 95,    # Top X% flow accumulation = stream
    "smooth_sigma"          : 0.5,   # Gaussian DEM smoothing σ
    "curve_number"          : 85,    # Urban residential CN (Philippines)
}

# HAND thresholds for each flood return period (metres)
FLOOD_SCENARIOS = {
    "5-Year Return"   : {"hand_thresh": 1.0, "color": "#FFFF00", "return_period":   5},
    "10-Year Return"  : {"hand_thresh": 2.0, "color": "#FFA500", "return_period":  10},
    "25-Year Return"  : {"hand_thresh": 3.5, "color": "#FF6600", "return_period":  25},
    "100-Year Return" : {"hand_thresh": 6.0, "color": "#FF0000", "return_period": 100},
}

# Risk zone classification by HAND (metres)
RISK_CLASSES = {
    "Very High Risk" : {"max_hand":  1.0, "color": "#D32F2F", "desc": "Prone to annual flooding"},
    "High Risk"      : {"max_hand":  3.0, "color": "#F57C00", "desc": "Frequent flood events"},
    "Medium Risk"    : {"max_hand":  6.0, "color": "#FBC02D", "desc": "Occasional flooding"},
    "Low Risk"       : {"max_hand": 12.0, "color": "#388E3C", "desc": "Rare flood events"},
    "Safe Zone"      : {"max_hand": 9999, "color": "#1565C0", "desc": "Minimal flood risk"},
}


# =============================================================================
# LIBRARY DETECTION
# =============================================================================

def check_libraries() -> dict:
    libs = {}
    for name, pkg in [("pysheds", "pysheds"), ("rasterio", "rasterio"),
                       ("ezdxf", "ezdxf"), ("pandas", "pandas")]:
        try:
            __import__(pkg)
            libs[name] = True
            print(f"  [OK]  {name}")
        except ImportError:
            libs[name] = False
            print(f"  [--]  {name} not installed (optional)")
    return libs


# =============================================================================
# DATA LOADING
# =============================================================================

def load_tif_dem(filepath: Path) -> tuple:
    """Load a GeoTIFF DEM and return (dem_array, metadata_dict)."""
    print(f"\nLoading DEM: {filepath.name}")
    try:
        import rasterio
        from rasterio.crs import CRS
        with rasterio.open(str(filepath)) as src:
            dem = src.read(1).astype(np.float64)
            nodata = src.nodata if src.nodata is not None else -9999.0
            dem[dem == nodata] = np.nan
            transform = src.transform
            cellsize_x = abs(float(transform.a))
            cellsize_y = abs(float(transform.e))
            # If geographic CRS (degrees), convert cell size to metres
            crs = src.crs
            if crs and crs.is_geographic:
                lat = float(src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2)
                cellsize = cellsize_x * 111320 * abs(np.cos(np.radians(lat)))
            else:
                cellsize = (cellsize_x + cellsize_y) / 2
            xll = float(src.bounds.left)
            yll = float(src.bounds.bottom)
    except ImportError:
        # Fallback: read with GDAL via subprocess is not available;
        # raise a clear error so the user knows rasterio is needed
        raise RuntimeError(
            "rasterio is required to read GeoTIFF. "
            "Install it with: pip install rasterio"
        )

    meta = {
        "ncols"          : int(dem.shape[1]),
        "nrows"          : int(dem.shape[0]),
        "xllcorner"      : xll,
        "yllcorner"      : yll,
        "cellsize"       : round(cellsize, 4),
        "nodata_value"   : nodata,
        "shape"          : list(dem.shape),
        "valid_cells"    : int(np.sum(~np.isnan(dem))),
        "min_elevation"  : float(np.nanmin(dem)),
        "max_elevation"  : float(np.nanmax(dem)),
        "mean_elevation" : float(np.nanmean(dem)),
        "area_m2"        : float(np.sum(~np.isnan(dem)) * cellsize ** 2),
    }
    print(f"  Grid   : {dem.shape[0]} rows × {dem.shape[1]} cols  |  cell ≈ {meta['cellsize']} m")
    print(f"  Elev.  : {meta['min_elevation']:.2f} – {meta['max_elevation']:.2f} m")
    print(f"  Area   : {meta['area_m2'] / 10000:.2f} ha")
    return dem, meta


def load_dxf_info(filepath: Path, label: str = "DXF") -> dict | None:
    """Return summary dict of DXF entities (requires ezdxf)."""
    try:
        import ezdxf
        doc = ezdxf.readfile(str(filepath))  # type: ignore[attr-defined]
        msp = doc.modelspace()
        entity_types: dict = {}
        layers: set = set()
        for ent in msp:
            t = ent.dxftype()
            entity_types[t] = entity_types.get(t, 0) + 1
            try:
                layers.add(ent.dxf.layer)
            except Exception:
                pass
        total = sum(entity_types.values())
        print(f"\n  {label}: {filepath.name}  — {total} entities, {len(layers)} layers")
        for k, v in sorted(entity_types.items()):
            print(f"    {k}: {v}")
        return {"entities": entity_types, "layers": sorted(layers), "total": total}
    except ImportError:
        print(f"\n  [skip] ezdxf not installed — {filepath.name}")
        return None
    except Exception as exc:
        print(f"\n  [warn] DXF load error ({filepath.name}): {exc}")
        return None


# =============================================================================
# DEM PREPROCESSING
# =============================================================================

def fill_nodata(dem: np.ndarray) -> np.ndarray:
    """Replace NaN cells with nearest valid elevation."""
    nan_mask = np.isnan(dem)
    if not nan_mask.any():
        return dem
    indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
    filled = dem.copy()
    row_idx = np.asarray(indices[0], dtype=int)[nan_mask]  # type: ignore[index]
    col_idx = np.asarray(indices[1], dtype=int)[nan_mask]  # type: ignore[index]
    filled[nan_mask] = dem[row_idx, col_idx]
    print(f"  Filled {int(nan_mask.sum())} NoData cells")
    return filled


def priority_flood_fill_sinks(dem: np.ndarray) -> np.ndarray:
    """
    Fill topographic sinks using the Priority Flood algorithm
    (Barnes et al. 2014). All interior depressions are raised to the
    spill elevation of their enclosing basin.
    """
    rows, cols = dem.shape
    filled = dem.copy()
    processed = np.zeros((rows, cols), dtype=bool)
    pq: list = []

    # Seed priority queue with all border cells
    def push(r: int, c: int):
        if not processed[r, c]:
            heapq.heappush(pq, (filled[r, c], r, c))
            processed[r, c] = True

    for r in range(rows):
        push(r, 0); push(r, cols - 1)
    for c in range(1, cols - 1):
        push(0, c); push(rows - 1, c)

    neighbors8 = [(-1, -1), (-1, 0), (-1, 1),
                  (0,  -1),          (0,  1),
                  (1,  -1), (1,  0), (1,  1)]

    while pq:
        elev, r, c = heapq.heappop(pq)
        for dr, dc in neighbors8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not processed[nr, nc]:
                processed[nr, nc] = True
                filled[nr, nc] = max(filled[nr, nc], elev)
                heapq.heappush(pq, (filled[nr, nc], nr, nc))

    return filled


def preprocess_dem(dem: np.ndarray, sigma: float = 0.5) -> tuple:
    """Fill NoData → Gaussian smooth → fill sinks. Returns (dem_proc, nan_mask)."""
    print("\nPreprocessing DEM…")
    nan_mask = np.isnan(dem)
    dem_proc = fill_nodata(dem)
    if sigma > 0:
        dem_proc = ndimage.gaussian_filter(dem_proc, sigma=sigma)
        print(f"  Gaussian smooth (σ={sigma})")
    dem_proc = priority_flood_fill_sinks(dem_proc)
    print("  Sinks filled (Priority Flood)")
    return dem_proc, nan_mask


def calculate_slope(dem: np.ndarray, cellsize: float = 1.0) -> np.ndarray:
    """Return slope in degrees using Sobel finite differences."""
    dz_dx = ndimage.sobel(dem, axis=1) / (8.0 * cellsize)
    dz_dy = ndimage.sobel(dem, axis=0) / (8.0 * cellsize)
    return np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))


# =============================================================================
# HYDROLOGICAL ANALYSIS — D8 FLOW DIRECTION
# =============================================================================

# Direction index → (row_offset, col_offset)
D8_OFFSETS = [(0, 1), (1, 1), (1, 0), (1, -1),
              (0, -1), (-1, -1), (-1, 0), (-1, 1)]
D8_WEIGHTS = [1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414]


def d8_flow_direction(dem: np.ndarray) -> np.ndarray:
    """
    Vectorised D8 flow direction.
    Returns int8 array; each value is the direction index (0–7) of steepest descent.
    """
    print("\nD8 flow direction…")
    rows, cols = dem.shape
    slopes = np.full((8, rows, cols), -np.inf)

    for d, ((dr, dc), w) in enumerate(zip(D8_OFFSETS, D8_WEIGHTS)):
        padded = np.pad(dem, 1, mode="edge")
        neighbor = padded[1 + dr: rows + 1 + dr, 1 + dc: cols + 1 + dc]
        s = (dem - neighbor) / w
        # Mask boundary edges so water doesn't flow off the grid
        if dr == 1:  s[-1, :] = -np.inf
        if dr == -1: s[0,  :] = -np.inf
        if dc == 1:  s[:, -1] = -np.inf
        if dc == -1: s[:,  0] = -np.inf
        slopes[d] = s

    return np.argmax(slopes, axis=0).astype(np.int8)


# =============================================================================
# FLOW ACCUMULATION
# =============================================================================

def flow_accumulation(flow_dir: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """
    Compute flow accumulation (upstream contributing cells) via
    topological sort (highest → lowest elevation).
    """
    print("Flow accumulation…")
    rows, cols = dem.shape
    accum = np.ones((rows, cols), dtype=np.float32)

    for flat_idx in np.argsort(dem.ravel())[::-1]:  # high → low
        r, c = divmod(int(flat_idx), cols)
        dr, dc = D8_OFFSETS[int(flow_dir[r, c])]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            accum[nr, nc] += accum[r, c]

    return accum


# =============================================================================
# STREAM NETWORK & HAND MODEL
# =============================================================================

def delineate_streams(accum: np.ndarray, threshold_pct: float = 95.0) -> np.ndarray:
    """Return boolean mask: True where flow accumulation ≥ Nth percentile."""
    thresh = np.percentile(accum, threshold_pct)
    streams = accum >= thresh
    print(f"Streams: {streams.sum()} cells (accum ≥ {thresh:.0f})")
    return streams


def calculate_hand(dem: np.ndarray, streams: np.ndarray,
                   flow_dir: np.ndarray) -> np.ndarray:
    """
    Height Above Nearest Drainage (HAND).
    For each cell, HAND = elevation − elevation of the nearest downstream
    stream cell along the D8 flow path.

    Algorithm: process cells in ascending elevation order so the downstream
    neighbour's nearest-stream-elevation is always known before the
    upstream cell needs it.
    """
    print("HAND model…")
    rows, cols = dem.shape
    nearest_str_elev = np.full((rows, cols), np.nan)
    nearest_str_elev[streams] = dem[streams]

    for flat_idx in np.argsort(dem.ravel()):        # low → high
        r, c = divmod(int(flat_idx), cols)
        if streams[r, c]:
            nearest_str_elev[r, c] = dem[r, c]
        else:
            dr, dc = D8_OFFSETS[int(flow_dir[r, c])]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = nearest_str_elev[nr, nc]
                if not np.isnan(v):
                    nearest_str_elev[r, c] = v

    hand = dem - nearest_str_elev
    hand = np.where(np.isnan(hand), float(np.nanmax(dem) - np.nanmin(dem)), hand)
    return np.maximum(hand, 0.0)


# =============================================================================
# PYSHEDS PATHWAY (preferred — much faster for large DEMs)
# =============================================================================

def run_pysheds(tif_file: Path, threshold_pct: float = 95.0) -> tuple:
    """Full hydrological chain via pysheds. Returns (dem, accum, hand, streams)."""
    from pysheds.grid import Grid  # type: ignore

    print("\n[pysheds] Loading grid…")
    grid = Grid.from_raster(str(tif_file))
    raw  = grid.read_raster(str(tif_file), dtype=np.float32)

    print("[pysheds] Conditioning DEM…")
    pit_filled = grid.fill_pits(raw)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)

    print("[pysheds] Flow direction…")
    fdir = grid.flowdir(inflated)

    print("[pysheds] Flow accumulation…")
    acc  = grid.accumulation(fdir)

    print("[pysheds] Delineating streams…")
    thresh  = np.percentile(np.asarray(acc), threshold_pct)
    streams = np.asarray(acc) >= thresh

    print("[pysheds] HAND…")
    hand = grid.compute_hand(fdir, inflated, streams)

    return (np.asarray(inflated, dtype=np.float64),
            np.asarray(acc,      dtype=np.float32),
            np.asarray(hand,     dtype=np.float64),
            streams.astype(bool))


# =============================================================================
# FLOOD SIMULATION & RISK CLASSIFICATION
# =============================================================================

def simulate_floods(hand: np.ndarray, metadata: dict) -> dict:
    """Map HAND thresholds to flood inundation masks and depth estimates."""
    print("\nSimulating flood scenarios…")
    cell_ha = (metadata["cellsize"] ** 2) / 10_000
    results = {}

    for name, cfg in FLOOD_SCENARIOS.items():
        thr   = cfg["hand_thresh"]
        mask  = hand <= thr
        depth = np.where(mask, np.maximum(thr - hand, 0.0), 0.0)
        n     = int(mask.sum())
        results[name] = {
            "mask"          : mask,
            "depth"         : depth,
            "flooded_cells" : n,
            "flooded_ha"    : float(n * cell_ha),
            "pct_area"      : float(n / hand.size * 100),
            "max_depth_m"   : float(depth.max()),
            "mean_depth_m"  : float(depth[mask].mean()) if mask.any() else 0.0,
            "hand_threshold": thr,
            "return_period" : cfg["return_period"],
            "color"         : cfg["color"],
        }
        print(f"  {name:<22} → {results[name]['flooded_ha']:7.2f} ha  "
              f"({results[name]['pct_area']:.1f}%)  max depth {results[name]['max_depth_m']:.2f} m")

    return results


def classify_risk(hand: np.ndarray, metadata: dict) -> tuple:
    """Assign each cell a 1–5 risk class based on HAND thresholds."""
    risk_map = np.zeros(hand.shape, dtype=np.int8)
    cell_ha  = (metadata["cellsize"] ** 2) / 10_000

    prev = 0.0
    for i, (name, cfg) in enumerate(RISK_CLASSES.items(), start=1):
        mask = (hand > prev) & (hand <= cfg["max_hand"])
        risk_map[mask] = i
        prev = cfg["max_hand"]

    stats = {}
    for i, (name, cfg) in enumerate(RISK_CLASSES.items(), start=1):
        n = int((risk_map == i).sum())
        stats[name] = {
            "area_ha"    : float(n * cell_ha),
            "percentage" : float(n / risk_map.size * 100),
            "color"      : cfg["color"],
            "desc"       : cfg["desc"],
        }
    return risk_map, stats


# =============================================================================
# VISUALISATION HELPERS
# =============================================================================

# =============================================================================
# OSM BACKGROUND HELPER
# =============================================================================

_OSM_BOUNDS: dict | None = None

def _get_tif_bounds() -> dict | None:
    """Read geographic bounds from the TIF file (EPSG:4326)."""
    global _OSM_BOUNDS
    if _OSM_BOUNDS is not None:
        return _OSM_BOUNDS
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(str(TIF_FILE)) as src:
            crs = src.crs
            bounds = src.bounds
            if crs and not crs.is_geographic:
                l, b, r, t = transform_bounds(crs, "EPSG:4326",
                                              bounds.left, bounds.bottom,
                                              bounds.right, bounds.top)
            else:
                l, b, r, t = bounds.left, bounds.bottom, bounds.right, bounds.top
        _OSM_BOUNDS = {"left": l, "bottom": b, "right": r, "top": t}
        return _OSM_BOUNDS
    except Exception:
        return None


def add_osm_background(ax: mpl_axes.Axes, alpha: float = 0.55) -> bool:
    """
    Add OpenStreetMap tiles as background to an axes that already has
    imshow data on it. Returns True if successful.
    """
    if not HAS_CONTEXTILY:
        return False
    bounds = _get_tif_bounds()
    if bounds is None:
        return False
    try:
        import contextily as ctx
        from rasterio.crs import CRS
        # Set axes extent in geographic coords, then add basemap
        ax.set_xlim(0, 1)  # imshow coords — use extent parameter below
        # We work with imshow so we add basemap as background image
        # by fetching tiles and displaying them as an imshow layer
        west, south, east, north = (bounds["left"], bounds["bottom"],
                                     bounds["right"], bounds["top"])
        img, ext = ctx.bounds2img(west, south, east, north,
                                  zoom="auto",
                                  source=ctx.providers.OpenStreetMap.Mapnik)  # type: ignore[attr-defined]
        # ext = (left, right, bottom, top) in EPSG:3857
        # Our imshow uses pixel coords — insert OSM as first layer using
        # a separate axes approach: draw behind via ax.imshow with extent
        rows, cols = img.shape[:2]
        ax.imshow(img, extent=(0.0, 1.0, 0.0, 1.0), aspect="auto",
                  transform=ax.transAxes, alpha=alpha,
                  zorder=0, origin="upper")
        return True
    except Exception:
        return False


def osm_background_fig(ax: mpl_axes.Axes, dem_shape: tuple,
                        alpha: float = 0.5) -> None:
    """
    Fetch OSM tiles matched to TIF bounds and display behind data layers.
    Uses a twin-axes approach so pixel-space imshow overlays stay aligned.
    """
    if not HAS_CONTEXTILY:
        return
    bounds = _get_tif_bounds()
    if bounds is None:
        return
    try:
        import contextily as ctx
        west  = bounds["left"]
        south = bounds["bottom"]
        east  = bounds["right"]
        north = bounds["top"]
        img, _ = ctx.bounds2img(west, south, east, north,
                                zoom="auto",
                                source=ctx.providers.OpenStreetMap.Mapnik)  # type: ignore[attr-defined]
        rows, cols = dem_shape
        # Resize OSM tile to DEM pixel dimensions
        from PIL import Image as PILImage
        from PIL.Image import Resampling
        pil = PILImage.fromarray(img)
        pil = pil.resize((cols, rows), Resampling.LANCZOS)
        osm_arr = np.array(pil)
        ax.imshow(osm_arr, extent=(0, cols, rows, 0),
                  aspect="auto", alpha=alpha, zorder=0)
    except Exception:
        pass


def hillshade(dem: np.ndarray, az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    az_r, alt_r = np.radians(360.0 - az + 90), np.radians(alt)
    dy, dx = np.gradient(dem)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(-dy, dx)
    hs = (np.cos(alt_r) * np.cos(slope) +
          np.sin(alt_r) * np.sin(slope) * np.cos(az_r - aspect))
    return np.clip(hs, 0, 1)


def _risk_cmap_norm():
    colors = [v["color"] for v in RISK_CLASSES.values()]
    cmap   = mcolors.ListedColormap(colors)
    bounds = [0] + list(range(1, len(RISK_CLASSES) + 1))
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


# =============================================================================
# OUTPUT MAPS
# =============================================================================

def save_terrain_overview(dem, meta, slope, accum, streams, fpath):
    hs = hillshade(dem)
    rows, cols = dem.shape
    ext = (0, cols, rows, 0)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Jade Valley Subdivision — Terrain Analysis\nDavao City, Philippines",
        fontsize=14, fontweight="bold"
    )
    # DEM
    ax = axes[0, 0]
    osm_background_fig(ax, dem.shape, alpha=0.45)
    im = ax.imshow(dem, cmap="terrain", aspect="auto", alpha=0.65,
                   extent=ext)
    plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.8)
    ax.contour(dem, levels=10, colors="gray", alpha=0.35, linewidths=0.4)
    ax.set_title("Digital Elevation Model", fontweight="bold"); ax.axis("off")

    # Slope
    ax = axes[0, 1]
    osm_background_fig(ax, dem.shape, alpha=0.45)
    im = ax.imshow(slope, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=45,
                   alpha=0.70, extent=ext)
    plt.colorbar(im, ax=ax, label="Slope (°)", shrink=0.8)
    ax.set_title("Slope", fontweight="bold"); ax.axis("off")

    # Flow accumulation + streams
    ax = axes[1, 0]
    osm_background_fig(ax, dem.shape, alpha=0.45)
    im = ax.imshow(np.log1p(accum), cmap="Blues", aspect="auto", alpha=0.65,
                   extent=ext)
    plt.colorbar(im, ax=ax, label="log(Flow Accumulation)", shrink=0.8)
    stream_overlay = np.ma.masked_where(~streams, np.ones_like(dem))
    ax.imshow(stream_overlay, cmap="winter_r", alpha=0.7, aspect="auto",
              extent=ext)
    ax.set_title("Flow Accumulation & Drainage Network", fontweight="bold"); ax.axis("off")

    # Hillshade
    ax = axes[1, 1]
    osm_background_fig(ax, dem.shape, alpha=0.50)
    ax.imshow(hs, cmap="gray", aspect="auto", alpha=0.45,
              extent=ext)
    ax.imshow(dem, cmap="terrain", aspect="auto", alpha=0.35,
              extent=ext)
    ax.set_title("Hillshade + Elevation", fontweight="bold"); ax.axis("off")

    info = (f"Area: {meta['area_m2']/10000:.1f} ha | "
            f"Elev: {meta['min_elevation']:.0f}–{meta['max_elevation']:.0f} m | "
            f"Cell: {meta['cellsize']} m | Date: {PARAMS['analysis_date']}")
    fig.text(0.5, 0.01, info, ha="center", fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {Path(fpath).name}")


def save_hand_map(hand, streams, meta, fpath):
    hs = hillshade(np.where(np.isnan(hand), 0, hand))
    rows, cols = hand.shape
    ext = (0, cols, rows, 0)
    fig, ax = plt.subplots(figsize=(10, 8))
    osm_background_fig(ax, hand.shape, alpha=0.50)
    ax.imshow(hs, cmap="gray", aspect="auto", alpha=0.35, extent=ext)

    colors = [v["color"] for v in RISK_CLASSES.values()]
    bounds = [0, 1, 3, 6, 12, float(hand.max()) + 1]
    cmap   = mcolors.ListedColormap(colors)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(hand, cmap=cmap, norm=norm, alpha=0.65, aspect="auto", extent=ext)

    # Stream overlay
    sov = np.ma.masked_where(~streams, np.ones_like(hand))
    ax.imshow(sov, cmap="winter_r", alpha=0.6, aspect="auto", extent=ext)

    ax.set_title(
        "Height Above Nearest Drainage (HAND)\nJade Valley Subdivision, Davao City",
        fontweight="bold"
    )
    ax.axis("off")
    legend = [mpatches.Patch(color=v["color"], label=k)
              for k, v in RISK_CLASSES.items()]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {Path(fpath).name}")


def save_flood_maps(dem, flood_results, meta, out_dir):
    hs = hillshade(dem)
    rows, cols = dem.shape
    ext = (0, cols, rows, 0)
    for name, res in flood_results.items():
        safe = name.replace(" ", "_").replace("/", "-").replace("≤", "lte")
        fpath = out_dir / f"flood_{safe}.png"
        fig, ax = plt.subplots(figsize=(10, 8))
        osm_background_fig(ax, dem.shape, alpha=0.55)
        ax.imshow(hs,  cmap="gray",    aspect="auto", alpha=0.30, extent=ext)
        ax.imshow(dem, cmap="terrain", aspect="auto", alpha=0.20, extent=ext)

        if res["mask"].any():
            rgba = np.zeros((*dem.shape, 4))
            depth_n = res["depth"] / max(res["depth"].max(), 1e-9)
            rgba[res["mask"], 0] = depth_n[res["mask"]] * 0.15
            rgba[res["mask"], 1] = (1 - depth_n[res["mask"]]) * 0.55
            rgba[res["mask"], 2] = 0.92
            rgba[res["mask"], 3] = 0.80
            ax.imshow(rgba, aspect="auto", extent=ext)

        ax.set_title(
            f"Flood Inundation — {res['return_period']}-Year Return Period\n"
            f"Jade Valley Subdivision, Davao City  |  HAND ≤ {res['hand_threshold']} m",
            fontweight="bold"
        )
        info = (f"Flooded: {res['flooded_ha']:.2f} ha ({res['pct_area']:.1f}%)\n"
                f"Max depth: {res['max_depth_m']:.2f} m  |  "
                f"Mean depth: {res['mean_depth_m']:.2f} m")
        ax.text(0.02, 0.03, info, transform=ax.transAxes, fontsize=9,
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                va="bottom")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(fpath), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  {fpath.name}")


def save_comparison_map(dem, flood_results, meta, fpath):
    hs = hillshade(dem)
    rows, cols = dem.shape
    ext = (0, cols, rows, 0)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "Flood Inundation Comparison — Multiple Return Periods\n"
        "Jade Valley Subdivision, Davao City, Philippines",
        fontsize=14, fontweight="bold"
    )
    for idx, (name, res) in enumerate(flood_results.items()):
        ax = axes[idx // 2][idx % 2]
        osm_background_fig(ax, dem.shape, alpha=0.55)
        ax.imshow(hs,  cmap="gray",    aspect="auto", alpha=0.25, extent=ext)
        ax.imshow(dem, cmap="terrain", aspect="auto", alpha=0.15, extent=ext)
        if res["mask"].any():
            rgba = np.zeros((*dem.shape, 4))
            rgba[res["mask"]] = [0.05, 0.40, 0.90, 0.80]
            ax.imshow(rgba, aspect="auto", extent=ext)
        ax.set_title(
            f"{res['return_period']}-Year  |  HAND ≤ {res['hand_threshold']} m\n"
            f"{res['flooded_ha']:.1f} ha  ({res['pct_area']:.1f}%)",
            fontsize=10, fontweight="bold"
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {Path(fpath).name}")


def save_risk_map(dem, hand, risk_map, risk_stats, meta, fpath):
    hs = hillshade(dem)
    rows, cols = dem.shape
    ext = (0, cols, rows, 0)
    fig, (ax_map, ax_stats) = plt.subplots(
        1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [3, 1]}
    )
    osm_background_fig(ax_map, dem.shape, alpha=0.55)
    ax_map.imshow(hs, cmap="gray", aspect="auto", alpha=0.25, extent=ext)
    cmap, norm = _risk_cmap_norm()
    disp = np.where(risk_map > 0, risk_map.astype(float), np.nan)
    ax_map.imshow(disp, cmap=cmap, norm=norm, alpha=0.70, aspect="auto", extent=ext)
    ax_map.set_title(
        "Flood Risk Zone Map\nJade Valley Subdivision, Davao City, Philippines",
        fontweight="bold", fontsize=13
    )
    ax_map.text(0.97, 0.97, "N\n↑", transform=ax_map.transAxes,
                fontsize=15, fontweight="bold", ha="center", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax_map.axis("off")

    # Statistics panel
    ax_stats.axis("off")
    ax_stats.set_title("Risk Zone Statistics", fontweight="bold", fontsize=12)
    total_ha = sum(v["area_ha"] for v in risk_stats.values())
    ax_stats.text(0.05, 0.97, f"Total area: {total_ha:.1f} ha",
                  transform=ax_stats.transAxes, fontsize=10, fontweight="bold")
    y = 0.88
    for rname, st in risk_stats.items():
        color = RISK_CLASSES[rname]["color"]
        ax_stats.add_patch(
            mpatches.Rectangle((0.04, y - 0.025), 0.14, 0.04,
                                color=color, transform=ax_stats.transAxes)
        )
        ax_stats.text(0.22, y,        rname,
                      transform=ax_stats.transAxes, fontsize=9, fontweight="bold", va="center")
        ax_stats.text(0.22, y - 0.03, f"{st['area_ha']:.2f} ha ({st['percentage']:.1f}%)",
                      transform=ax_stats.transAxes, fontsize=8.5, color="gray", va="center")
        ax_stats.text(0.22, y - 0.055, st["desc"],
                      transform=ax_stats.transAxes, fontsize=7.5, color="#666", va="center",
                      style="italic")
        y -= 0.14
    ax_stats.text(0.05, 0.08,
                  f"Method: HAND Model\nResolution: {meta['cellsize']} m\n"
                  f"Date: {PARAMS['analysis_date']}",
                  transform=ax_stats.transAxes, fontsize=8, fontfamily="monospace",
                  color="gray",
                  bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9))
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {Path(fpath).name}")


# =============================================================================
# REPORT & DATA EXPORT
# =============================================================================

def generate_report(meta, slope, hand, flood_results, risk_stats, out_path):
    """Write a human-readable flood risk assessment report (.txt)."""
    W = 72
    lines = []
    lines += ["=" * W,
              "  JADE VALLEY SUBDIVISION — FLOOD RISK ASSESSMENT REPORT",
              "  Davao City, Philippines",
              "=" * W,
              f"  Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
              f"  Method    : HAND (Height Above Nearest Drainage) Model",
              f"  Sources   : GeoTIFF DEM + 2D/3D DXF topographic models",
              ""]

    lines += ["-" * W, "1. STUDY AREA", "-" * W,
              f"  Location   : {PARAMS['location']}",
              f"  Grid       : {meta['nrows']} rows × {meta['ncols']} cols",
              f"  Cell size  : {meta['cellsize']} m",
              f"  Total area : {meta['area_m2']/10000:.2f} ha",
              f"  Elevation  : {meta['min_elevation']:.2f} – {meta['max_elevation']:.2f} m",
              f"  Mean elev. : {meta['mean_elevation']:.2f} m", ""]

    lines += ["-" * W, "2. TERRAIN ANALYSIS", "-" * W,
              f"  Mean slope : {float(np.nanmean(slope)):.1f}°",
              f"  Max slope  : {float(np.nanmax(slope)):.1f}°",
              f"  Flat (<5°) : {float((slope < 5).sum() / slope.size * 100):.1f}%",
              f"  Gentle     : {float(((slope>=5)&(slope<15)).sum()/slope.size*100):.1f}%  (5–15°)",
              f"  Steep (>15°): {float((slope>=15).sum()/slope.size*100):.1f}%", ""]

    lines += ["-" * W, "3. FLOOD SCENARIOS", "-" * W]
    for name, res in flood_results.items():
        lines += [f"  [{res['return_period']}-Year Return Period]",
                  f"  HAND threshold : {res['hand_threshold']} m",
                  f"  Flooded area   : {res['flooded_ha']:.2f} ha  ({res['pct_area']:.1f}%)",
                  f"  Max depth      : {res['max_depth_m']:.2f} m",
                  f"  Mean depth     : {res['mean_depth_m']:.2f} m", ""]

    lines += ["-" * W, "4. RISK ZONE CLASSIFICATION", "-" * W]
    for name, st in risk_stats.items():
        lines += [f"  {name}",
                  f"    {st['area_ha']:.2f} ha  ({st['percentage']:.1f}%)  — {st['desc']}", ""]

    lines += ["-" * W, "5. METHODOLOGY", "-" * W,
              "  The HAND model measures the vertical distance from each terrain",
              "  cell to the nearest drainage channel along the flow path. Lower",
              "  HAND values indicate higher flood susceptibility.",
              "",
              "  Steps executed:",
              "  1. NoData filling (nearest-neighbour)",
              "  2. Gaussian DEM smoothing (σ=0.5)",
              "  3. Sink filling — Priority Flood algorithm (Barnes et al. 2014)",
              "  4. D8 flow direction (steepest 8-neighbour descent)",
              "  5. Flow accumulation (topological sort)",
              "  6. Stream delineation (top 5% flow accumulation)",
              "  7. HAND computation (ascending elevation propagation)",
              "  8. Flood scenarios mapped to HAND thresholds",
              "  9. 5-class risk zonation",
              "",
              "  Reference: Nobre et al. (2011) — Journal of Hydrology 404:13–29",
              "=" * W, "  END OF REPORT", "=" * W]

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  {Path(out_path).name}")
    return text


def save_statistics_json(meta, slope, hand, flood_results, risk_stats, out_path):
    """Export all simulation statistics as JSON."""
    def _serial(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, dict):           return {k: _serial(v) for k, v in obj.items()}
        if isinstance(obj, list):           return [_serial(v) for v in obj]
        return obj

    payload = {
        "metadata"  : meta,
        "params"    : PARAMS,
        "terrain"   : {
            "mean_slope_deg": float(np.nanmean(slope)),
            "max_slope_deg" : float(np.nanmax(slope)),
            "hand_min"      : float(np.nanmin(hand)),
            "hand_max"      : float(np.nanmax(hand)),
            "hand_mean"     : float(np.nanmean(hand)),
        },
        "flood_scenarios": {
            k: {ik: iv for ik, iv in v.items() if ik not in ("mask", "depth")}
            for k, v in flood_results.items()
        },
        "risk_zones": risk_stats,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_serial(payload), fh, indent=2, ensure_ascii=False)
    print(f"  {Path(out_path).name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  JADE VALLEY FLOOD RISK SIMULATION")
    print("  Davao City, Philippines")
    print("=" * 60)
    print(f"  Date   : {PARAMS['analysis_date']}")
    print(f"  Output : {RESULTS_DIR}\n")

    print("Checking libraries…")
    libs = check_libraries()

    # ── Load data ─────────────────────────────────────────────────────────────
    if not TIF_FILE.exists():
        sys.exit(f"\n[ERROR] DEM file not found:\n  {TIF_FILE}")

    dem_raw, meta = load_tif_dem(TIF_FILE)

    print("\nLoading DXF files…")
    dxf_info = {
        "2D": load_dxf_info(DXF_2D, "2D Vectorial"),
        "3D": load_dxf_info(DXF_3D, "3D Model"),
    }

    # ── Hydrological analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  HYDROLOGICAL ANALYSIS")
    print("=" * 40)
    t0 = time.time()

    if libs.get("pysheds"):
        try:
            dem_proc, accum, hand, streams = run_pysheds(
                TIF_FILE, PARAMS["stream_threshold_pct"]
            )
            slope = calculate_slope(dem_proc, meta["cellsize"])
            print(f"\n[pysheds] Done in {time.time()-t0:.1f}s")
        except Exception as exc:
            print(f"\n[pysheds ERROR] {exc}\nUsing manual fallback…")
            dem_proc, _ = preprocess_dem(dem_raw, PARAMS["smooth_sigma"])
            slope        = calculate_slope(dem_proc, meta["cellsize"])
            flow_dir     = d8_flow_direction(dem_proc)
            accum        = flow_accumulation(flow_dir, dem_proc)
            streams      = delineate_streams(accum, PARAMS["stream_threshold_pct"])
            hand         = calculate_hand(dem_proc, streams, flow_dir)
            print(f"\n[manual] Done in {time.time()-t0:.1f}s")
    else:
        dem_proc, _ = preprocess_dem(dem_raw, PARAMS["smooth_sigma"])
        slope        = calculate_slope(dem_proc, meta["cellsize"])
        flow_dir     = d8_flow_direction(dem_proc)
        accum        = flow_accumulation(flow_dir, dem_proc)
        streams      = delineate_streams(accum, PARAMS["stream_threshold_pct"])
        hand         = calculate_hand(dem_proc, streams, flow_dir)
        print(f"\n[manual] Done in {time.time()-t0:.1f}s")

    # ── Flood simulation ───────────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  FLOOD SIMULATION")
    print("=" * 40)
    flood_results       = simulate_floods(hand, meta)
    risk_map, risk_stats = classify_risk(hand, meta)

    # ── Generate outputs ───────────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  GENERATING OUTPUTS")
    print("=" * 40)
    print("\nMaps →")
    save_terrain_overview(dem_proc, meta, slope, accum, streams,
                          str(MAPS_DIR / "01_terrain_analysis.png"))
    save_hand_map(hand, streams, meta, str(MAPS_DIR / "02_HAND_map.png"))
    save_flood_maps(dem_proc, flood_results, meta, MAPS_DIR)
    save_comparison_map(dem_proc, flood_results, meta,
                        str(MAPS_DIR / "07_flood_comparison.png"))
    save_risk_map(dem_proc, hand, risk_map, risk_stats, meta,
                  str(MAPS_DIR / "08_flood_risk_zones.png"))

    print("\nData →")
    save_statistics_json(meta, slope, hand, flood_results, risk_stats,
                         str(DATA_OUT / "flood_statistics.json"))
    generate_report(meta, slope, hand, flood_results, risk_stats,
                    str(DATA_OUT / "flood_risk_report.txt"))
    for name, arr in [("dem_processed", dem_proc), ("hand_model", hand),
                      ("flow_accumulation", accum), ("slope", slope),
                      ("risk_map", risk_map)]:
        np.save(str(DATA_OUT / f"{name}.npy"), arr)
    print(f"  numpy arrays (.npy) — 5 files")

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"\n  Location : {PARAMS['location']}")
    print(f"  Area     : {meta['area_m2']/10000:.1f} ha")
    print(f"  Duration : {elapsed:.1f}s\n")
    print("  Risk Zone Summary:")
    print(f"  {'Zone':<22}  {'Area (ha)':>10}  {'%':>6}  Bar")
    print(f"  {'-'*56}")
    for rname, st in risk_stats.items():
        bar = "█" * int(st["percentage"] / 2)
        print(f"  {rname:<22}  {st['area_ha']:>10.2f}  {st['percentage']:>5.1f}%  {bar}")
    print(f"\n  Results → {RESULTS_DIR}")
    print("=" * 60)

    return {
        "dem": dem_proc, "slope": slope, "accum": accum,
        "streams": streams, "hand": hand,
        "risk_map": risk_map, "flood_results": flood_results,
        "risk_stats": risk_stats, "metadata": meta,
    }


if __name__ == "__main__":
    main()
