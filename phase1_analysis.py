"""
╔══════════════════════════════════════════════════════════════════╗
║     FLOOD RISK SPATIAL ANALYSIS TOOL — Phase 1: Data Pipeline   ║
║     Author : [Your Name]                                         ║
║     Region : Tunis Coastal Area, Tunisia (36.6-37.2°N, 10-10.6°E)║
║     Models : SCS Curve Number + Bathtub Inundation               ║
╚══════════════════════════════════════════════════════════════════╝

Phase 1 covers:
  1. DEM loading and terrain analysis (slope, depressions, flow)
  2. Rainfall data ingestion (3 scenarios: normal / moderate / extreme)
  3. SCS Curve Number runoff model
  4. Coastal inundation (bathtub) model
  5. Risk zone classification (low / medium / high / extreme)
  6. Output: GeoTIFF flood rasters + GeoJSON vector zones + PNG dashboard
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from scipy.ndimage import gaussian_filter, binary_dilation, uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import json, os, requests

# ─── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    # Study area: Tunis coastal region
	"lat_min": 36.0, "lat_max": 37.5,
	"lon_min":  9.5, "lon_max": 11.2,
    "grid_size": 300,          # DEM resolution (300x300 pixels)
    "pixel_spacing_m": 200,    # approximate meters per pixel

    # SCS Curve Number defaults (55=forest, 90=impervious urban)
    "CN_hills": 65,
    "CN_coast": 85,

    # Flood thresholds (meters depth)
    "flood_threshold_m": 0.05,
    "risk_low_m":    0.05,
    "risk_medium_m": 0.25,
    "risk_high_m":   0.75,
    "risk_extreme_m": 1.50,

    # Output paths
    "data_dir": "data/",
    "output_dir": "outputs/",
}

os.makedirs(CONFIG["data_dir"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate / Load DEM
# ══════════════════════════════════════════════════════════════════════════════

def generate_dem(config):
    """Load real SRTM DEM instead of synthetic terrain."""
    import glob

    # Find the real DEM file — supports .hgt or .tif
    candidates = (
        glob.glob(config["data_dir"] + "*.hgt") +
        glob.glob(config["data_dir"] + "dem_real*.tif") +
        glob.glob(config["data_dir"] + "output_SRTM*.tif")
    )

    if not candidates:
        raise FileNotFoundError(
            "No real DEM found in data/. "
            "Download N36E010.hgt from https://dwtkns.com/srtm30m/ "
            "and put it in your data/ folder."
        )

    real_path = candidates[0]
    print(f"[DEM] Loading real SRTM data: {real_path}")

    with rasterio.open(real_path) as src:
        # Crop to the Tunis coastal region we care about
        from rasterio.windows import from_bounds
        window = from_bounds(
            config["lon_min"], config["lat_min"],
            config["lon_max"], config["lat_max"],
            src.transform
        )
        dem = src.read(1, window=window).astype(np.float32)
        transform = src.window_transform(window)
        crs = src.crs

    # Replace no-data values (SRTM uses -32768)
    dem[dem < -100] = 0

    # Clip to reasonable range
    dem = np.clip(dem, 0, 1000)

    rows, cols = dem.shape
    print(f"[DEM] Loaded: {rows}×{cols} pixels")
    print(f"      Elev  : {dem.min():.1f}m – {dem.max():.1f}m (mean {dem.mean():.1f}m)")
    print(f"      Coastal: {100*(dem<5).mean():.1f}% of area below 5m")

    # Save a copy as GeoTIFF for the rest of the pipeline
    out_path = config["data_dir"] + "dem_tunisia.tif"
    with rasterio.open(out_path, 'w', driver='GTiff',
                       height=rows, width=cols, count=1,
                       dtype=np.float32, crs=crs,
                       transform=transform, nodata=-9999) as dst:
        dst.write(dem, 1)

    print(f"[DEM] Saved cropped DEM: {out_path}")
    return dem, transform, crs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Rainfall Scenarios
# ══════════════════════════════════════════════════════════════════════════════

def get_rainfall_data(config: dict) -> dict:
    """
    In a real project: call Open-Meteo API (free, no key required):
      https://api.open-meteo.com/v1/forecast?latitude=36.8&longitude=10.18
      &daily=precipitation_sum&forecast_days=7

    Here we define three physically calibrated scenarios for Tunisia.
    Return periods are approximate (based on Mediterranean climate records).
    """
    data = {
        "location": {
            "name": "Tunis, Tunisia",
            "lat": 36.8, "lon": 10.18
        },
        "scenarios": {
            "normal": {
                "total_rain_mm": 2.0,
                "sea_level_rise_m": 0.0,
                "description": "Typical spring day — no flood risk",
                "return_period_years": "< 2"
            },
            "moderate": {
                "total_rain_mm": 55.5,
                "sea_level_rise_m": 0.3,
                "description": "Significant 3-day storm event",
                "return_period_years": "~10"
            },
            "extreme": {
                "total_rain_mm": 176.0,
                "sea_level_rise_m": 1.2,
                "description": "Catastrophic multi-day event",
                "return_period_years": "~100"
            }
        }
    }
    path = config["data_dir"] + "rainfall_scenarios.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n[RAIN] Scenarios saved: {path}")
    for name, s in data["scenarios"].items():
        print(f"       {name:8s}: {s['total_rain_mm']:6.1f}mm rain, "
              f"+{s['sea_level_rise_m']}m sea  ({s['description']})")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Terrain Derivatives
# ══════════════════════════════════════════════════════════════════════════════

def compute_terrain(dem: np.ndarray, config: dict) -> dict:
    """
    Derive secondary terrain attributes from the DEM:
      - slope_deg   : steepness of terrain (affects runoff speed)
      - CN_grid     : Curve Number (proxy for soil permeability / land use)
      - depression  : local topographic depressions (where water pools)
      - flow_acc    : simplified flow accumulation (river channels)
    """
    spacing = config["pixel_spacing_m"]

    # Slope in degrees
    dy, dx = np.gradient(dem, spacing, spacing)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Curve Number: higher CN = more runoff (impervious/urban)
    # Lower elevation → more urban/coastal → higher CN
    cn_range = config["CN_coast"] - config["CN_hills"]
    CN = config["CN_hills"] + (1 - dem / dem.max()) * cn_range
    CN = np.clip(CN, 50, 92).astype(np.float32)

    # Local depressions: where water pools (local mean minus actual elevation)
    local_mean = uniform_filter(dem.astype(float), size=15)
    depression = np.maximum(0, local_mean - dem)

    # River channels: deep, connected depressions
    flow_acc = gaussian_filter(depression, sigma=4)
    flow_acc /= max(flow_acc.max(), 1)  # normalize 0–1

    print(f"\n[TERRAIN] Slope: {slope_deg.min():.1f}° – {slope_deg.max():.1f}°")
    print(f"          CN: {CN.min():.0f} – {CN.max():.0f}")
    print(f"          Depression depth: max {depression.max():.1f}m")

    return {
        "slope_deg": slope_deg,
        "CN": CN,
        "depression": depression,
        "flow_acc": flow_acc
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — SCS Curve Number Runoff Model
# ══════════════════════════════════════════════════════════════════════════════

def scs_runoff(rain_mm: float, CN: np.ndarray) -> np.ndarray:
    """
    USDA Soil Conservation Service Curve Number method.
    Converts total rainfall (mm) to surface runoff (mm).

    Formula:
        S  = (25400 / CN) - 254          retention capacity (mm)
        Ia = 0.2 × S                     initial abstraction (mm)
        Q  = (P - Ia)² / (P - Ia + S)   runoff depth (mm), if P > Ia

    CN=90 (urban): most rain becomes runoff
    CN=65 (forest): rain mostly infiltrates into soil
    """
    S = (25400.0 / CN) - 254.0  # maximum retention
    Ia = 0.2 * S                 # initial abstraction
    Q = np.where(
        rain_mm > Ia,
        (rain_mm - Ia) ** 2 / (rain_mm - Ia + S),
        0.0
    )
    return Q.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Flood Simulation
# ══════════════════════════════════════════════════════════════════════════════

def bathtub_coastal_flood(dem: np.ndarray, water_level_m: float) -> np.ndarray:
    """
    Bathtub model: all terrain connected to the coast (left edge)
    and below water_level is considered flooded.
    Uses binary dilation to enforce spatial connectivity
    (no 'floating' inland lakes unless truly connected to coast).
    """
    below_wl = dem <= water_level_m
    # Seed from coastal edges
    seed = np.zeros_like(below_wl)
    seed[:, 0] = below_wl[:, 0]
    seed[0, :] = below_wl[0, :]
    seed[-1, :] = below_wl[-1, :]
    # Dilate flood through connected low pixels
    for _ in range(80):
        expanded = binary_dilation(seed) & below_wl
        if (expanded == seed).all():
            break
        seed = expanded
    return seed


def simulate_flood(dem: np.ndarray, terrain: dict,
                   scenario: dict, transform, crs,
                   config: dict) -> tuple[np.ndarray, dict]:
    """
    Combine three flood mechanisms:
      1. Coastal inundation  (sea level rise + storm surge)
      2. Depression pooling  (runoff accumulates in low spots)
      3. River/channel flood (runoff follows valley network)
    """
    rain       = scenario["total_rain_mm"]
    sea_rise   = scenario["sea_level_rise_m"]
    CN         = terrain["CN"]
    depression = terrain["depression"]
    flow_acc   = terrain["flow_acc"]

    # 1. Coastal inundation
    water_level = sea_rise + rain * 0.004
    coastal_depth = bathtub_coastal_flood(dem, water_level).astype(float)
    coastal_depth *= np.maximum(0, water_level - dem)

    # 2. SCS runoff → depression pooling
    runoff = scs_runoff(rain, CN)
    pool_depth = depression * (runoff / 300.0)
    pool_depth = gaussian_filter(pool_depth, sigma=2)

    # 3. River channel overflow
    river_mask = depression > 8
    river_depth = np.where(river_mask, runoff / 80.0, 0)
    river_depth = gaussian_filter(river_depth, sigma=3)

    # Combine
    total = np.maximum(0, coastal_depth + pool_depth + river_depth).astype(np.float32)

    # Risk classification
    ft = config["flood_threshold_m"]
    risk = np.zeros(dem.shape, dtype=np.uint8)
    risk[total > ft]                              = 1  # low
    risk[total > config["risk_medium_m"]]         = 2  # medium
    risk[total > config["risk_high_m"]]           = 3  # high
    risk[total > config["risk_extreme_m"]]        = 4  # extreme

    # Area statistics
    bounds = rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], transform)
    pixel_km2 = ((bounds[2]-bounds[0]) * (bounds[3]-bounds[1]) / dem.size) * (111**2)
    flooded = total > ft

    stats = {
        "description":        scenario["description"],
        "total_rain_mm":      rain,
        "sea_level_rise_m":   sea_rise,
        "mean_runoff_mm":     round(float(runoff.mean()), 2),
        "flooded_area_km2":   round(flooded.sum() * pixel_km2, 2),
        "flooded_pct":        round(100 * flooded.mean(), 2),
        "max_depth_m":        round(float(total.max()), 3),
        "mean_depth_m":       round(float(total[flooded].mean()) if flooded.any() else 0, 3),
        "risk_counts": {
            "none":    int((risk == 0).sum()),
            "low":     int((risk == 1).sum()),
            "medium":  int((risk == 2).sum()),
            "high":    int((risk == 3).sum()),
            "extreme": int((risk == 4).sum()),
        }
    }
    return total, stats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Save Outputs (GeoTIFF + GeoJSON)
# ══════════════════════════════════════════════════════════════════════════════

def save_flood_raster(depth: np.ndarray, scenario_name: str,
                      transform, crs, config: dict):
    """Save flood depth as GeoTIFF (opens in QGIS, ArcGIS, GRASS)"""
    path = config["data_dir"] + f"flood_{scenario_name}.tif"
    rows, cols = depth.shape
    with rasterio.open(path, 'w', driver='GTiff',
                       height=rows, width=cols, count=1,
                       dtype=np.float32, crs=crs,
                       transform=transform, nodata=-9999) as dst:
        dst.write(depth, 1)
    return path


def save_flood_geojson(depth: np.ndarray, dem: np.ndarray,
                       scenario_name: str, stats: dict,
                       transform, config: dict, step: int = 6):
    """
    Sample the flood raster to a GeoJSON FeatureCollection.
    Each feature is a small polygon with depth and risk attributes.
    This is the format that Leaflet.js will consume in Phase 3.
    """
    features = []
    for r in range(0, depth.shape[0] - step, step):
        for c in range(0, depth.shape[1] - step, step):
            d = float(depth[r, c])
            if d < config["flood_threshold_m"]:
                continue
            lon, lat = rasterio.transform.xy(transform, r, c)
            hl = abs(transform.a) * step / 2
            hh = abs(transform.e) * step / 2

            if d > config["risk_extreme_m"]:    risk = "extreme"
            elif d > config["risk_high_m"]:     risk = "high"
            elif d > config["risk_medium_m"]:   risk = "medium"
            else:                                risk = "low"

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon-hl, lat-hh], [lon+hl, lat-hh],
                        [lon+hl, lat+hh], [lon-hl, lat+hh],
                        [lon-hl, lat-hh]
                    ]]
                },
                "properties": {
                    "depth_m":     round(d, 2),
                    "elevation_m": round(float(dem[r, c]), 1),
                    "risk_level":  risk,
                    "risk_score":  {"low":1,"medium":2,"high":3,"extreme":4}[risk]
                }
            })

    geojson = {
        "type": "FeatureCollection",
        "metadata": {"scenario": scenario_name, **stats},
        "features": features
    }
    path = config["data_dir"] + f"flood_{scenario_name}.geojson"
    with open(path, 'w') as f:
        json.dump(geojson, f)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Visualization Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def build_dashboard(dem, floods, all_stats, transform, config):
    """
    Generate a 4-panel scientific visualization dashboard:
      - DEM elevation map
      - Flood maps for each scenario
      - Risk zone bar chart
      - Elevation histogram with thresholds
      - Scenario summary statistics
    """
    bounds = rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], transform)

    fig = plt.figure(figsize=(18, 14), facecolor='#0d1117')
    fig.suptitle(
        'Flood Risk Spatial Analysis — Tunis Coastal Region, Tunisia',
        fontsize=18, fontweight='bold', color='white', y=0.98
    )
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.05, right=0.97, top=0.94, bottom=0.06)

    flood_cmap = mcolors.LinearSegmentedColormap.from_list('flood', [
        (0,    'none'), (0.001,'#4fc3f780'), (0.15, '#0288d1bb'),
        (0.5,  '#e65100cc'), (1, '#b71c1c')
    ], N=256)

    # Panel 1: DEM
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dem, cmap='terrain', vmin=-2, vmax=160,
                     extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    ax1.set_title('Digital Elevation Model', color='white', fontsize=11, fontweight='bold', pad=6)
    ax1.set_xlabel('Longitude (E)', color='#aaa', fontsize=8)
    ax1.set_ylabel('Latitude (N)', color='#aaa', fontsize=8)
    ax1.tick_params(colors='#888', labelsize=7)
    for s in ax1.spines.values(): s.set_edgecolor('#333')
    cb = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label('Elevation (m)', color='#aaa', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='#aaa', labelsize=7)
    plt.setp(cb.ax.get_yticklabels(), color='#aaa')

    # Panels 2-4: Flood maps
    titles  = ['Normal (2mm)', 'Moderate (55mm)', 'Extreme (176mm)']
    s_names = ['normal', 'moderate', 'extreme']
    s_cols  = ['#4caf50', '#ff9800', '#f44336']
    for i, (sn, st, sc) in enumerate(zip(s_names, titles, s_cols)):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(dem, cmap='Greys_r', vmin=-5, vmax=180, alpha=0.7,
                  extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
        flood_ma = np.ma.masked_where(floods[sn] < 0.05, floods[sn])
        ax.imshow(flood_ma, cmap=flood_cmap, vmin=0.05, vmax=3.0, alpha=0.9,
                  extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
        r = all_stats[sn]
        ax.set_title(f"{st}\n{r['flooded_area_km2']} km² flooded",
                     color=sc, fontsize=9, fontweight='bold', pad=4)
        ax.tick_params(colors='#888', labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.legend(handles=[
            Patch(facecolor='#4fc3f7', alpha=0.7, label='Low (<0.25m)'),
            Patch(facecolor='#0288d1', alpha=0.8, label='Med (0.25-0.75m)'),
            Patch(facecolor='#e65100', alpha=0.9, label='High (>0.75m)'),
        ], loc='lower right', fontsize=6, facecolor='#1a1a2e',
           edgecolor='#333', labelcolor='white')

    # Panel 5: Risk bar chart
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.set_facecolor('#161b22')
    x = np.arange(3)
    risk_keys = ['low','medium','high','extreme']
    risk_cols = ['#4fc3f7','#ff9800','#e65100','#b71c1c']
    pixel_km2 = ((bounds[2]-bounds[0])*(bounds[3]-bounds[1])/dem.size)*(111**2)
    for j, (rk, rc) in enumerate(zip(risk_keys, risk_cols)):
        vals = [all_stats[s]['risk_counts'][rk]*pixel_km2 for s in s_names]
        bars = ax5.bar(x+j*0.18-0.27, vals, 0.18, label=rk.capitalize(), color=rc, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0.5:
                ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                         f'{v:.0f}', ha='center', va='bottom', color='white', fontsize=7)
    ax5.set_xticks(x); ax5.set_xticklabels(['Normal','Moderate','Extreme'], color='white', fontsize=10)
    ax5.set_ylabel('Area at Risk (km²)', color='#aaa', fontsize=9)
    ax5.set_title('Risk Zone Areas by Scenario', color='white', fontsize=11, fontweight='bold')
    ax5.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white', fontsize=8)
    ax5.tick_params(colors='#888')
    ax5.spines['bottom'].set_color('#333'); ax5.spines['left'].set_color('#333')
    ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

    # Panel 6: Elevation histogram
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.set_facecolor('#161b22')
    n, bins, patches = ax6.hist(dem.flatten(), bins=80, edgecolor='none', alpha=0.75)
    for patch, left in zip(patches, bins[:-1]):
        if left < 5:    patch.set_facecolor('#4fc3f7')
        elif left < 15: patch.set_facecolor('#0288d1')
        elif left < 40: patch.set_facecolor('#4a90d9')
        else:           patch.set_facecolor('#4a7ab5')
    for t, c, lbl in [(0.3,'#ff9800','Moderate +0.3m'),(1.2,'#f44336','Extreme +1.2m'),(5.0,'#b71c1c','5m threshold')]:
        ax6.axvline(t, color=c, lw=1.5, ls='--', alpha=0.9)
        ax6.text(t+0.8, n.max()*0.85, lbl, color=c, fontsize=7, rotation=90, va='top')
    ax6.set_xlabel('Elevation (m)', color='#aaa', fontsize=9)
    ax6.set_ylabel('Pixel count', color='#aaa', fontsize=9)
    ax6.set_title('Elevation Distribution + Flood Thresholds', color='white', fontsize=11, fontweight='bold')
    ax6.tick_params(colors='#888', labelsize=8)
    ax6.set_xlim(-5, 100)
    ax6.spines['bottom'].set_color('#333'); ax6.spines['left'].set_color('#333')
    ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False)

    # Panel 7: Summary stats
    ax7 = fig.add_subplot(gs[2, :])
    ax7.set_facecolor('#0d1117'); ax7.axis('off')
    card_info = [
        ('Normal\n(2mm/day)',   '#1a2744', '#4fc3f7', 'normal'),
        ('Moderate\n(55mm)',    '#2a1a00', '#ff9800', 'moderate'),
        ('Extreme\n(176mm)',    '#2a0a0a', '#f44336', 'extreme'),
    ]
    for idx, (title, bg, col, sn) in enumerate(card_info):
        x0 = 0.02 + idx * 0.33
        ax7.add_patch(plt.Rectangle((x0, 0.05), 0.30, 0.88,
                                     transform=ax7.transAxes,
                                     facecolor=bg, edgecolor='#333', lw=0.8, zorder=1))
        ax7.text(x0+0.15, 0.82, title, transform=ax7.transAxes,
                 color=col, fontsize=11, fontweight='bold', ha='center', va='center', zorder=2)
        r = all_stats[sn]
        ax7.text(x0+0.15, 0.42,
                 f"Rainfall:     {r['total_rain_mm']} mm total\n"
                 f"Flooded:      {r['flooded_area_km2']} km²  ({r['flooded_pct']}%)\n"
                 f"Max depth:    {r['max_depth_m']} m\n"
                 f"Mean depth:   {r['mean_depth_m']} m\n"
                 f"High+Extreme: {r['risk_counts']['high']+r['risk_counts']['extreme']} grid cells",
                 transform=ax7.transAxes, color='#e0e0e0', fontsize=9,
                 ha='center', va='center', fontfamily='monospace', zorder=2, linespacing=1.8)
    ax7.set_title('Scenario Summary Statistics', color='white', fontsize=11, fontweight='bold')

    path = config["output_dir"] + "flood_analysis_phase1.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"\n[VIZ] Dashboard saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Run the full Phase 1 pipeline
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  FLOOD RISK ANALYSIS — Phase 1 Pipeline Starting")
    print("=" * 60)

    # 1. Load DEM
    dem, transform, crs = generate_dem(CONFIG)

    # 2. Rainfall scenarios
    rainfall_data = get_rainfall_data(CONFIG)

    # 3. Terrain derivatives
    terrain = compute_terrain(dem, CONFIG)

    # 4 & 5. Simulate floods for each scenario
    floods = {}
    all_stats = {}
    print("\n[FLOOD MODEL] Running simulations...")
    for name, scenario in rainfall_data["scenarios"].items():
        depth, stats = simulate_flood(dem, terrain, scenario, transform, crs, CONFIG)
        floods[name] = depth
        all_stats[name] = stats
        print(f"  {name:8s} → {stats['flooded_area_km2']:6.1f} km² flooded "
              f"| max depth {stats['max_depth_m']}m "
              f"| high+extreme risk: {stats['risk_counts']['high']+stats['risk_counts']['extreme']} cells")

    # 6. Save outputs
    print("\n[SAVE] Writing GeoTIFF and GeoJSON files...")
    for name in ['normal', 'moderate', 'extreme']:
        tif  = save_flood_raster(floods[name], name, transform, crs, CONFIG)
        gjsn = save_flood_geojson(floods[name], dem, name, all_stats[name], transform, CONFIG)
        print(f"  {name:8s} → {tif}  |  {gjsn}")

    with open(CONFIG["data_dir"] + "analysis_results.json", 'w') as f:
        json.dump({"dem_info": {"rows": dem.shape[0], "cols": dem.shape[1]}, "scenarios": all_stats}, f, indent=2)

    # 7. Visualization
    build_dashboard(dem, floods, all_stats, transform, CONFIG)

    print("\n" + "=" * 60)
    print("  Phase 1 COMPLETE. Files in data/ and outputs/")
    print("  Next: Phase 2 = add ML risk prediction")
    print("        Phase 3 = build the Leaflet.js web map")
    print("=" * 60)
