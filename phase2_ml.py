"""
╔══════════════════════════════════════════════════════════════════╗
║     FLOOD RISK SPATIAL ANALYSIS TOOL — Phase 2: ML Prediction   ║
║     Model  : Random Forest Classifier (scikit-learn)             ║
║     Input  : Phase 1 DEM + flood rasters                        ║
║     Output : ML risk map (GeoTIFF + GeoJSON) + dashboard        ║
╚══════════════════════════════════════════════════════════════════╝

What this phase adds:
  - Extracts 6 terrain features from the DEM
  - Uses Phase 1 physics results as training labels
  - Trains a Random Forest to learn which terrain predicts flood risk
  - Produces a continuous risk probability map (0–1 per pixel)
  - Output is ready to layer on the Leaflet.js map in Phase 3

Why ML on top of physics?
  The physics model is accurate but slow — it must re-run the full
  SCS + bathtub calculation for every new scenario. The ML model
  learns the pattern once, then predicts instantly for any input.
"""

import numpy as np
import os, json
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ─── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir":    "data/",
    "output_dir":  "outputs/",

    # Random Forest settings
    "n_estimators":     100,   # number of trees — more = more accurate, but slower
    "max_depth":        12,    # how deep each tree grows
    "min_samples_leaf": 20,    # prevents overfitting on tiny clusters
    "test_size":        0.25,  # 25% of pixels held out for evaluation

    # Risk thresholds (metres flood depth, same as Phase 1)
    "risk_low_m":     0.05,
    "risk_medium_m":  0.25,
    "risk_high_m":    0.75,
    "risk_extreme_m": 1.50,

    # GeoJSON sampling step (every N pixels to keep file size manageable)
    "geojson_step": 6,
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Phase 1 outputs
# ══════════════════════════════════════════════════════════════════════════════

def load_phase1_data(config):
    """Load the DEM and flood depth raster produced by phase1_analysis.py"""
    with rasterio.open(config["data_dir"] + "dem_tunisia.tif") as src:
        dem       = src.read(1).astype(np.float32)
        transform = src.transform
        crs       = src.crs
        bounds    = src.bounds

    # We use the extreme scenario as training labels — it activates all risk zones
    with rasterio.open(config["data_dir"] + "flood_extreme.tif") as src:
        flood_extreme = src.read(1).astype(np.float32)

    print(f"[DATA] DEM loaded: {dem.shape[0]}×{dem.shape[1]} pixels")
    print(f"       Flood range: {flood_extreme.min():.2f}m – {flood_extreme.max():.2f}m")
    return dem, flood_extreme, transform, crs, bounds


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(dem, bounds, config):
    """
    Extract 6 terrain features from the DEM.
    Each feature captures a different aspect of flood susceptibility.

    Features:
      1. elevation    — low areas are more likely to flood
      2. slope        — steep areas drain fast, flat areas pool water
      3. twi          — Topographic Wetness Index: ln(flow_acc / tan(slope))
                        high TWI = naturally wet, low-drainage terrain
      4. coast_dist   — proximity to coast (coastal flooding source)
      5. relief       — local height variation; low relief = flat = prone to pooling
      6. curvature    — negative = concave (bowl) = water accumulates there

    Returns:
      features  — ndarray of shape (rows, cols, 6)
      names     — list of 6 feature name strings
    """
    rows, cols = dem.shape
    spacing = 200  # approximate metres per pixel

    # 1. Elevation — direct DEM value
    feat_elev = dem.copy()

    # 2. Slope in degrees
    dy, dx = np.gradient(dem, spacing, spacing)
    feat_slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)

    # 3. Topographic Wetness Index
    slope_rad     = np.maximum(np.arctan(np.sqrt(dx**2 + dy**2)), 0.001)
    local_mean    = uniform_filter(dem.astype(float), size=15)
    depression    = np.maximum(0, local_mean - dem)
    flow_acc      = gaussian_filter(depression + 1, sigma=6)
    feat_twi      = np.log(flow_acc / np.tan(slope_rad)).astype(np.float32)

    # 4. Distance to coast (left edge of our DEM = coastline)
    coast_lon = bounds.left + (bounds.right - bounds.left) * 0.18
    lon_grid  = np.linspace(bounds.left, bounds.right, cols)
    feat_coast_dist = np.abs(np.tile(lon_grid, (rows, 1)) - coast_lon).astype(np.float32)

    # 5. Local relief (max - min in a 20-pixel neighbourhood)
    local_max   = uniform_filter(dem, size=20)
    local_min   = -uniform_filter(-dem, size=20)
    feat_relief = (local_max - local_min).astype(np.float32)

    # 6. Curvature — negative means concave (hollows where water collects)
    feat_curv = (dem - gaussian_filter(dem, sigma=5)).astype(np.float32)

    features = np.stack([
        feat_elev, feat_slope, feat_twi,
        feat_coast_dist, feat_relief, feat_curv
    ], axis=-1)

    names = ['elevation', 'slope', 'twi', 'coast_dist', 'relief', 'curvature']
    print(f"\n[FEATURES] Matrix: {features.shape}  ({len(names)} features per pixel)")
    return features, names


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build training labels from physics model
# ══════════════════════════════════════════════════════════════════════════════

def build_labels(flood_extreme, config):
    """
    Convert continuous flood depth (metres) to 5-class risk label.
    This turns the Phase 1 physics output into supervised ML targets.

    Classes:
      0 = no flood       (depth < 0.05m)
      1 = low risk       (0.05 – 0.25m)
      2 = medium risk    (0.25 – 0.75m)
      3 = high risk      (0.75 – 1.50m)
      4 = extreme risk   (> 1.50m)
    """
    labels = np.zeros(flood_extreme.shape, dtype=np.uint8)
    labels[flood_extreme > config["risk_low_m"]]     = 1
    labels[flood_extreme > config["risk_medium_m"]]  = 2
    labels[flood_extreme > config["risk_high_m"]]    = 3
    labels[flood_extreme > config["risk_extreme_m"]] = 4

    print(f"\n[LABELS] Class distribution:")
    class_names = {0:'none', 1:'low', 2:'medium', 3:'high', 4:'extreme'}
    for cls, name in class_names.items():
        count = (labels == cls).sum()
        print(f"         {name:8s}: {count:6d} pixels  ({100*count/labels.size:.1f}%)")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train & evaluate Random Forest
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(features, labels, config):
    """
    Train a Random Forest classifier.

    Why Random Forest?
      - Handles non-linear relationships between terrain and flood risk
      - Works well with mixed feature scales (no normalisation needed)
      - Gives feature importance scores for free
      - Robust — 100 trees average out individual tree errors
      - Fast prediction after training (instant for 90,000 pixels)

    Returns the trained model and evaluation statistics.
    """
    rows, cols, n_feat = features.shape
    X = features.reshape(-1, n_feat)   # (90000, 6) — one row per pixel
    y = labels.flatten()               # (90000,)   — one label per pixel

    # Stratified split — ensures all 5 risk classes appear in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=42,
        stratify=y
    )
    print(f"\n[ML] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,} samples")

    # Train
    print("     Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators  = config["n_estimators"],
        max_depth     = config["max_depth"],
        min_samples_leaf = config["min_samples_leaf"],
        class_weight  = 'balanced',   # upweights rare classes (high/extreme)
        random_state  = 42,
        n_jobs        = -1            # use all CPU cores
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred   = rf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    report   = classification_report(
        y_test, y_pred,
        target_names=['none','low','medium','high','extreme'],
        zero_division=0
    )
    print(f"\n[EVAL] Overall accuracy: {accuracy*100:.1f}%")
    print(report)

    # Feature importance
    print("[FEATURES] Importance ranking:")
    feature_names = ['elevation','slope','twi','coast_dist','relief','curvature']
    for name, imp in sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f"           {name:12s}: {imp:.3f}  {bar}")

    return rf, accuracy


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Predict full map & save outputs
# ══════════════════════════════════════════════════════════════════════════════

def predict_and_save(rf, features, dem, transform, crs, bounds, accuracy, config):
    """
    Run the trained model on every pixel and save:
      - risk_ml.tif          — discrete risk class (0–4) GeoTIFF
      - risk_ml_proba.tif    — continuous high-risk probability (0–1) GeoTIFF
      - risk_ml.geojson      — vector polygons for the Leaflet.js web map
      - ml_model_meta.json   — model metadata for the dashboard
    """
    rows, cols, n_feat = features.shape
    X = features.reshape(-1, n_feat)

    print("\n[PREDICT] Running model on full 300×300 grid...")
    risk_ml   = rf.predict(X).reshape(rows, cols).astype(np.uint8)
    risk_proba = rf.predict_proba(X)                     # shape: (90000, 5)
    high_proba = (risk_proba[:, 3] + risk_proba[:, 4])   # P(high) + P(extreme)
    high_proba = high_proba.reshape(rows, cols).astype(np.float32)

    # Save discrete risk raster
    tif1 = config["data_dir"] + "risk_ml.tif"
    with rasterio.open(tif1, 'w', driver='GTiff',
                       height=rows, width=cols, count=1,
                       dtype=np.uint8, crs=crs, transform=transform, nodata=255) as dst:
        dst.write(risk_ml, 1)

    # Save probability raster
    tif2 = config["data_dir"] + "risk_ml_proba.tif"
    with rasterio.open(tif2, 'w', driver='GTiff',
                       height=rows, width=cols, count=1,
                       dtype=np.float32, crs=crs, transform=transform, nodata=-9999) as dst:
        dst.write(high_proba, 1)

    # Build GeoJSON for web map
    step = config["geojson_step"]
    geo_features = []
    risk_labels  = {0:'none', 1:'low', 2:'medium', 3:'high', 4:'extreme'}
    for r in range(0, rows - step, step):
        for c in range(0, cols - step, step):
            rv = int(risk_ml[r, c])
            pv = float(high_proba[r, c])
            if rv == 0 and pv < 0.05:
                continue
            lon, lat = rasterio.transform.xy(transform, r, c)
            hl = abs(transform.a) * step / 2
            hh = abs(transform.e) * step / 2
            geo_features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[
                    [lon-hl, lat-hh], [lon+hl, lat-hh],
                    [lon+hl, lat+hh], [lon-hl, lat+hh],
                    [lon-hl, lat-hh]
                ]]},
                "properties": {
                    "risk_level":           risk_labels[rv],
                    "risk_score":           rv,
                    "high_risk_probability": round(pv, 3),
                    "elevation_m":          round(float(dem[r, c]), 1),
                }
            })

    feature_names = ['elevation','slope','twi','coast_dist','relief','curvature']
    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "model":    "RandomForestClassifier",
            "accuracy": round(float(accuracy), 4),
            "features": feature_names
        },
        "features": geo_features
    }
    gjsn = config["data_dir"] + "risk_ml.geojson"
    with open(gjsn, 'w') as f:
        json.dump(geojson, f)

    # Save metadata
    meta = {
        "model": "RandomForestClassifier",
        "n_estimators": config["n_estimators"],
        "max_depth":    config["max_depth"],
        "accuracy":     round(float(accuracy), 4),
        "feature_names": feature_names,
        "feature_importances": {
            n: round(float(i), 4)
            for n, i in zip(feature_names, rf.feature_importances_)
        },
        "class_names": ['none','low','medium','high','extreme'],
    }
    with open(config["data_dir"] + "ml_model_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVE] {tif1}")
    print(f"       {tif2}")
    print(f"       {gjsn}  ({len(geo_features)} features)")
    print(f"       {config['data_dir']}ml_model_meta.json")

    return risk_ml, high_proba, meta


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Visualization dashboard
# ══════════════════════════════════════════════════════════════════════════════

def build_dashboard(dem, flood_extreme, risk_ml, high_proba, meta, bounds, config):
    """Generate a 6-panel Phase 2 analysis dashboard."""
    ext = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    risk_cmap = mcolors.ListedColormap(['#1a1a2e','#4fc3f7','#ff9800','#e65100','#b71c1c'])
    risk_norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5], risk_cmap.N)

    # Build physics risk for comparison
    physics_risk = np.zeros_like(dem, dtype=np.uint8)
    physics_risk[flood_extreme > config["risk_low_m"]]     = 1
    physics_risk[flood_extreme > config["risk_medium_m"]]  = 2
    physics_risk[flood_extreme > config["risk_high_m"]]    = 3
    physics_risk[flood_extreme > config["risk_extreme_m"]] = 4

    fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
    fig.suptitle(
        f'Phase 2 — ML Flood Risk Prediction  |  Random Forest  ·  Accuracy: {meta["accuracy"]*100:.1f}%',
        fontsize=16, fontweight='bold', color='white', y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.28,
                           left=0.05, right=0.97, top=0.93, bottom=0.07)

    # Row 1: three risk maps
    for idx, (data, title, col) in enumerate([
        (physics_risk, 'Physics model risk\n(Phase 1 — SCS + Bathtub)', '#4fc3f7'),
        (risk_ml,      'ML model risk\n(Phase 2 — Random Forest)',      '#ff9800'),
    ]):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(data, cmap=risk_cmap, norm=risk_norm, extent=ext)
        ax.set_title(title, color=col, fontsize=10, fontweight='bold')
        ax.tick_params(colors='#888', labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor('#333')

    legend_elements = [
        Patch(facecolor=c, label=n, edgecolor='#555' if c == '#1a1a2e' else c)
        for c, n in zip(['#1a1a2e','#4fc3f7','#ff9800','#e65100','#b71c1c'],
                        ['None','Low','Medium','High','Extreme'])
    ]
    fig.axes[1].legend(handles=legend_elements, loc='lower right', fontsize=7,
                       facecolor='#161b22', edgecolor='#444', labelcolor='white')

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(high_proba, cmap='YlOrRd', vmin=0, vmax=1, extent=ext)
    ax3.set_title('High-risk probability\n(continuous 0–1 score)', color='#f44336', fontsize=10, fontweight='bold')
    ax3.tick_params(colors='#888', labelsize=7)
    for s in ax3.spines.values():
        s.set_edgecolor('#333')
    cb = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb.set_label('P(high or extreme)', color='#aaa', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='#aaa')
    plt.setp(cb.ax.get_yticklabels(), color='#aaa')

    # Feature importance
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor('#161b22')
    names = list(meta['feature_importances'].keys())
    vals  = list(meta['feature_importances'].values())
    colors_bar = ['#b71c1c' if v == max(vals) else '#378ADD' for v in vals]
    bars = ax4.barh(names, vals, color=colors_bar, alpha=0.85, height=0.6)
    for bar, v in zip(bars, vals):
        ax4.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{v:.3f}', va='center', color='white', fontsize=8)
    ax4.set_xlabel('Importance score', color='#aaa', fontsize=9)
    ax4.set_title('Feature importance', color='white', fontsize=10, fontweight='bold')
    ax4.tick_params(colors='#aaa', labelsize=8)
    ax4.set_xlim(0, 0.6)
    ax4.spines['bottom'].set_color('#333'); ax4.spines['left'].set_color('#333')
    ax4.spines['top'].set_visible(False);  ax4.spines['right'].set_visible(False)

    # Physics vs ML comparison
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor('#161b22')
    x = np.arange(5)
    pixel_km2 = ((bounds.right - bounds.left) * (bounds.top - bounds.bottom) / physics_risk.size) * (111**2)
    phys_counts = [(physics_risk == i).sum() * pixel_km2 for i in range(5)]
    ml_counts   = [(risk_ml == i).sum() * pixel_km2 for i in range(5)]
    ax5.bar(x - 0.2, phys_counts, 0.38, label='Physics', color='#378ADD', alpha=0.85)
    ax5.bar(x + 0.2, ml_counts,   0.38, label='ML',      color='#ff9800', alpha=0.85)
    ax5.set_xticks(x)
    ax5.set_xticklabels(['none','low','med','high','extreme'], color='white', fontsize=8)
    ax5.set_ylabel('Area (km²)', color='#aaa', fontsize=9)
    ax5.set_title('Physics vs ML zones', color='white', fontsize=10, fontweight='bold')
    ax5.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white', fontsize=9)
    ax5.tick_params(colors='#888')
    ax5.spines['bottom'].set_color('#333'); ax5.spines['left'].set_color('#333')
    ax5.spines['top'].set_visible(False);  ax5.spines['right'].set_visible(False)

    # Summary card
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax6.transAxes,
                                 facecolor='#0d1521', edgecolor='#333', lw=0.8))
    summary_lines = (
        f"Model:           Random Forest\n"
        f"Trees:           {meta['n_estimators']}\n"
        f"Max depth:       {meta['max_depth']}\n\n"
        f"Overall accuracy: {meta['accuracy']*100:.1f}%\n\n"
        f"Top predictors:\n"
        + "\n".join(
            f"  {n:12s} {v*100:.1f}%"
            for n, v in sorted(meta['feature_importances'].items(), key=lambda x: -x[1])[:4]
        )
    )
    ax6.text(0.08, 0.93, summary_lines, transform=ax6.transAxes,
             color='#e0e0e0', fontsize=9, va='top',
             fontfamily='monospace', linespacing=1.7, zorder=2)
    ax6.set_title('Model summary', color='white', fontsize=10, fontweight='bold')

    os.makedirs(config["output_dir"], exist_ok=True)
    out = config["output_dir"] + "flood_analysis_phase2.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"\n[VIZ] Dashboard saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  FLOOD RISK ANALYSIS — Phase 2 ML Pipeline")
    print("=" * 60)

    dem, flood_extreme, transform, crs, bounds = load_phase1_data(CONFIG)
    features, feature_names = engineer_features(dem, bounds, CONFIG)
    labels = build_labels(flood_extreme, CONFIG)
    rf, accuracy = train_random_forest(features, labels, CONFIG)
    risk_ml, high_proba, meta = predict_and_save(
        rf, features, dem, transform, crs, bounds, accuracy, CONFIG
    )
    build_dashboard(dem, flood_extreme, risk_ml, high_proba, meta, bounds, CONFIG)

    print("\n" + "=" * 60)
    print("  Phase 2 COMPLETE.")
    print("  New files in data/:  risk_ml.tif  risk_ml_proba.tif")
    print("                       risk_ml.geojson  ml_model_meta.json")
    print("  Next: Phase 3 = Leaflet.js interactive web map")
    print("=" * 60)
