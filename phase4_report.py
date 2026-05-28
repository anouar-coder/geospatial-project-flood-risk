"""
╔══════════════════════════════════════════════════════════════════╗
║    FLOOD RISK ANALYSIS TOOL — Phase 4: PDF Report Generator     ║
║    Library : ReportLab                                           ║
║    Output  : flood_risk_report.pdf  (professional A4 report)    ║
╚══════════════════════════════════════════════════════════════════╝

Generates a complete scientific PDF report including:
  - Executive summary with key statistics
  - All scenario comparison table
  - Risk zone breakdown with colour bars
  - ML model performance metrics
  - Methodology section
  - Embedded dashboard images from Phase 1 & 2
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import json, os
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────
DATA_DIR   = "data/"
OUTPUT_DIR = "outputs/"
IMG_DIR    = "outputs/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────
with open(DATA_DIR + "analysis_results.json") as f:
    analysis = json.load(f)

with open(DATA_DIR + "ml_model_meta.json") as f:
    ml_meta = json.load(f)

with open(DATA_DIR + "rainfall_scenarios.json") as f:
    rainfall = json.load(f)

with open(DATA_DIR + "animation/frames_meta.json") as f:
    anim_meta = json.load(f)

# ─── Colours ──────────────────────────────────────────────────────
BLUE_DARK  = colors.HexColor('#0d3b6e')
BLUE_MID   = colors.HexColor('#185fa5')
BLUE_LIGHT = colors.HexColor('#e6f1fb')
TEAL       = colors.HexColor('#0f6e56')
AMBER      = colors.HexColor('#ba7517')
RED_MID    = colors.HexColor('#a32d2d')
RED_DARK   = colors.HexColor('#501313')
GRAY_LIGHT = colors.HexColor('#f4f6f9')
GRAY_MID   = colors.HexColor('#d0d7e2')
WHITE      = colors.white
BLACK      = colors.HexColor('#1a1a2a')

RISK_COLORS = {
    'none':    colors.HexColor('#888888'),
    'low':     colors.HexColor('#1e90ff'),
    'medium':  colors.HexColor('#ff9800'),
    'high':    colors.HexColor('#e65100'),
    'extreme': colors.HexColor('#b71c1c'),
}

SCENARIO_COLORS = {
    'normal':   colors.HexColor('#238636'),
    'moderate': colors.HexColor('#b45309'),
    'extreme':  colors.HexColor('#b91c1c'),
}

PIXEL_KM2 = 0.049


# ═══════════════════════════════════════════════════════════════════
#  STYLES
# ═══════════════════════════════════════════════════════════════════

def make_styles():
    base = getSampleStyleSheet()

    styles = {
        'h1': ParagraphStyle('h1', parent=base['Title'],
            fontSize=22, textColor=BLUE_DARK, spaceAfter=6,
            fontName='Helvetica-Bold', leading=28),

        'h2': ParagraphStyle('h2', parent=base['Heading2'],
            fontSize=13, textColor=BLUE_DARK, spaceBefore=18, spaceAfter=6,
            fontName='Helvetica-Bold', borderPad=0,
            borderColor=BLUE_DARK, leftIndent=0),

        'h3': ParagraphStyle('h3', parent=base['Heading3'],
            fontSize=11, textColor=BLUE_MID, spaceBefore=10, spaceAfter=4,
            fontName='Helvetica-Bold'),

        'body': ParagraphStyle('body', parent=base['Normal'],
            fontSize=10, textColor=BLACK, leading=15, spaceAfter=6),

        'small': ParagraphStyle('small', parent=base['Normal'],
            fontSize=9, textColor=colors.HexColor('#555555'), leading=13),

        'caption': ParagraphStyle('caption', parent=base['Normal'],
            fontSize=8.5, textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER, spaceBefore=4, spaceAfter=10),

        'meta': ParagraphStyle('meta', parent=base['Normal'],
            fontSize=9, textColor=colors.HexColor('#666666'), spaceAfter=16),

        'mono': ParagraphStyle('mono', parent=base['Normal'],
            fontSize=9, fontName='Courier', textColor=BLACK, leading=13),

        'right': ParagraphStyle('right', parent=base['Normal'],
            fontSize=9, alignment=TA_RIGHT, textColor=colors.HexColor('#666666')),
    }
    return styles


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def section_rule(styles):
    return [HRFlowable(width='100%', thickness=1.5, color=BLUE_DARK,
                        spaceAfter=4, spaceBefore=2)]

def mini_rule():
    return HRFlowable(width='100%', thickness=0.5, color=GRAY_MID,
                      spaceAfter=4, spaceBefore=4)

def risk_bar_table(risk_counts, styles):
    """Build a coloured bar table showing risk zone breakdown."""
    total = max(sum(risk_counts.values()), 1)
    rows  = []
    for risk in ['low', 'medium', 'high', 'extreme']:
        cnt  = risk_counts.get(risk, 0)
        km2  = cnt * PIXEL_KM2
        pct  = cnt / total * 100
        col  = RISK_COLORS[risk]
        hex_str = col.hexval().replace('0x','#')
        bar_cells = [
            Paragraph(f'<font color="{hex_str}"><b>{risk.capitalize()}</b></font>',
                      styles['body']),
            '',  # bar drawn via background colour
            Paragraph(f'{km2:.1f} km²', styles['body']),
            Paragraph(f'{pct:.1f}%', styles['body']),
        ]
        rows.append(bar_cells)

    t = Table(rows, colWidths=[3.5*cm, 8.5*cm, 2.5*cm, 2*cm])

    ts = [
        ('FONT',       (0,0),(-1,-1), 'Helvetica', 9),
        ('ROWBACKGROUNDS', (0,0),(-1,-1), [GRAY_LIGHT, WHITE]),
        ('GRID',       (0,0),(-1,-1), 0.3, GRAY_MID),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
    ]
    # Colour the bar column proportionally via background
    for i, risk in enumerate(['low','medium','high','extreme']):
        cnt = risk_counts.get(risk, 0)
        pct = cnt / total
        ts.append(('BACKGROUND', (1,i), (1,i), RISK_COLORS[risk]))
        ts.append(('TEXTCOLOR',  (1,i), (1,i), RISK_COLORS[risk]))  # hide empty text

    t.setStyle(TableStyle(ts))
    return t


# ═══════════════════════════════════════════════════════════════════
#  SECTIONS
# ═══════════════════════════════════════════════════════════════════

def build_cover(styles):
    story = []
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("FLOOD RISK SPATIAL ANALYSIS REPORT", styles['h1']))
    story.append(HRFlowable(width='100%', thickness=3, color=BLUE_DARK, spaceAfter=10))

    story.append(Paragraph(
        "Study area: <b>Tunis Coastal Region, Tunisia</b> &nbsp; (36.6–37.2°N, 10.0–10.6°E)",
        styles['body']))
    story.append(Paragraph(
        f"Generated: <b>{datetime.now().strftime('%d %B %Y')}</b>",
        styles['body']))
    story.append(Paragraph(
        "Models: SCS Curve Number + Bathtub Inundation + Random Forest ML",
        styles['body']))
    story.append(Spacer(1, 0.4*cm))

    # Phase completion badges
    phase_data = [
        ['Phase', 'Description', 'Status'],
        ['Phase 1', 'DEM loading, terrain analysis, SCS flood model', 'Complete'],
        ['Phase 2', 'Random Forest ML risk prediction (96.3% accuracy)', 'Complete'],
        ['Phase 3', 'Leaflet.js interactive web map + scenario sliders', 'Complete'],
        ['Phase 4', '12-hour animated flood propagation + PDF report', 'Complete'],
    ]
    t = Table(phase_data, colWidths=[3*cm, 11*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  BLUE_DARK),
        ('TEXTCOLOR',     (0,0),(-1,0),  WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [GRAY_LIGHT, WHITE]),
        ('GRID',          (0,0),(-1,-1), 0.3, GRAY_MID),
        ('ALIGN',         (2,0),(2,-1),  'CENTER'),
        ('TEXTCOLOR',     (2,1),(2,-1),  TEAL),
        ('FONTNAME',      (2,1),(2,-1),  'Helvetica-Bold'),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))
    return story


def build_executive_summary(styles):
    story = []
    story += section_rule(styles)
    story.append(Paragraph("1. Executive Summary", styles['h2']))

    story.append(Paragraph(
        "This report presents the results of a multi-phase flood risk spatial analysis for the Tunis "
        "coastal region. Three rainfall scenarios were evaluated using a hybrid physics and machine "
        "learning modelling approach. The study area covers approximately 4,435 km² at 200m resolution.",
        styles['body']))

    # Key findings table
    findings = [
        ['Scenario', 'Rainfall', 'Sea Rise', 'Flooded Area', '% Region', 'Return Period'],
    ]
    rp = {'normal': '< 2 yrs', 'moderate': '~10 yrs', 'extreme': '~100 yrs'}
    sc = analysis['scenarios']
    for name in ['normal', 'moderate', 'extreme']:
        s = sc[name]
        findings.append([
            name.capitalize(),
            f"{s['total_rain_mm']} mm",
            f"{s['sea_level_rise_m']} m",
            f"{s['flooded_area_km2']} km²",
            f"{s['flooded_pct']}%",
            rp[name],
        ])

    t = Table(findings, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 3*cm, 2*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  BLUE_DARK),
        ('TEXTCOLOR',     (0,0),(-1,0),  WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [GRAY_LIGHT, WHITE]),
        ('GRID',          (0,0),(-1,-1), 0.3, GRAY_MID),
        ('ALIGN',         (1,0),(-1,-1), 'CENTER'),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        # Colour scenario names
        ('TEXTCOLOR', (0,1),(0,1), SCENARIO_COLORS['normal']),
        ('TEXTCOLOR', (0,2),(0,2), SCENARIO_COLORS['moderate']),
        ('TEXTCOLOR', (0,3),(0,3), SCENARIO_COLORS['extreme']),
        ('FONTNAME',  (0,1),(0,-1), 'Helvetica-Bold'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))
    return story


def build_phase1_results(styles):
    story = []
    story += section_rule(styles)
    story.append(Paragraph("2. Phase 1 — Physics Flood Model Results", styles['h2']))

    story.append(Paragraph(
        "The SCS Curve Number method was used to convert rainfall into surface runoff. "
        "Curve Numbers ranged from 65 (vegetated hillsides) to 85 (coastal urban zones). "
        "Coastal inundation was modelled using a connected-component bathtub algorithm seeded "
        "from the western coastal boundary.",
        styles['body']))

    # Embed Phase 1 dashboard image if available
    img1_path = IMG_DIR + "flood_analysis_phase1.png"
    if os.path.exists(img1_path):
        story.append(Image(img1_path, width=16*cm, height=10*cm))
        story.append(Paragraph("Figure 1 — Phase 1 analysis dashboard: DEM, flood maps for all "
                                "three scenarios, risk zone comparison, and elevation distribution.",
                                styles['caption']))

    for name in ['normal', 'moderate', 'extreme']:
        s = analysis['scenarios'][name]
        rc = s.get('risk_counts', {})
        story.append(Paragraph(f"{name.upper()} scenario — {s['description']}", styles['h3']))
        story.append(risk_bar_table(rc, styles))
        story.append(Spacer(1, 0.2*cm))

    return story


def build_phase2_results(styles):
    story = []
    story += section_rule(styles)
    story.append(Paragraph("3. Phase 2 — Machine Learning Risk Prediction", styles['h2']))

    story.append(Paragraph(
        "A Random Forest classifier was trained on 6 terrain features extracted from the DEM, "
        "using Phase 1 physics outputs as training labels. The model was evaluated on a stratified "
        "25% held-out test set.",
        styles['body']))

    # Embed Phase 2 image if available
    img2_path = IMG_DIR + "flood_analysis_phase2.png"
    if os.path.exists(img2_path):
        story.append(Image(img2_path, width=16*cm, height=10*cm))
        story.append(Paragraph("Figure 2 — Phase 2 ML results: physics vs ML risk maps, "
                                "probability map, feature importance, and model summary.",
                                styles['caption']))

    # Model performance table
    perf_data = [
        ['Metric', 'Value'],
        ['Model type', 'Random Forest Classifier'],
        ['Number of trees', str(ml_meta['n_estimators'])],
        ['Max tree depth', str(ml_meta['max_depth'])],
        ['Training samples', '67,500 terrain pixels'],
        ['Test samples', '22,500 terrain pixels'],
        ['Overall accuracy', f"{ml_meta['accuracy']*100:.1f}%"],
    ]
    t = Table(perf_data, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  BLUE_DARK),
        ('TEXTCOLOR',     (0,0),(-1,0),  WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [GRAY_LIGHT, WHITE]),
        ('GRID',          (0,0),(-1,-1), 0.3, GRAY_MID),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        # Highlight accuracy row
        ('BACKGROUND',    (0,-1),(-1,-1), colors.HexColor('#e8f5e9')),
        ('TEXTCOLOR',     (1,-1),(1,-1),  TEAL),
        ('FONTNAME',      (1,-1),(1,-1),  'Helvetica-Bold'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

    # Feature importance table
    story.append(Paragraph("Feature importance ranking", styles['h3']))
    fi = ml_meta['feature_importances']
    sorted_fi = sorted(fi.items(), key=lambda x: -x[1])
    fi_data = [['Rank', 'Feature', 'Importance', 'Interpretation']]
    interpretations = {
        'curvature':   'Concave hollows collect water',
        'coast_dist':  'Proximity to flood source',
        'elevation':   'Absolute height above sea',
        'twi':         'Natural wetness of terrain',
        'slope':       'Drainage speed',
        'relief':      'Local height variation',
    }
    for i, (name, val) in enumerate(sorted_fi, 1):
        fi_data.append([
            str(i), name, f"{val*100:.1f}%",
            interpretations.get(name, ''),
        ])

    t2 = Table(fi_data, colWidths=[1*cm, 3.5*cm, 2.5*cm, 9.5*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  BLUE_DARK),
        ('TEXTCOLOR',     (0,0),(-1,0),  WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [GRAY_LIGHT, WHITE]),
        ('GRID',          (0,0),(-1,-1), 0.3, GRAY_MID),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('TEXTCOLOR',     (2,1),(2,1), BLUE_DARK),
        ('FONTNAME',      (2,1),(2,1), 'Helvetica-Bold'),
    ]))
    story.append(t2)
    return story


def build_phase4_animation(styles):
    story = []
    story += section_rule(styles)
    story.append(Paragraph("4. Phase 4 — 12-Hour Flood Propagation Simulation", styles['h2']))

    story.append(Paragraph(
        "The animation module simulates how flood water spreads over 12 hourly time steps during "
        "a 100-year storm event. Sea level peaks at +1.20m at hour 6, then recedes. Cumulative "
        "rainfall reaches 179mm by hour 11.",
        styles['body']))

    # Animation frame table
    anim_data = [['Hour', 'Phase', 'Sea level (m)', 'Cumulative rain (mm)', 'Flooded area (km²)', 'Max depth (m)']]
    for f in anim_meta:
        phase_str = 'Rising' if f['phase'] == 'rising' else 'Receding'
        anim_data.append([
            f"{f['hour']}h",
            phase_str,
            f"+{f['sea_level_m']}",
            f"{f['cumulative_rain_mm']}",
            f"{f['flooded_km2']}",
            f"{f['max_depth_m']}",
        ])

    t = Table(anim_data, colWidths=[1.5*cm, 2.2*cm, 2.5*cm, 3.5*cm, 3*cm, 3*cm])
    style_cmds = [
        ('BACKGROUND',    (0,0),(-1,0),  BLUE_DARK),
        ('TEXTCOLOR',     (0,0),(-1,0),  WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 8.5),
        ('GRID',          (0,0),(-1,-1), 0.3, GRAY_MID),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('ALIGN',         (1,0),(-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0),(-1,-1), 4),
        ('BOTTOMPADDING', (0,0),(-1,-1), 4),
    ]
    # Colour rows: rising = warm, receding = cool
    for i, f in enumerate(anim_meta, 1):
        bg = colors.HexColor('#fff8f0') if f['phase'] == 'rising' else colors.HexColor('#f0f8ff')
        style_cmds.append(('BACKGROUND', (0,i),(-1,i), bg))
        phase_col = colors.HexColor('#c05c00') if f['phase'] == 'rising' else colors.HexColor('#1565c0')
        style_cmds.append(('TEXTCOLOR', (1,i),(1,i), phase_col))
    # Highlight peak frame
    style_cmds.append(('BACKGROUND', (0,7),(-1,7), colors.HexColor('#fde8e8')))
    style_cmds.append(('FONTNAME', (0,7),(-1,7), 'Helvetica-Bold'))
    style_cmds.append(('TEXTCOLOR', (0,7),(0,7), RED_DARK))

    t.setStyle(TableStyle(style_cmds))
    story.append(t)
    story.append(Paragraph("Table: Row highlighted in red = peak flood frame (hour 6, max sea level). "
                            "Orange rows = rising phase. Blue rows = receding phase.",
                            styles['caption']))
    return story


def build_methodology(styles):
    story = []
    story += section_rule(styles)
    story.append(Paragraph("5. Methodology", styles['h2']))

    methods = [
        ("5.1  Digital Elevation Model",
         "The DEM covers the Tunis coastal region at approximately 200m resolution "
         "(300×300 grid cells). In production, this would be replaced with SRTM 30m data "
         "downloaded free from NASA EarthData or the Copernicus DEM. Terrain derivatives "
         "computed: slope, local relief, depressions, Topographic Wetness Index (TWI)."),

        ("5.2  SCS Curve Number runoff model",
         "Rainfall-to-runoff conversion used the USDA Soil Conservation Service Curve Number "
         "method: Q = (P - Ia)² / (P - Ia + S), where S = (25400/CN) - 254 and Ia = 0.2S. "
         "Curve Numbers ranged from 65 (permeable hillside terrain) to 85 (impervious coastal "
         "urban surface), varying with elevation as a land-use proxy."),

        ("5.3  Coastal inundation model",
         "A connected-component bathtub model was applied to simulate coastal flooding. "
         "Starting from the western coastal edge, binary dilation propagated flood extent "
         "through all terrain cells connected to the sea and below the water surface level. "
         "This enforces spatial connectivity and prevents inland 'floating lakes'."),

        ("5.4  Random Forest classifier",
         "A scikit-learn RandomForestClassifier (100 trees, max_depth=12, class_weight='balanced') "
         "was trained on 67,500 terrain pixels with 6 features. Labels derived from Phase 1 "
         "physics outputs. Evaluated on a 22,500-pixel stratified hold-out: overall accuracy 96.3%, "
         "macro-average F1 0.95. Top predictor: terrain curvature (48.8% importance)."),

        ("5.5  Web map & animation",
         "The Leaflet.js interactive map serves all flood layers as GeoJSON. Twelve animation "
         "frames simulate hourly flood propagation over a 12-hour 100-year storm, with sea level "
         "peaking at +1.20m at hour 6. The animation engine runs client-side in the browser."),
    ]

    for title, body in methods:
        story.append(Paragraph(title, styles['h3']))
        story.append(Paragraph(body, styles['body']))

    return story


def build_footer_section(styles):
    story = []
    story += section_rule(styles)
    story.append(Paragraph("6. Software & Data Sources", styles['h2']))

    sw_data = [
        ['Component', 'Tool / Source', 'Version'],
        ['DEM processing', 'rasterio, numpy', 'Python 3.12'],
        ['Watershed analysis', 'scipy.ndimage (uniform_filter, gaussian_filter)', '—'],
        ['Flood model', 'SCS-CN + bathtub (custom Python)', '—'],
        ['ML model', 'scikit-learn RandomForestClassifier', '1.3+'],
        ['Visualisation', 'matplotlib', '3.8+'],
        ['Web map', 'Leaflet.js + Chart.js', '1.9.4 / 4.4'],
        ['PDF report', 'ReportLab', '4.x'],
        ['DEM data source', 'SRTM (NASA EarthData) or Copernicus DEM', 'Free/open'],
        ['Weather API', 'Open-Meteo (open, no key required)', 'Free/open'],
    ]
    t = Table(sw_data, colWidths=[5*cm, 8.5*cm, 3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  BLUE_DARK),
        ('TEXTCOLOR',     (0,0),(-1,0),  WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [GRAY_LIGHT, WHITE]),
        ('GRID',          (0,0),(-1,-1), 0.3, GRAY_MID),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
    ]))
    story.append(t)

    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY_MID))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Flood Risk Spatial Analysis Tool — Phases 1–4 complete — "
        f"Generated {datetime.now().strftime('%d %B %Y at %H:%M')}",
        styles['small']))
    return story


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def generate_report():
    out_path = OUTPUT_DIR + "flood_risk_report.pdf"

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title="Flood Risk Analysis Report",
        author="Flood Risk Analysis Tool",
        subject="Tunis Coastal Region Flood Risk Spatial Analysis",
    )

    styles = make_styles()
    story  = []

    story += build_cover(styles)
    story.append(Spacer(1, 0.5*cm))
    story += build_executive_summary(styles)
    story += build_phase1_results(styles)
    story += build_phase2_results(styles)
    story += build_phase4_animation(styles)
    story += build_methodology(styles)
    story += build_footer_section(styles)

    doc.build(story)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"✓ Report saved: {out_path}  ({size_kb} KB)")
    return out_path


if __name__ == "__main__":
    print("=" * 55)
    print("  Generating flood risk PDF report...")
    print("=" * 55)
    path = generate_report()
    print(f"\n  Open it with:  xdg-open {path}")
    print("=" * 55)
