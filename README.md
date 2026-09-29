# geospatial-project-flood-risk

Projet **géospatial d'analyse du risque d'inondation**. Chaîne de traitement raster en Python
(rasterio / geopandas / scikit-learn), génération d'un rapport PDF, et une webapp
statique de cartographie interactive des cellules de risque.

## Phases du projet

| Fichier | Étape |
| --- | --- |
| `phase1_analysis.py` | Analyse exploratoire et prétraitement des rasters |
| `phase2_ml.py` | Apprentissage automatique et prédiction du risque |
| `phase4_report.py` | Génération du rapport PDF de synthèse |

## Résultats

- `outputs/flood_risk_report.pdf` — rapport complet
- `outputs/flood_analysis_phase1.png` — visuels d'analyse
- `outputs/flood_analysis_phase2.png` — résultats du modèle
- `data/analysis_results.json`, `data/ml_model_meta.json`, `data/rainfall_scenarios.json`

## Webapp

Le dossier `webapp/` est une application statique (aucun build requis) :

| Fichier | Rôle |
| --- | --- |
| `index.html` | Page principale |
| `grid_data.js` | Grille de risque |
| `ml_grid.js` | Grille issue du modèle ML |
| `animation_data.js` | Animation des scénarios de pluie |
| `data.js`, `model_data.js` | Données brutes et métadonnées modèle |

```bash
cd webapp && python -m http.server 8000
# puis ouvrir http://localhost:8000
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install rasterio geopandas numpy matplotlib scipy scikit-learn reportlab requests
```

## Auteur

Anwar Ben Brahim — [GitHub](https://github.com/anouar-coder)
