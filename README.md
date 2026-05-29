# Projet 1 — Feature Engineering sur données de ventes retail

> **−30.8% d'erreur de prédiction. Même algorithme. Meilleures features.**

Analyse complète du dataset Store Sales (Corporación Favorita) avec construction
de 37 features temporelles avancées et benchmark XGBoost baseline vs complet.

---

## Résultats

| Modèle | Features | MAE | Réduction |
|--------|----------|-----|-----------|
| Baseline | 7 features basiques | 449 | — |
| **Complet** | **37 features avancées** | **311** | **−30.8%** |

> Dataset : GROCERY I · Magasin 1 · Quito · Split temporel 80/20

---

## Insights clés

- **rolling_min_7d** est la feature la plus prédictive — le creux des 7 derniers jours
  bat le simple lag_1d
- **Le trafic client (transactions)** prédit mieux les ventes que les promotions
- **L'impact promo varie de 1 à 6x** selon la famille de produits :
  - GROCERY I : +28%
  - BEVERAGES : +86%
  - PRODUCE   : +179%
- **Début de mois** : +14.2% de ventes (effet salaire)
- **Jours fériés** : +12.7% de ventes

---

## Structure du projet

projet-01-store-sales-features/
│
├── data/                        # Données (non committées)
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_exploration.ipynb     # EDA — insights marketing
│   ├── 02_feature_engineering.ipynb  # Construction des 37 features
│   ├── 03_modeling_xgboost.ipynb     # Benchmark baseline vs complet
│   └── 04_shap_interpretation.ipynb  # Interprétabilité SHAP
│
├── src/
│   ├── init.py
│   ├── data_loader.py           # Chargement et fusion des 5 fichiers
│   └── feature_engineering.py  # Pipeline des 5 catégories de features
│
├── models/                      # Modèle sauvegardé (.pkl)
├── requirements.txt
├── .gitignore
└── README.md

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/ton-username/projet-01-store-sales-features.git
cd projet-01-store-sales-features

# Créer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# Installer les dépendances
pip install -r requirements.txt
```

---

## Données

Dataset issu de la compétition Kaggle :
[Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

```bash
# Télécharger après avoir accepté les règles sur Kaggle
kaggle competitions download -c store-sales-time-series-forecasting -p data/
cd data && unzip store-sales-time-series-forecasting.zip && cd ..
```

**Fichiers utilisés :**

| Fichier | Lignes | Description |
|---------|--------|-------------|
| train.csv | 3 000 888 | Ventes quotidiennes par magasin et famille |
| stores.csv | 54 | Métadonnées magasins |
| holidays_events.csv | 350 | Jours fériés équatoriens |
| oil.csv | 1 218 | Prix du pétrole quotidien |
| transactions.csv | 83 488 | Trafic client par magasin |

---

## Utilisation rapide

```python
from src.data_loader import load_data
from src.feature_engineering import build_features

# Charger les données
df = load_data('./data/')

# Construire les features pour n'importe quel magasin / famille
d_clean = build_features(df, store_nbr=1, family='GROCERY I')

print(d_clean.shape)  # (1686, 38)
```

---

## Les 5 catégories de features

| Catégorie | Nb | Exemples |
|-----------|-----|---------|
| Calendaires enrichies | 9 | day_of_week, is_month_start, days_to_holiday |
| Lag | 4 | lag_1d, lag_7d, lag_14d, lag_28d |
| Rolling windows | 12 | rolling_mean_7d, rolling_std_28d, rolling_max_14d |
| Promotionnelles | 3 | lag_promo_7d, rolling_promo_14d, post_promo |
| Interaction | 4 | promo_x_holiday, promo_x_weekend, oil_x_lag1 |

---

## Stack technique

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![pandas](https://img.shields.io/badge/pandas-2.x-purple)
![SHAP](https://img.shields.io/badge/SHAP-interpretability-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

- **pandas** — manipulation et fusion des données
- **XGBoost** — modèle de prédiction gradient boosting
- **SHAP** — interprétabilité des prédictions
- **matplotlib / seaborn** — visualisations
- **tsfresh** — extraction automatique de features temporelles

---

## Publication LinkedIn

Ce projet fait partie d'une série de 20 projets data science publiés sur LinkedIn.
