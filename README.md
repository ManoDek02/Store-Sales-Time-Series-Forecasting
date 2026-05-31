# 🛒 Projet 1 — Feature Engineering & Prévision des Ventes Retail

> **De −30.8% à −42%+ d'erreur. Même algorithme. Meilleures features. Meilleur ensemble.**

Analyse complète du dataset **Store Sales — Corporación Favorita** (Kaggle) :
feature engineering avancé, optimisation Optuna, ensemble XGBoost + LightGBM + CatBoost,
interprétabilité SHAP et dashboard décisionnel interactif.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Résultats

| Modèle | Features | MAE | vs Baseline |
|--------|----------|-----|-------------|
| Baseline | 7 features basiques | 449 | — |
| Feature engineering | 37 features | 311 | **−30.8%** |
| + Features avancées | 53 features | ~280 | **−37%+** |
| + Optuna XGBoost | 53 features | ~265 | **−41%+** |
| **Ensemble Stacking** | **53 features** | **~260** | **🏆 −42%+** |

> Dataset : GROCERY I · Magasin 1 · Quito · Split temporel 80/20

---

## Demo — Dashboard interactif

```bash
streamlit run app.py
```

Le dashboard propose deux modules :

| Module | Fonctionnalités |
|--------|----------------|
| 📊 **Dashboard** | KPIs temps réel · Prédictions vs Réel · Impact marketing · Ventes hebdomadaires |
| 🔮 **Simulateur** | What-if : promo, jours fériés, trafic · Décomposition d'impact · Recommandations auto |
| 🔍 **Interprétabilité** | Feature importance · Résidus · Insights SHAP |

---

## Structure du projet

```
projet-01-store-sales-features/
│
├── 📂 data/                            # Données (non committées — voir Installation)
│   └── .gitkeep
│
├── 📂 notebooks/
│   ├── 01_exploration.ipynb            # EDA — insights marketing chiffrés
│   ├── 02_feature_engineering.ipynb   # 37 features en 5 catégories
│   ├── 03_modeling_xgboost.ipynb      # XGBoost + validation croisée temporelle
│   ├── 04_shap_interpretation.ipynb   # Interprétabilité SHAP complète
│   ├── 05_model_improvement.ipynb     # Optuna + LightGBM + CatBoost + Ensemble
│   └── 06_multi_family.ipynb          # Pipeline 10 familles de produits
│
├── 📂 src/
│   ├── __init__.py
│   ├── data_loader.py                 # Chargement et fusion des 5 fichiers
│   └── feature_engineering.py        # Pipeline des 5 catégories de features
│
├── 📂 models/                          # Modèles sauvegardés (.pkl)
│   ├── xgboost_grocery1_store1.pkl
│   ├── ensemble_final.pkl
│   └── multi_family_results.pkl
│
├── 🐍 app.py                           # Dashboard Streamlit
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md
```

---

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/ton-username/projet-01-store-sales-features.git
cd projet-01-store-sales-features

# 2. Environnement virtuel
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# 3. Dépendances
pip install -r requirements.txt
```

---

## Données

Dataset : [Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) (Kaggle)

```bash
# Accepter les règles sur Kaggle, puis :
kaggle competitions download -c store-sales-time-series-forecasting -p data/
cd data && unzip store-sales-time-series-forecasting.zip && cd ..
```

| Fichier | Lignes | Description |
|---------|--------|-------------|
| `train.csv` | 3 000 888 | Ventes quotidiennes par magasin et famille |
| `stores.csv` | 54 | Type, cluster, ville et état |
| `holidays_events.csv` | 350 | Jours fériés équatoriens |
| `oil.csv` | 1 218 | Prix du pétrole quotidien |
| `transactions.csv` | 83 488 | Trafic client par magasin |

---

## Utilisation rapide

```python
from src.data_loader import load_data
from src.feature_engineering import build_features

# Charger les données
df = load_data('./data/')

# Features pour n'importe quel magasin / famille
d_clean = build_features(df, store_nbr=1, family='GROCERY I')
print(d_clean.shape)  # (1686, 38)

# Charger l'ensemble entraîné
import joblib
models = joblib.load('models/ensemble_final.pkl')
```

---

## Les 5 catégories de features

| Catégorie | Nb | Exemples clés |
|-----------|-----|--------------|
| 🗓️ Calendaires enrichies | 9 | `is_month_start`, `days_to_holiday`, `quarter` |
| ⏪ Lag | 4 | `lag_1d`, `lag_7d`, `lag_14d`, `lag_28d` |
| 📊 Rolling windows | 12 | `rolling_mean_7d`, `rolling_min_7d`, `rolling_std_28d` |
| 🏷️ Promotionnelles | 3 | `lag_promo_7d`, `rolling_promo_14d`, `post_promo` |
| ⚡ Interaction | 4 | `promo_x_holiday`, `promo_x_weekend`, `oil_x_lag1` |
| 🌊 Avancées (Fourier) | 8 | `fourier_sin_1`, `dow_sin`, `trend`, `spike_flag` |

---

## Insights clés

### Feature engineering
- **`rolling_min_7d`** est la feature la plus prédictive — le creux des 7 derniers jours bat le simple `lag_1d`
- **Le trafic client (`transactions`)** prédit mieux les ventes que les promotions
- **L'encodage cyclique** (sin/cos) améliore la capture des saisonnalités vs encodage naïf

### Impact marketing (Magasin 1 · Quito)
| Levier | Impact |
|--------|--------|
| 🎉 Jours fériés | **+12.7%** |
| 📅 Début de mois | **+14.2%** |
| 🏷️ Promo GROCERY I | **+28.2%** |
| 🥤 Promo BEVERAGES | **+86.4%** |
| 🥦 Promo PRODUCE | **+179.2%** |

### Résultats multi-familles (Notebook 06)
Les 10 familles principales couvrent des profils très différents :
- **Familles très prévisibles** (MAE relative < 20%) : GROCERY I, DAIRY, CLEANING
- **Familles moyennement prévisibles** (20–40%) : BEVERAGES, PRODUCE
- **Familles difficiles** (> 40%) : saisonnalité forte ou demande très volatile

---

## Parcours des notebooks

```
01_exploration      →   Comprendre les données et identifier les leviers
        ↓
02_feature_eng      →   Construire les 37 features de base
        ↓
03_modeling         →   Entraîner XGBoost + validation croisée temporelle
        ↓
04_shap             →   Interpréter les prédictions (feature importance, waterfall)
        ↓
05_improvement      →   Optuna + LightGBM + CatBoost + Ensemble stacking
        ↓
06_multi_family     →   Généraliser à 10 familles de produits
        ↓
app.py              →   Dashboard interactif pour les décideurs
```

---

## Optimisation (Notebook 05)

L'optimisation Optuna a testé **120 combinaisons d'hyperparamètres** sur 3 algorithmes :

```python
# Paramètres optimisés automatiquement
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

L'ensemble final combine XGBoost + LightGBM avec des poids inversement proportionnels
à leur MAE individuelle — le meilleur des deux mondes.

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.11 |
| Modèles | XGBoost · LightGBM · CatBoost |
| Optimisation | Optuna (120 essais) |
| Interprétabilité | SHAP |
| Dashboard | Streamlit |
| Data | pandas · numpy |
| Visualisation | matplotlib · seaborn |
| Persistance | joblib |

---

## Recommandations business

**Pour l'équipe marketing**
- Concentrer les promos sur **PRODUCE (+179%)** et **BEVERAGES (+86%)** plutôt que GROCERY I (+28%)
- Planifier les campagnes en **début de mois** pour capitaliser sur l'effet salaire

**Pour la supply chain**
- Augmenter les commandes **les 25-28 du mois** pour anticiper le pic de début de mois
- Monitorer le **trafic client en temps réel** — meilleur prédicteur que les promos

**Pour l'équipe data**
- Déployer le pipeline sur les **54 magasins** pour généraliser
- Mettre en place un **monitoring de drift** mensuel

---

## Publication LinkedIn

Ce projet fait partie d'une série de **20 projets data science** publiés sur LinkedIn.


---

## Licence

MIT — libre d'utilisation avec attribution.