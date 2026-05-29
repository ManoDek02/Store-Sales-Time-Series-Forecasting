# src/feature_engineering.py

import pandas as pd
import numpy as np


def add_calendar_features(d: pd.DataFrame) -> pd.DataFrame:
    """Catégorie 1 — Features calendaires enrichies."""
    d['day_of_week']     = d.index.dayofweek
    d['day_of_month']    = d.index.day
    d['month']           = d.index.month
    d['week_of_year']    = d.index.isocalendar().week.astype(int)
    d['quarter']         = d.index.quarter
    d['is_weekend']      = (d.index.dayofweek >= 5).astype('int8')
    d['is_month_start']  = (d.index.day <= 5).astype('int8')
    d['is_month_end']    = (d.index.day >= 25).astype('int8')
    d['days_to_holiday'] = (d['is_holiday']
                            .shift(-1).fillna(0)
                            .rolling(7, min_periods=1).sum())
    return d


def add_lag_features(d: pd.DataFrame,
                     lags: list = [1, 7, 14, 28]) -> pd.DataFrame:
    """Catégorie 2 — Features de lag."""
    for lag in lags:
        d[f'lag_{lag}d'] = d['sales'].shift(lag)
    return d


def add_rolling_features(d: pd.DataFrame,
                         windows: list = [7, 14, 28]) -> pd.DataFrame:
    """Catégorie 3 — Rolling windows (fenêtres glissantes)."""
    for w in windows:
        base = d['sales'].shift(1).rolling(w)
        d[f'rolling_mean_{w}d'] = base.mean()
        d[f'rolling_std_{w}d']  = base.std()
        d[f'rolling_max_{w}d']  = base.max()
        d[f'rolling_min_{w}d']  = base.min()
    return d


def add_promo_features(d: pd.DataFrame) -> pd.DataFrame:
    """Catégorie 4 — Features promotionnelles."""
    d['lag_promo_7d']      = d['has_promotion'].shift(1).rolling(7).sum()
    d['rolling_promo_14d'] = d['has_promotion'].shift(1).rolling(14).mean()
    d['post_promo']        = d['has_promotion'].shift(1)
    return d


def add_interaction_features(d: pd.DataFrame) -> pd.DataFrame:
    """Catégorie 5 — Features d'interaction."""
    d['promo_x_holiday']   = d['has_promotion'] * d['is_holiday']
    d['promo_x_weekend']   = d['has_promotion'] * d['is_weekend']
    d['promo_x_month_end'] = d['has_promotion'] * d['is_month_end']
    d['oil_x_lag1']        = d['dcoilwtico'] * d['lag_1d'].fillna(0)
    return d


def build_features(df: pd.DataFrame,
                   store_nbr: int,
                   family: str) -> pd.DataFrame:
    """
    Pipeline complet : isolation + 5 catégories de features + nettoyage.

    Args:
        df:        DataFrame maître issu de load_data()
        store_nbr: numéro du magasin
        family:    famille de produits

    Returns:
        DataFrame prêt pour la modélisation (sans NaN)
    """
    from src.data_loader import filter_series

    d = filter_series(df, store_nbr, family)
    d = add_calendar_features(d)
    d = add_lag_features(d)
    d = add_rolling_features(d)
    d = add_promo_features(d)
    d = add_interaction_features(d)

    n_before = len(d)
    d = d.dropna()
    print(f"[build_features] {store_nbr} · {family}")
    print(f"  Lignes avant dropna : {n_before}")
    print(f"  Lignes après dropna : {len(d)}")
    print(f"  Features construites : {d.shape[1] - 1}")

    return d