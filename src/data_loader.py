# src/data_loader.py

import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
    """
    Charge et fusionne les 5 fichiers du dataset Store Sales.

    Args:
        data_path: chemin vers le dossier data/ (ex: '../data/')

    Returns:
        DataFrame maître fusionné et nettoyé
    """
    p = data_path.rstrip('/')

    # Chargement
    train    = pd.read_csv(f'{p}/train.csv',            parse_dates=['date'])
    stores   = pd.read_csv(f'{p}/stores.csv')
    holidays = pd.read_csv(f'{p}/holidays_events.csv',  parse_dates=['date'])
    oil      = pd.read_csv(f'{p}/oil.csv',              parse_dates=['date'])
    transac  = pd.read_csv(f'{p}/transactions.csv',     parse_dates=['date'])

    # Fusion
    df = train.merge(stores,  on='store_nbr',            how='left')
    df = df.merge(transac,    on=['date', 'store_nbr'],  how='left')
    df = df.merge(oil,        on='date',                 how='left')

    # Jours fériés — exclure les jours transférés
    hol = (holidays[holidays['transferred'] == False]
           [['date', 'type', 'locale']]
           .rename(columns={'type': 'holiday_type',
                            'locale': 'holiday_locale'}))
    df = df.merge(hol, on='date', how='left')

    # Nettoyage
    df['is_holiday']    = df['holiday_type'].notna().astype('int8')
    df['has_promotion'] = (df['onpromotion'] > 0).astype(int)
    df['dcoilwtico']    = df['dcoilwtico'].ffill().bfill()
    df['transactions']  = df['transactions'].fillna(0).astype('int32')

    return df


def filter_series(df: pd.DataFrame,
                  store_nbr: int,
                  family: str) -> pd.DataFrame:
    """
    Isole une série temporelle pour un magasin et une famille donnés.

    Args:
        df:        DataFrame maître issu de load_data()
        store_nbr: numéro du magasin (1 à 54)
        family:    famille de produits (ex: 'GROCERY I')

    Returns:
        DataFrame trié par date avec index datetime
    """
    mask = (df['store_nbr'] == store_nbr) & (df['family'] == family)
    cols = ['sales', 'onpromotion', 'dcoilwtico',
            'is_holiday', 'has_promotion', 'transactions']
    return (df[mask][cols]
            .sort_values('date')
            .set_index('date')
            .copy())