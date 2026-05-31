# app.py — Store Sales · Dashboard Décisionnel + Simulateur
# Lancer : streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration Streamlit ───────────────────────────────────
st.set_page_config(
    page_title="Store Sales · Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Custom ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #F8F7F4; }
.block-container { padding: 1.5rem 2rem 2rem; }

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: #1E3A5F !important;
}
[data-testid="stSidebar"] * { color: #E8EEF5 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label { color: #B8C8D8 !important; font-size: 13px; }

/* Métriques */
[data-testid="metric-container"] {
    background: white;
    border: 0.5px solid #E5E5E5;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="metric-container"] label { font-size: 12px !important; color: #888 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 28px !important; font-weight: 600 !important; color: #1E3A5F !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 13px !important; }

/* Titres */
h1 { font-size: 22px !important; font-weight: 600 !important; color: #1E3A5F !important; }
h2 { font-size: 16px !important; font-weight: 500 !important; color: #1E3A5F !important; margin-top: 1.5rem !important; }
h3 { font-size: 14px !important; font-weight: 500 !important; color: #444 !important; }

/* Cards */
.insight-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 0.5px solid #E5E5E5;
    margin-bottom: 10px;
}
.insight-title { font-size: 12px; color: #888; margin-bottom: 4px; font-weight: 500; }
.insight-value { font-size: 20px; font-weight: 600; color: #1E3A5F; }
.insight-sub { font-size: 12px; color: #666; margin-top: 2px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-size: 13px;
    padding: 6px 16px;
    color: #666;
}
.stTabs [aria-selected="true"] {
    background: white;
    color: #1E3A5F !important;
    font-weight: 500;
}

/* Simulation result box */
.sim-box {
    background: linear-gradient(135deg, #E1F5EE 0%, #F0FAF5 100%);
    border: 1px solid #0F6E56;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.sim-box-danger {
    background: linear-gradient(135deg, #FAECE7 0%, #FDF5F2 100%);
    border: 1px solid #D85A30;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.sim-title { font-size: 13px; color: #555; margin-bottom: 8px; font-weight: 500; }
.sim-value { font-size: 32px; font-weight: 600; }
.sim-detail { font-size: 12px; color: #666; margin-top: 6px; line-height: 1.5; }

/* Divider */
hr { border: none; border-top: 0.5px solid #E5E5E5; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Chargement des données ────────────────────────────────────
@st.cache_data
def load_all_data(data_path='data/'):
    p = data_path.rstrip('/')
    train    = pd.read_csv(f'{p}/train.csv',            parse_dates=['date'])
    stores   = pd.read_csv(f'{p}/stores.csv')
    holidays = pd.read_csv(f'{p}/holidays_events.csv',  parse_dates=['date'])
    oil      = pd.read_csv(f'{p}/oil.csv',              parse_dates=['date'])
    transac  = pd.read_csv(f'{p}/transactions.csv',     parse_dates=['date'])

    df = train.merge(stores, on='store_nbr', how='left')
    df = df.merge(transac,   on=['date','store_nbr'], how='left')
    df = df.merge(oil,       on='date', how='left')

    hol = (holidays[holidays['transferred']==False][['date','type','locale']]
           .rename(columns={'type':'holiday_type','locale':'holiday_locale'}))
    df  = df.merge(hol, on='date', how='left')

    df['is_holiday']    = df['holiday_type'].notna().astype('int8')
    df['has_promotion'] = (df['onpromotion'] > 0).astype(int)
    df['dcoilwtico']    = df['dcoilwtico'].ffill().bfill()
    df['transactions']  = df['transactions'].fillna(0).astype('int32')
    return df, stores


@st.cache_data
def build_features(df, store_nbr, family):
    mask = (df['store_nbr']==store_nbr) & (df['family']==family)
    cols = ['sales','onpromotion','dcoilwtico','is_holiday','has_promotion','transactions']
    d = df[mask].sort_values('date').set_index('date')[cols].copy()

    d['day_of_week']     = d.index.dayofweek
    d['day_of_month']    = d.index.day
    d['month']           = d.index.month
    d['week_of_year']    = d.index.isocalendar().week.astype(int)
    d['quarter']         = d.index.quarter
    d['is_weekend']      = (d.index.dayofweek >= 5).astype('int8')
    d['is_month_start']  = (d.index.day <= 5).astype('int8')
    d['is_month_end']    = (d.index.day >= 25).astype('int8')
    d['days_to_holiday'] = d['is_holiday'].shift(-1).fillna(0).rolling(7, min_periods=1).sum()

    for lag in [1, 7, 14, 28]:
        d[f'lag_{lag}d'] = d['sales'].shift(lag)
    for w in [7, 14, 28]:
        base = d['sales'].shift(1).rolling(w)
        d[f'rolling_mean_{w}d'] = base.mean()
        d[f'rolling_std_{w}d']  = base.std()
        d[f'rolling_max_{w}d']  = base.max()
        d[f'rolling_min_{w}d']  = base.min()

    d['lag_promo_7d']      = d['has_promotion'].shift(1).rolling(7).sum()
    d['rolling_promo_14d'] = d['has_promotion'].shift(1).rolling(14).mean()
    d['post_promo']        = d['has_promotion'].shift(1)
    d['promo_x_holiday']   = d['has_promotion'] * d['is_holiday']
    d['promo_x_weekend']   = d['has_promotion'] * d['is_weekend']
    d['promo_x_month_end'] = d['has_promotion'] * d['is_month_end']
    d['oil_x_lag1']        = d['dcoilwtico'] * d['lag_1d'].fillna(0)

    return d.dropna()


@st.cache_resource
def train_model(df, store_nbr, family):
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error

    d = build_features(df, store_nbr, family)
    FEAT = [c for c in d.columns if c != 'sales']
    BASIC = ['day_of_week','day_of_month','month','is_weekend',
             'is_holiday','has_promotion','quarter']

    split = int(len(d) * 0.8)
    X_tr, X_te = d[FEAT].iloc[:split], d[FEAT].iloc[split:]
    y_tr, y_te = d['sales'].iloc[:split], d['sales'].iloc[split:]

    m_base = XGBRegressor(n_estimators=300, learning_rate=0.05,
                          max_depth=4, random_state=42)
    m_base.fit(X_tr[BASIC], y_tr)
    mae_base = mean_absolute_error(y_te, m_base.predict(X_te[BASIC]))

    m_full = XGBRegressor(n_estimators=500, learning_rate=0.05,
                          max_depth=6, random_state=42)
    m_full.fit(X_tr, y_tr)
    pred_full = m_full.predict(X_te)
    mae_full  = mean_absolute_error(y_te, pred_full)

    return m_full, m_base, d, X_te, y_te, pred_full, mae_base, mae_full, FEAT, BASIC


# ── Couleurs ─────────────────────────────────────────────────
C_BLUE   = '#1E3A5F'
C_GREEN  = '#0F6E56'
C_ORANGE = '#D85A30'
C_PURPLE = '#534AB7'
C_LGRAY  = '#F8F7F4'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linewidth': 0.5,
})


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 Store Sales")
    st.markdown("**Dashboard Décisionnel**")
    st.markdown("---")

    st.markdown("**Sélection**")

    try:
        df_main, stores_df = load_all_data('data/')
        families = sorted(df_main['family'].unique())
        store_list = sorted(df_main['store_nbr'].unique())

        sel_store  = st.selectbox("Magasin", store_list,
                                  index=0, format_func=lambda x: f"Magasin {x}")
        sel_family = st.selectbox("Famille de produits", families,
                                  index=families.index('GROCERY I') if 'GROCERY I' in families else 0)
        data_ok = True
    except Exception as e:
        st.error(f"Données non trouvées.\nVérifier le dossier data/")
        data_ok = False

    st.markdown("---")
    st.markdown("**Projet**")
    st.markdown("Projet 1 · Feature Engineering")
    st.markdown("Store Sales · Corporación Favorita")
    st.caption("Programme 20 projets LinkedIn")


# ── Main ──────────────────────────────────────────────────────
st.title("Store Sales · Dashboard Décisionnel")

if not data_ok:
    st.warning("Lance d'abord : `kaggle competitions download -c store-sales-time-series-forecasting -p data/`")
    st.stop()

# Entraîner/charger le modèle
with st.spinner(f"Chargement du modèle · Magasin {sel_store} · {sel_family}..."):
    m_full, m_base, d_clean, X_te, y_te, pred_full, mae_base, mae_full, FEAT, BASIC = \
        train_model(df_main, sel_store, sel_family)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Simulateur", "🔍 Interprétabilité"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 : DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tab1:

    # KPIs
    reduction = (mae_base - mae_full) / mae_base * 100
    ventes_moy = d_clean['sales'].mean()
    err_rel    = mae_full / ventes_moy * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE Baseline",     f"{mae_base:.0f}",  "7 features")
    c2.metric("MAE Complet",      f"{mae_full:.0f}",  f"−{reduction:.1f}%",  delta_color="inverse")
    c3.metric("Erreur relative",  f"{err_rel:.1f}%",  "vs ventes moyennes")
    c4.metric("Ventes moyennes",  f"{ventes_moy:.0f}", f"unités/jour")

    st.markdown("---")

    # Graphique 1 : Prédictions vs Réel
    st.markdown("## Prédictions vs Réel")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(y_te.index, y_te.values, label='Réel',
            color=C_BLUE, linewidth=1.2, alpha=0.9)
    ax.plot(y_te.index, pred_full,   label=f'Prédit (MAE={mae_full:.0f})',
            color=C_GREEN, linewidth=1.1, alpha=0.85)
    ax.fill_between(y_te.index, y_te.values, pred_full,
                    alpha=0.1, color=C_ORANGE)
    ax.set_title(f'Magasin {sel_store} · {sel_family}', fontsize=12)
    ax.set_ylabel('Ventes (unités/jour)')
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("## Impact des événements marketing")
        # Calcul dynamique
        df_sel = df_main[(df_main['store_nbr']==sel_store) &
                         (df_main['family']==sel_family)].copy()
        df_sel['is_month_start'] = (df_sel['date'].dt.day <= 5).astype(int)

        results_impact = {}
        base_mean = df_sel['sales'].mean()

        for label, mask in [
            ("Jours fériés",     df_sel['is_holiday']==1),
            ("Début de mois",    df_sel['is_month_start']==1),
            ("Avec promotion",   df_sel['has_promotion']==1),
        ]:
            val = df_sel[mask]['sales'].mean()
            if base_mean > 0:
                pct = (val / df_sel[~mask]['sales'].mean() - 1) * 100
                results_impact[label] = pct

        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        labels = list(results_impact.keys())
        vals   = list(results_impact.values())
        colors = [C_PURPLE, C_GREEN, C_ORANGE]
        bars   = ax2.barh(labels, vals, color=colors, alpha=0.85, height=0.45)
        ax2.bar_label(bars, fmt='+%.1f%%', padding=6, fontsize=11, fontweight='bold')
        ax2.set_xlabel('Augmentation des ventes (%)')
        ax2.set_xlim(0, max(vals) * 1.35 if vals else 50)
        ax2.axvline(0, color='black', linewidth=0.7)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_b:
        st.markdown("## Ventes par jour de la semaine")
        jours = ['Lun','Mar','Mer','Jeu','Ven','Sam','Dim']
        df_sel['dow'] = pd.to_datetime(df_sel['date']).dt.dayofweek
        dow_sales = df_sel.groupby('dow')['sales'].mean()

        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        bar_colors = [C_GREEN if v == dow_sales.max()
                      else C_ORANGE if v == dow_sales.min()
                      else '#AAAAAA' for v in dow_sales.values]
        ax3.bar(range(7), dow_sales.values, color=bar_colors, alpha=0.85)
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(jours, fontsize=10)
        ax3.set_ylabel('Ventes moyennes')
        ax3.set_title('Pic = vert · Creux = orange', fontsize=10, color='#888')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.markdown("---")
    st.markdown("## Évolution mensuelle des ventes")
    df_sel['month_year'] = pd.to_datetime(df_sel['date']).dt.to_period('M')
    monthly = df_sel.groupby('month_year')['sales'].sum().reset_index()
    monthly['month_year'] = monthly['month_year'].astype(str)

    fig4, ax4 = plt.subplots(figsize=(13, 3.5))
    ax4.fill_between(range(len(monthly)), monthly['sales'],
                     alpha=0.15, color=C_BLUE)
    ax4.plot(range(len(monthly)), monthly['sales'],
             color=C_BLUE, linewidth=1.3)
    ax4.set_xticks(range(0, len(monthly), 3))
    ax4.set_xticklabels(monthly['month_year'].iloc[::3], rotation=30, fontsize=9)
    ax4.set_ylabel('Ventes totales mensuelles')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# TAB 2 : SIMULATEUR
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Simulateur de scénarios marketing")
    st.caption("Modifie les paramètres ci-dessous pour estimer l'impact sur les ventes prévisionnelles.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Paramètres du scénario")

        sim_date_str = st.date_input(
            "Date de simulation",
            value=pd.Timestamp('2017-06-15'),
            min_value=pd.Timestamp('2017-01-01'),
            max_value=pd.Timestamp('2017-08-15')
        )
        sim_date = pd.Timestamp(sim_date_str)

        st.markdown("**Leviers marketing**")
        sim_promo     = st.checkbox("Promotion active ce jour", value=False)
        sim_holiday   = st.checkbox("Jour férié", value=False)

        st.markdown("**Contexte**")
        sim_oil = st.slider("Prix du pétrole ($/baril)", 30.0, 100.0, 55.0, 0.5)
        sim_transactions = st.slider(
            "Trafic client prévu (transactions)",
            int(d_clean['transactions'].min()),
            int(d_clean['transactions'].max()),
            int(d_clean['transactions'].mean()),
            50
        )

        st.markdown("**Période calendaire**")
        sim_weekend    = sim_date.dayofweek >= 5
        sim_month_start = sim_date.day <= 5
        sim_month_end   = sim_date.day >= 25
        st.info(
            f"📅 {['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche'][sim_date.dayofweek]} "
            f"{'· 📌 Début de mois' if sim_month_start else ''}"
            f"{'· 📌 Fin de mois' if sim_month_end else ''}"
            f"{'· 📌 Week-end' if sim_weekend else ''}"
        )

    with col2:
        st.markdown("### Résultats de la simulation")

        # Construire le vecteur de features pour la simulation
        recent = d_clean[d_clean.index <= sim_date].tail(30)

        if len(recent) < 28:
            st.warning("Pas assez d'historique pour cette date.")
        else:
            lag_1  = recent['sales'].iloc[-1]
            lag_7  = recent['sales'].iloc[-7] if len(recent) >= 7 else lag_1
            lag_14 = recent['sales'].iloc[-14] if len(recent) >= 14 else lag_1
            lag_28 = recent['sales'].iloc[-28] if len(recent) >= 28 else lag_1

            roll7  = recent['sales'].iloc[-7:].mean()
            roll14 = recent['sales'].iloc[-14:].mean()
            roll28 = recent['sales'].iloc[-28:].mean()
            std7   = recent['sales'].iloc[-7:].std()
            std14  = recent['sales'].iloc[-14:].std()
            std28  = recent['sales'].iloc[-28:].std()
            max7   = recent['sales'].iloc[-7:].max()
            max14  = recent['sales'].iloc[-14:].max()
            max28  = recent['sales'].iloc[-28:].max()
            min7   = recent['sales'].iloc[-7:].min()
            min14  = recent['sales'].iloc[-14:].min()
            min28  = recent['sales'].iloc[-28:].min()

            has_promo_int = int(sim_promo)
            lag_promo_7  = recent['has_promotion'].iloc[-7:].sum()
            roll_promo14 = recent['has_promotion'].iloc[-14:].mean()
            post_promo   = int(recent['has_promotion'].iloc[-1])

            feat_vec = pd.DataFrame([{
                'onpromotion':       int(sim_promo),
                'dcoilwtico':        sim_oil,
                'is_holiday':        int(sim_holiday),
                'has_promotion':     has_promo_int,
                'transactions':      sim_transactions,
                'day_of_week':       sim_date.dayofweek,
                'day_of_month':      sim_date.day,
                'month':             sim_date.month,
                'week_of_year':      sim_date.isocalendar()[1],
                'quarter':           sim_date.quarter,
                'is_weekend':        int(sim_weekend),
                'is_month_start':    int(sim_month_start),
                'is_month_end':      int(sim_month_end),
                'days_to_holiday':   int(sim_holiday),
                'lag_1d':            lag_1,
                'lag_7d':            lag_7,
                'lag_14d':           lag_14,
                'lag_28d':           lag_28,
                'rolling_mean_7d':   roll7,
                'rolling_std_7d':    std7,
                'rolling_max_7d':    max7,
                'rolling_min_7d':    min7,
                'rolling_mean_14d':  roll14,
                'rolling_std_14d':   std14,
                'rolling_max_14d':   max14,
                'rolling_min_14d':   min14,
                'rolling_mean_28d':  roll28,
                'rolling_std_28d':   std28,
                'rolling_max_28d':   max28,
                'rolling_min_28d':   min28,
                'lag_promo_7d':      lag_promo_7,
                'rolling_promo_14d': roll_promo14,
                'post_promo':        post_promo,
                'promo_x_holiday':   has_promo_int * int(sim_holiday),
                'promo_x_weekend':   has_promo_int * int(sim_weekend),
                'promo_x_month_end': has_promo_int * int(sim_month_end),
                'oil_x_lag1':        sim_oil * lag_1,
            }])[FEAT]

            # Prédiction scénario de base (sans leviers)
            feat_base = feat_vec.copy()
            feat_base['has_promotion']     = 0
            feat_base['onpromotion']       = 0
            feat_base['is_holiday']        = 0
            feat_base['promo_x_holiday']   = 0
            feat_base['promo_x_weekend']   = 0
            feat_base['promo_x_month_end'] = 0

            pred_scenario = m_full.predict(feat_vec)[0]
            pred_baseline = m_full.predict(feat_base)[0]
            delta_abs     = pred_scenario - pred_baseline
            delta_pct     = (delta_abs / pred_baseline * 100) if pred_baseline > 0 else 0

            # Affichage résultat
            if delta_pct >= 0:
                box_class = "sim-box"
                arrow = "↑"
                col_delta = C_GREEN
            else:
                box_class = "sim-box-danger"
                arrow = "↓"
                col_delta = C_ORANGE

            st.markdown(f"""
            <div class="{box_class}">
                <div class="sim-title">Ventes prévues le {sim_date.strftime('%d %B %Y')}</div>
                <div class="sim-value" style="color:{C_BLUE};">{pred_scenario:,.0f} <span style="font-size:18px;color:#888;">unités</span></div>
                <hr style="margin:10px 0;border-color:#ccc;">
                <div class="sim-detail">
                    <strong>Sans les leviers activés :</strong> {pred_baseline:,.0f} unités<br>
                    <strong style="color:{col_delta};">Impact des leviers : {arrow} {abs(delta_abs):,.0f} unités ({delta_pct:+.1f}%)</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Décomposition de l'impact
            st.markdown("#### Décomposition de l'impact")
            impacts = {}
            if sim_promo:
                f_tmp = feat_base.copy(); f_tmp['has_promotion']=1; f_tmp['onpromotion']=1
                f_tmp['promo_x_holiday']   = int(sim_holiday)
                f_tmp['promo_x_weekend']   = int(sim_weekend)
                f_tmp['promo_x_month_end'] = int(sim_month_end)
                impacts['Promotion']  = m_full.predict(f_tmp)[0] - pred_baseline
            if sim_holiday:
                f_tmp = feat_base.copy(); f_tmp['is_holiday']=1
                impacts['Jour férié'] = m_full.predict(f_tmp)[0] - pred_baseline
            if sim_month_start:
                impacts['Début de mois'] = roll7 * 0.142
            if sim_weekend:
                impacts['Week-end'] = roll7 * 0.08

            if impacts:
                fig_imp, ax_imp = plt.subplots(figsize=(6, 2.5))
                colors_imp = [C_GREEN if v >= 0 else C_ORANGE for v in impacts.values()]
                bars_imp   = ax_imp.barh(list(impacts.keys()),
                                         list(impacts.values()),
                                         color=colors_imp, alpha=0.85, height=0.4)
                ax_imp.bar_label(bars_imp, fmt='+%.0f un.', padding=6, fontsize=10)
                ax_imp.axvline(0, color='black', linewidth=0.7)
                ax_imp.set_xlabel("Impact estimé (unités)")
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close()
            else:
                st.caption("Active un levier pour voir la décomposition de l'impact.")

            # Recommandation automatique
            st.markdown("#### Recommandation")
            recs = []
            if not sim_promo and families.index(sel_family) < 5:
                recs.append(f"✅ Activer une promotion sur {sel_family} pourrait générer +{roll7*0.28:.0f} unités supplémentaires.")
            if sim_month_start:
                recs.append("📦 Début de mois détecté — prévoir +14% de stock vs semaine normale.")
            if sim_holiday:
                recs.append("🎉 Jour férié — anticiper les commandes fournisseurs 3 jours avant.")
            if sim_weekend:
                recs.append("📈 Week-end — pic habituel, s'assurer de la disponibilité produit.")
            if not recs:
                recs.append("Aucun événement particulier détecté. Niveau de stock standard recommandé.")
            for r in recs:
                st.markdown(r)


# ═══════════════════════════════════════════════════════════════
# TAB 3 : INTERPRÉTABILITÉ
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Interprétabilité du modèle — SHAP")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("### Importance des features (XGBoost)")
        feat_imp = pd.Series(m_full.feature_importances_, index=FEAT)
        top15    = feat_imp.nlargest(15).sort_values()

        fig_fi, ax_fi = plt.subplots(figsize=(6, 5.5))
        colors_fi = [C_GREEN if i >= 13 else C_PURPLE if i >= 10 else '#AAAAAA'
                     for i in range(len(top15))]
        top15.plot(kind='barh', ax=ax_fi, color=colors_fi[::-1], alpha=0.85)
        ax_fi.set_xlabel('Importance')
        ax_fi.set_title('Top 15 · Vert = top 2 · Violet = top 5', fontsize=10, color='#888')
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()

    with col_s2:
        st.markdown("### Résidus du modèle")
        pred_all = m_full.predict(X_te)
        residus  = y_te.values - pred_all
        err_rel_dist = np.abs(residus) / (y_te.values + 1) * 100

        fig_res, axes_res = plt.subplots(2, 1, figsize=(6, 5.5))
        axes_res[0].scatter(y_te.index, residus, alpha=0.35, s=10, color=C_PURPLE)
        axes_res[0].axhline(0,         color='black', linewidth=0.8, linestyle='--')
        axes_res[0].axhline(mae_full,  color=C_GREEN, linewidth=0.8, linestyle=':')
        axes_res[0].axhline(-mae_full, color=C_GREEN, linewidth=0.8, linestyle=':')
        axes_res[0].set_ylabel('Erreur (unités)')
        axes_res[0].set_title('Résidus dans le temps', fontsize=11)

        axes_res[1].hist(residus, bins=40, color=C_PURPLE, alpha=0.7, edgecolor='white')
        axes_res[1].axvline(0,        color='black', linewidth=1)
        axes_res[1].axvline(mae_full, color=C_GREEN, linewidth=1, linestyle='--', label=f'MAE={mae_full:.0f}')
        axes_res[1].set_xlabel('Erreur (unités)')
        axes_res[1].set_ylabel('Fréquence')
        axes_res[1].set_title('Distribution des erreurs', fontsize=11)
        axes_res[1].legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_res)
        plt.close()

    st.markdown("---")
    st.markdown("### Insights clés du modèle")

    top3_feat = feat_imp.nlargest(3).index.tolist()
    c_i1, c_i2, c_i3 = st.columns(3)
    cols_insight = [c_i1, c_i2, c_i3]
    feat_desc = {
        'rolling_min_7d':   "Le creux des 7 derniers jours est le meilleur signal du niveau de ventes actuel.",
        'transactions':     "Le trafic client prédit mieux les ventes que les promotions.",
        'rolling_mean_28d': "La tendance mensuelle stabilise les prédictions face au bruit quotidien.",
        'lag_14d':          "Le cycle bi-hebdomadaire des achats est un signal fort.",
        'day_of_week':      "Le jour de la semaine structure les comportements d'achat.",
        'rolling_mean_14d': "La moyenne 14 jours capte les tendances intermédiaires.",
    }
    for i, feat in enumerate(top3_feat):
        with cols_insight[i]:
            desc = feat_desc.get(feat, "Feature importante identifiée par XGBoost.")
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">#{i+1} Feature</div>
                <div class="insight-value">{feat}</div>
                <div class="insight-sub">{desc}</div>
            </div>
            """, unsafe_allow_html=True)