# app.py — Store Sales · Dashboard v3
# Charge les modèles pré-entraînés depuis models/
# Lancer : streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Store Sales · Dashboard",
    page_icon="🛒", layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #F8F7F4; }
.block-container { padding: 1.5rem 2rem 2rem; }
[data-testid="stSidebar"] { background: #1E3A5F !important; }
[data-testid="stSidebar"] * { color: #E8EEF5 !important; }
[data-testid="stSidebar"] label { color: #B8C8D8 !important; font-size: 13px; }
[data-testid="metric-container"] {
    background: white; border: 0.5px solid #E5E5E5;
    border-radius: 12px; padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="metric-container"] label { font-size: 12px !important; color: #888 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px !important; font-weight: 600 !important; color: #1E3A5F !important;
}
h1 { font-size: 22px !important; font-weight: 600 !important; color: #1E3A5F !important; }
h2 { font-size: 16px !important; font-weight: 500 !important; color: #1E3A5F !important; margin-top: 1.2rem !important; }
.card {
    background: white; border-radius: 12px; padding: 1rem 1.2rem;
    border: 0.5px solid #E5E5E5; margin-bottom: 8px;
}
.model-badge {
    display: inline-block; font-size: 11px; padding: 3px 10px;
    border-radius: 999px; font-weight: 500; margin-bottom: 8px;
}
.sim-box {
    background: linear-gradient(135deg,#E1F5EE,#F0FAF5);
    border: 1px solid #0F6E56; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-top: 1rem;
}
.sim-box-warn {
    background: linear-gradient(135deg,#FAECE7,#FDF5F2);
    border: 1px solid #D85A30; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-top: 1rem;
}
.stTabs [data-baseweb="tab"] { font-size: 13px; padding: 6px 14px; color: #666; }
.stTabs [aria-selected="true"] { background: white; color: #1E3A5F !important; font-weight: 500; }
hr { border: none; border-top: 0.5px solid #E5E5E5; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

C = {'blue':'#1E3A5F','green':'#0F6E56','orange':'#D85A30',
     'purple':'#534AB7','gray':'#888888','lgray':'#F8F7F4'}

plt.rcParams.update({
    'font.family':'DejaVu Sans','axes.spines.top':False,
    'axes.spines.right':False,'axes.facecolor':'white',
    'figure.facecolor':'white','axes.grid':True,
    'grid.alpha':0.35,'grid.linewidth':0.5,
})

# ── Chargement données ────────────────────────────────────────
@st.cache_data
def load_data(path='data/'):
    p = path.rstrip('/')
    train    = pd.read_csv(f'{p}/train.csv',           parse_dates=['date'])
    stores   = pd.read_csv(f'{p}/stores.csv')
    holidays = pd.read_csv(f'{p}/holidays_events.csv', parse_dates=['date'])
    oil      = pd.read_csv(f'{p}/oil.csv',             parse_dates=['date'])
    transac  = pd.read_csv(f'{p}/transactions.csv',    parse_dates=['date'])
    df = train.merge(stores, on='store_nbr', how='left')
    df = df.merge(transac,   on=['date','store_nbr'], how='left')
    df = df.merge(oil,       on='date', how='left')
    hol = (holidays[holidays['transferred']==False][['date','type','locale']]
           .rename(columns={'type':'holiday_type','locale':'holiday_locale'}))
    df = df.merge(hol, on='date', how='left')
    df['is_holiday']    = df['holiday_type'].notna().astype('int8')
    df['has_promotion'] = (df['onpromotion'] > 0).astype(int)
    df['dcoilwtico']    = df['dcoilwtico'].ffill().bfill()
    df['transactions']  = df['transactions'].fillna(0).astype('int32')
    return df

# ── Chargement modèles pré-entraînés ─────────────────────────
@st.cache_resource
def load_models():
    models = {}
    paths = {
        'XGBoost (notebook 02)':  'models/xgboost_grocery1_store1.pkl',
        'Ensemble (notebook 05)': 'models/ensemble_final.pkl',
        'Multi-familles (notebook 06)': 'models/multi_family_results.pkl',
    }
    missing = []
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            missing.append(path)
    return models, missing

# ── Features ─────────────────────────────────────────────────
@st.cache_data
def build_features(df_id, store_nbr, family):
    from src.data_loader import filter_series
    from src.feature_engineering import (add_calendar_features, add_lag_features,
                                          add_rolling_features, add_promo_features,
                                          add_interaction_features)
    d = filter_series(df, store_nbr, family)
    for fn in [add_calendar_features, add_lag_features,
               add_rolling_features, add_promo_features, add_interaction_features]:
        d = fn(d)
    t = (d.index - d.index.min()).days
    d['dow_sin']   = np.sin(2*np.pi*d.index.dayofweek/7)
    d['dow_cos']   = np.cos(2*np.pi*d.index.dayofweek/7)
    d['month_sin'] = np.sin(2*np.pi*d.index.month/12)
    d['month_cos'] = np.cos(2*np.pi*d.index.month/12)
    d['trend']     = t/365.25
    for k in [1,2,3]:
        d[f'fourier_sin_{k}'] = np.sin(2*np.pi*k*t/365.25)
        d[f'fourier_cos_{k}'] = np.cos(2*np.pi*k*t/365.25)
    d['rolling_mean_90d'] = d['sales'].shift(1).rolling(90,min_periods=30).mean()
    d['rolling_mean_60d'] = d['sales'].shift(1).rolling(60,min_periods=20).mean()
    d['ratio_7_28']  = (d['rolling_mean_7d']/d['rolling_mean_28d'].replace(0,np.nan)).fillna(1)
    d['spike_flag']  = (d['rolling_max_7d']>d['rolling_mean_28d']*1.5).astype('int8')
    return d.dropna()

# ── Prédiction selon modèle sélectionné ──────────────────────
def model_feature_names(model):
    """Retourne les features attendues par un modèle, si disponibles."""
    for attr in ['feature_names_in_', 'feature_name_', 'feature_names_']:
        names = getattr(model, attr, None)
        if callable(names):
            names = names()
        if names is not None:
            return list(names)
    try:
        names = model.get_booster().feature_names
        if names:
            return list(names)
    except Exception:
        pass
    try:
        names = model.booster_.feature_name()
        if names:
            return list(names)
    except Exception:
        pass
    return None

def align_features_for_model(model, X, FEAT):
    """Aligne les colonnes sur celles vues à l'entraînement."""
    expected = model_feature_names(model)
    if expected:
        present = [c for c in expected if c in X.columns]
        if not present:
            expected = None

    if expected:
        missing = [c for c in expected if c not in X.columns]
        if missing:
            X = X.copy()
            for c in missing:
                X[c] = 0
        return X[expected]

    n_features = getattr(model, 'n_features_in_', None)
    if n_features is not None and len(X.columns) != n_features:
        available = [c for c in FEAT if c in X.columns]
        if len(available) >= n_features:
            return X[available[:n_features]]

    return X

def predict_with_model(model_name, models, X, FEAT):
    """Dispatch vers le bon modèle selon la sélection."""
    if model_name == 'XGBoost (notebook 02)':
        m = models['XGBoost (notebook 02)']
        return m.predict(align_features_for_model(m, X, FEAT))

    elif model_name == 'Ensemble XGBoost+LightGBM pondéré':
        ens = models['Ensemble (notebook 05)']
        p_xgb  = ens['xgb'].predict(align_features_for_model(ens['xgb'], X, FEAT))
        p_lgbm = ens['lgbm'].predict(align_features_for_model(ens['lgbm'], X, FEAT))
        w = ens['poids']
        return w[0]*p_xgb + w[1]*p_lgbm

    elif model_name == 'Ensemble + CatBoost pondéré':
        ens = models['Ensemble (notebook 05)']
        p_xgb  = ens['xgb'].predict(align_features_for_model(ens['xgb'], X, FEAT))
        p_lgbm = ens['lgbm'].predict(align_features_for_model(ens['lgbm'], X, FEAT))
        p_cat  = ens['cat'].predict(align_features_for_model(ens['cat'], X, FEAT))
        w = ens['poids']
        return w[0]*p_xgb + w[1]*p_lgbm + w[2]*p_cat if len(w)>2 \
               else (p_xgb+p_lgbm+p_cat)/3

    elif model_name == 'Stacking Ridge':
        ens = models['Ensemble (notebook 05)']
        p_xgb  = ens['xgb'].predict(align_features_for_model(ens['xgb'], X, FEAT))
        p_lgbm = ens['lgbm'].predict(align_features_for_model(ens['lgbm'], X, FEAT))
        stack  = np.column_stack([p_xgb, p_lgbm])
        return ens['meta'].predict(stack)

    return np.zeros(len(X))

# ── Disponibilité des modèles ─────────────────────────────────
MODEL_OPTIONS = {
    'XGBoost (notebook 02)': {
        'requires': 'XGBoost (notebook 02)',
        'color': C['purple'],
        'desc': 'XGBoost entraîné dans le notebook 02 · 37 features',
        'notebook': '02'
    },
    'Ensemble XGBoost+LightGBM pondéré': {
        'requires': 'Ensemble (notebook 05)',
        'color': C['green'],
        'desc': 'Poids inversement proportionnels à la MAE · 53 features',
        'notebook': '05'
    },
    'Ensemble + CatBoost pondéré': {
        'requires': 'Ensemble (notebook 05)',
        'color': C['orange'],
        'desc': 'XGBoost + LightGBM + CatBoost · optimisé Optuna',
        'notebook': '05'
    },
    'Stacking Ridge': {
        'requires': 'Ensemble (notebook 05)',
        'color': C['blue'],
        'desc': 'Méta-modèle Ridge sur XGBoost + LightGBM · meilleur ensemble',
        'notebook': '05'
    },
}

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛒 Store Sales")
    st.markdown("**Dashboard — Modèles pré-entraînés**")
    st.markdown("---")

    try:
        df = load_data('data/')
        models, missing = load_models()
        families   = sorted(df['family'].unique())
        store_list = sorted(df['store_nbr'].unique())

        st.markdown("**Données**")
        sel_store  = st.selectbox("Magasin", store_list,
                                  format_func=lambda x: f"Magasin {x}")
        sel_family = st.selectbox("Famille principale", families,
                                  index=families.index('GROCERY I') if 'GROCERY I' in families else 0)

        st.markdown("---")
        st.markdown("**Modèle actif**")

        # Filtrer les modèles disponibles
        available = [k for k,v in MODEL_OPTIONS.items()
                     if v['requires'] in models]
        unavailable = [k for k,v in MODEL_OPTIONS.items()
                       if v['requires'] not in models]

        if not available:
            st.error("Aucun modèle trouvé dans models/\nEntraîner d'abord les notebooks.")
            st.stop()

        sel_model = st.selectbox("Choisir le modèle", available)
        mo = MODEL_OPTIONS[sel_model]
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 12px;margin-top:4px;">
            <div style="font-size:11px;color:#B8C8D8;margin-bottom:4px;">Notebook {mo['notebook']}</div>
            <div style="font-size:12px;color:#E8EEF5;line-height:1.5;">{mo['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        if unavailable:
            st.markdown("---")
            st.markdown("**Modèles non disponibles**")
            for m in unavailable:
                st.caption(f"⚠ {m}")
            st.caption("Entraîner notebooks 05 et 06 pour les débloquer.")

        st.markdown("---")
        st.markdown("**Multi-familles**")
        n_fam = st.slider("Nb familles", 3, 10, 6)
        top_fams = (df[df['store_nbr']==sel_store]
                    .groupby('family')['sales'].sum()
                    .sort_values(ascending=False)
                    .head(n_fam).index.tolist())

        data_ok = True
    except Exception as e:
        st.error(f"Erreur : {e}")
        data_ok = False

    st.markdown("---")
    st.caption("Projet 1 · Programme 20 projets LinkedIn")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
st.title("Store Sales · Dashboard")
st.caption(f"Modèle actif : **{sel_model}** · Magasin {sel_store} · {sel_family}")

if not data_ok:
    st.stop()

# Construire features + prédictions
with st.spinner("Chargement des features..."):
    d = build_features(id(df), sel_store, sel_family)
    FEAT  = [c for c in d.columns if c != 'sales']
    split = int(len(d)*0.8)
    X_tr, X_te = d[FEAT].iloc[:split], d[FEAT].iloc[split:]
    y_tr, y_te = d['sales'].iloc[:split], d['sales'].iloc[split:]

    pred_active = predict_with_model(sel_model, models, X_te, FEAT)
    mae_active  = mean_absolute_error(y_te, pred_active)

    # Baseline toujours calculé pour comparaison
    BASIC = ['day_of_week','day_of_month','month','is_weekend',
             'is_holiday','has_promotion','quarter']
    if 'XGBoost (notebook 02)' in models:
        pred_base    = predict_with_model('XGBoost (notebook 02)', models, X_te, FEAT)
        mae_base     = mean_absolute_error(y_te, pred_base)
    else:
        pred_base = pred_active
        mae_base  = mae_active


tabs = st.tabs(["Dashboard","Comparaison modèles","Simulateur",
                "Multi-familles","Interprétabilité"])


# ════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
with tabs[0]:
    red_pct  = (mae_base - mae_active)/mae_base*100 if mae_base != mae_active else 0
    err_rel  = mae_active/y_te.mean()*100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Modèle actif",    sel_model.split('(')[0].strip(), f"Notebook {mo['notebook']}")
    c2.metric("MAE",             f"{mae_active:.0f}", f"−{red_pct:.1f}% vs baseline" if red_pct>0 else "—", delta_color="inverse")
    c3.metric("Erreur relative", f"{err_rel:.1f}%",  "vs ventes moyennes")
    c4.metric("Jours de test",   len(y_te), f"{split} jours d'entraînement")

    st.markdown("---")
    st.markdown(f"## Prédictions vs Réel")

    fig, ax = plt.subplots(figsize=(13,4))
    ax.plot(y_te.index, y_te.values,  label='Réel',
            color=C['blue'],  linewidth=1.2, alpha=0.9)
    ax.plot(y_te.index, pred_active,
            label=f'{sel_model.split("(")[0].strip()} (MAE={mae_active:.0f})',
            color=mo['color'], linewidth=1.1, alpha=0.85)
    ax.fill_between(y_te.index, y_te.values, pred_active,
                    alpha=0.08, color=C['orange'])
    ax.set_ylabel('Ventes'); ax.legend(fontsize=10)
    ax.set_title(f'Magasin {sel_store} · {sel_family}', fontsize=12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    cola, colb = st.columns(2)

    with cola:
        st.markdown("## Impact événements marketing")
        df_s = df[(df['store_nbr']==sel_store)&(df['family']==sel_family)].copy()
        df_s['is_month_start'] = (df_s['date'].dt.day<=5).astype(int)
        impacts = {}
        for lbl, mask in [("Jours fériés",  df_s['is_holiday']==1),
                           ("Début de mois", df_s['is_month_start']==1),
                           ("Promotion",     df_s['has_promotion']==1)]:
            mn = df_s[~mask]['sales'].mean()
            if mn>0: impacts[lbl] = (df_s[mask]['sales'].mean()/mn-1)*100
        fig2,ax2 = plt.subplots(figsize=(6,3))
        bc = [C['purple'],C['green'],C['orange']]
        bars2 = ax2.barh(list(impacts.keys()),list(impacts.values()),color=bc,alpha=0.85,height=0.4)
        ax2.bar_label(bars2,fmt='+%.1f%%',padding=6,fontsize=11,fontweight='bold')
        ax2.set_xlabel('Augmentation des ventes (%)'); ax2.axvline(0,color='black',linewidth=0.7)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    with colb:
        st.markdown("## Ventes par jour de la semaine")
        df_s['dow'] = pd.to_datetime(df_s['date']).dt.dayofweek
        dow = df_s.groupby('dow')['sales'].mean()
        jours = ['Lun','Mar','Mer','Jeu','Ven','Sam','Dim']
        fig3,ax3 = plt.subplots(figsize=(6,3))
        bc3 = [C['green'] if v==dow.max() else C['orange'] if v==dow.min() else '#BBBBBB' for v in dow.values]
        ax3.bar(range(7),dow.values,color=bc3,alpha=0.85)
        ax3.set_xticks(range(7)); ax3.set_xticklabels(jours,fontsize=10)
        ax3.set_ylabel('Ventes moyennes')
        plt.tight_layout(); st.pyplot(fig3); plt.close()


# ════════════════════════════════════════════════════════════
# TAB 2 — COMPARAISON MODÈLES
# ════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Comparaison de tous les modèles disponibles")

    # Calculer MAE pour chaque modèle disponible
    with st.spinner("Calcul des performances de chaque modèle..."):
        comp_results = {}
        for mname in available:
            try:
                p = predict_with_model(mname, models, X_te, FEAT)
                comp_results[mname] = {
                    'mae':    mean_absolute_error(y_te, p),
                    'pred':   p,
                    'color':  MODEL_OPTIONS[mname]['color'],
                    'nb':     MODEL_OPTIONS[mname]['notebook'],
                    'desc':   MODEL_OPTIONS[mname]['desc'],
                }
            except Exception as e:
                st.warning(f"Erreur sur {mname} : {e}")

    if not comp_results:
        st.warning("Aucun modèle à comparer.")
    else:
        # Trier par MAE
        sorted_models = sorted(comp_results.items(), key=lambda x: x[1]['mae'])
        best_name, best_r = sorted_models[0]
        mae_worst = sorted_models[-1][1]['mae']

        # KPIs
        cols_kpi = st.columns(len(comp_results))
        for i,(mname,r) in enumerate(sorted_models):
            delta = f"−{(mae_worst-r['mae'])/mae_worst*100:.1f}% vs pire" if i>0 else "🏆 Meilleur"
            cols_kpi[i].metric(
                mname.split('(')[0].strip()[:18],
                f"{r['mae']:.0f}",
                delta,
                delta_color="inverse" if i>0 else "off"
            )

        st.markdown("---")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("### MAE par modèle")
            fig4,ax4 = plt.subplots(figsize=(7,4))
            noms4  = [m.split('(')[0].strip()[:20] for m,_ in sorted_models]
            vals4  = [r['mae'] for _,r in sorted_models]
            cols4  = [r['color'] for _,r in sorted_models]
            bars4  = ax4.barh(noms4, vals4, color=cols4, alpha=0.85, height=0.5)
            ax4.bar_label(bars4, fmt='%.0f', padding=5, fontsize=11, fontweight='bold')
            # Badge meilleur
            ax4.text(vals4[0]+2, 0, '🏆', fontsize=14, va='center')
            ax4.set_xlabel('MAE (Mean Absolute Error)')
            ax4.set_title('Plus la barre est courte, mieux c\'est', fontsize=10, color='#888')
            ax4.invert_yaxis()
            plt.tight_layout(); st.pyplot(fig4); plt.close()

        with col_r:
            st.markdown("### Résidus comparés")
            fig5,ax5 = plt.subplots(figsize=(7,4))
            for mname,r in sorted_models:
                res = y_te.values - r['pred']
                ax5.hist(res, bins=35, alpha=0.4, color=r['color'],
                         label=f"{mname.split('(')[0].strip()[:15]} (σ={np.std(res):.0f})",
                         edgecolor='white')
            ax5.axvline(0,color='black',linewidth=1)
            ax5.set_xlabel('Erreur (unités)'); ax5.set_ylabel('Fréquence')
            ax5.set_title('Distribution des erreurs par modèle', fontsize=11)
            ax5.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig5); plt.close()

        st.markdown("---")
        st.markdown("### Prédictions vs Réel — comparer deux modèles")
        mc1, mc2 = st.columns(2)
        with mc1:
            m_left = st.selectbox("Modèle A", list(comp_results.keys()), key='ml')
        with mc2:
            m_right = st.selectbox("Modèle B", list(comp_results.keys()),
                                   index=min(1,len(comp_results)-1), key='mr')

        fig6,axes6 = plt.subplots(2,1,figsize=(13,7),sharex=True)
        for ax6, mname in zip(axes6,[m_left,m_right]):
            r = comp_results[mname]
            ax6.plot(y_te.index, y_te.values, label='Réel',
                     color=C['blue'], linewidth=1.2, alpha=0.9)
            ax6.plot(y_te.index, r['pred'],
                     label=f"{mname.split('(')[0].strip()} (MAE={r['mae']:.0f})",
                     color=r['color'], linewidth=1.1)
            ax6.fill_between(y_te.index, y_te.values, r['pred'],
                             alpha=0.08, color=C['orange'])
            ax6.set_ylabel('Ventes'); ax6.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig6); plt.close()

        # Tableau récap
        st.markdown("### Tableau récapitulatif")
        recap_df = pd.DataFrame([{
            'Modèle': mname,
            'Notebook': r['nb'],
            'MAE': f"{r['mae']:.1f}",
            'Erreur relative': f"{r['mae']/y_te.mean()*100:.1f}%",
            'Description': MODEL_OPTIONS[mname]['desc']
        } for mname,r in sorted_models])
        st.dataframe(recap_df, width='stretch', hide_index=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — SIMULATEUR
# ════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(f"## Simulateur — modèle : **{sel_model}**")
    st.caption("Modifie les paramètres pour estimer les ventes prévisionnelles.")

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("### Paramètres du scénario")
        sim_date = pd.Timestamp(st.date_input("Date",
            value=pd.Timestamp('2017-06-15'),
            min_value=pd.Timestamp('2017-01-01'),
            max_value=pd.Timestamp('2017-08-15')))

        st.markdown("**Leviers**")
        sim_promo   = st.checkbox("Promotion active", value=False)
        sim_holiday = st.checkbox("Jour férié",       value=False)

        st.markdown("**Contexte**")
        sim_oil   = st.slider("Prix pétrole ($/baril)", 30.0,100.0,55.0,0.5)
        sim_trans = st.slider("Trafic client",
                              int(d['transactions'].min()),
                              int(d['transactions'].max()),
                              int(d['transactions'].mean()), 50)
        sim_weekend     = sim_date.dayofweek >= 5
        sim_month_start = sim_date.day <= 5
        sim_month_end   = sim_date.day >= 25

        flags = ([f"{'Week-end'   if sim_weekend else ''}",
                  f"{'Début mois' if sim_month_start else ''}",
                  f"{'Fin de mois' if sim_month_end else ''}"])
        flags = [f for f in flags if f]
        st.info(f"📅 {['Lun','Mar','Mer','Jeu','Ven','Sam','Dim'][sim_date.dayofweek]}"
                + (f" · {' · '.join(flags)}" if flags else ""))

    with col2:
        st.markdown("### Résultat")
        recent = d[d.index <= sim_date].tail(90)

        if len(recent) < 30:
            st.warning("Historique insuffisant.")
        else:
            def glg(n): return recent['sales'].iloc[-n] if len(recent)>=n else recent['sales'].mean()
            def rol(col,n,fn='mean'):
                s = recent[col].iloc[-n:]
                return getattr(s,fn)() if len(s)>0 else 0

            hp = int(sim_promo)
            t_days = (sim_date - d.index.min()).days
            fv = pd.DataFrame([{
                'onpromotion':hp,'dcoilwtico':sim_oil,'is_holiday':int(sim_holiday),
                'has_promotion':hp,'transactions':sim_trans,
                'day_of_week':sim_date.dayofweek,'day_of_month':sim_date.day,
                'month':sim_date.month,'week_of_year':sim_date.isocalendar()[1],
                'quarter':sim_date.quarter,'is_weekend':int(sim_weekend),
                'is_month_start':int(sim_month_start),'is_month_end':int(sim_month_end),
                'days_to_holiday':int(sim_holiday),
                'lag_1d':glg(1),'lag_7d':glg(7),'lag_14d':glg(14),'lag_28d':glg(28),
                'rolling_mean_7d':rol('sales',7),'rolling_std_7d':rol('sales',7,'std'),
                'rolling_max_7d':rol('sales',7,'max'),'rolling_min_7d':rol('sales',7,'min'),
                'rolling_mean_14d':rol('sales',14),'rolling_std_14d':rol('sales',14,'std'),
                'rolling_max_14d':rol('sales',14,'max'),'rolling_min_14d':rol('sales',14,'min'),
                'rolling_mean_28d':rol('sales',28),'rolling_std_28d':rol('sales',28,'std'),
                'rolling_max_28d':rol('sales',28,'max'),'rolling_min_28d':rol('sales',28,'min'),
                'lag_promo_7d':rol('has_promotion',7,'sum'),
                'rolling_promo_14d':rol('has_promotion',14),
                'post_promo':int(recent['has_promotion'].iloc[-1]),
                'promo_x_holiday':hp*int(sim_holiday),'promo_x_weekend':hp*int(sim_weekend),
                'promo_x_month_end':hp*int(sim_month_end),'oil_x_lag1':sim_oil*glg(1),
                'dow_sin':np.sin(2*np.pi*sim_date.dayofweek/7),
                'dow_cos':np.cos(2*np.pi*sim_date.dayofweek/7),
                'month_sin':np.sin(2*np.pi*sim_date.month/12),
                'month_cos':np.cos(2*np.pi*sim_date.month/12),
                'trend':t_days/365.25,
                'fourier_sin_1':np.sin(2*np.pi*1*t_days/365.25),
                'fourier_cos_1':np.cos(2*np.pi*1*t_days/365.25),
                'fourier_sin_2':np.sin(2*np.pi*2*t_days/365.25),
                'fourier_cos_2':np.cos(2*np.pi*2*t_days/365.25),
                'fourier_sin_3':np.sin(2*np.pi*3*t_days/365.25),
                'fourier_cos_3':np.cos(2*np.pi*3*t_days/365.25),
                'rolling_mean_90d':rol('sales',90),'rolling_mean_60d':rol('sales',60),
                'ratio_7_28': rol('sales',7)/max(rol('sales',28),1),
                'spike_flag': int(rol('sales',7,'max')>rol('sales',28)*1.5),
            }])[FEAT]

            fv_base = fv.copy()
            fv_base[['has_promotion','onpromotion','is_holiday',
                     'promo_x_holiday','promo_x_weekend','promo_x_month_end']] = 0

            pred_s = predict_with_model(sel_model, models, fv, FEAT)[0]
            pred_b = predict_with_model(sel_model, models, fv_base, FEAT)[0]
            delta  = pred_s - pred_b
            pct    = delta/pred_b*100 if pred_b>0 else 0
            box_cls = "sim-box" if delta>=0 else "sim-box-warn"
            col_d   = C['green'] if delta>=0 else C['orange']
            arrow   = "↑" if delta>=0 else "↓"

            st.markdown(f"""
            <div class="{box_cls}">
                <div style="font-size:12px;color:#555;margin-bottom:6px;">
                    {sim_date.strftime('%d %B %Y')} · {sel_model.split('(')[0].strip()}
                </div>
                <div style="font-size:36px;font-weight:600;color:{C['blue']};">
                    {pred_s:,.0f}
                    <span style="font-size:18px;color:#888;">unités</span>
                </div>
                <hr style="margin:10px 0;border-color:#ccc;">
                <div style="font-size:12px;color:#555;line-height:1.7;">
                    <b>Sans leviers :</b> {pred_b:,.0f} unités<br>
                    <b style="color:{col_d};">Impact : {arrow} {abs(delta):,.0f} unités ({pct:+.1f}%)</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Recommandations")
            recs = []
            if sim_month_start: recs.append("Début de mois — +14% de stock recommandé")
            if sim_holiday:     recs.append("Jour férié — commander 3 jours avant")
            if sim_weekend:     recs.append("Week-end — vérifier la disponibilité")
            if not sim_promo:
                recs.append(f"Une promo générerait ~+{rol('sales',7)*0.28:.0f} unités supplémentaires")
            if not recs: recs.append("Journée standard — stock habituel recommandé.")
            for r in recs: st.markdown(r)


# ════════════════════════════════════════════════════════════
# TAB 4 — MULTI-FAMILLES
# ════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## Analyse multi-familles")

    if 'Multi-familles (notebook 06)' in models:
        MF = models['Multi-familles (notebook 06)']
        fams_av = [f for f in top_fams if f in MF]

        if not fams_av:
            st.info("Lancer le notebook 06 avec les mêmes familles pour voir les résultats.")
        else:
            maes_rel = [MF[f]['mae_rel'] for f in fams_av]
            order    = np.argsort(maes_rel)
            fams_s   = [fams_av[i] for i in order]
            maes_s   = [maes_rel[i] for i in order]

            c1,c2,c3 = st.columns(3)
            c1.metric("Familles disponibles", len(MF))
            c2.metric("Meilleure famille",    fams_s[0], f"{maes_s[0]:.1f}%")
            c3.metric("Erreur moy.",          f"{np.mean(maes_rel):.1f}%")

            st.markdown("---")
            col_l, col_r = st.columns([1.2,1])

            with col_l:
                st.markdown("### Prévisibilité par famille")
                fig_mf,ax_mf = plt.subplots(figsize=(7,5))
                cms = [C['green'] if m<20 else C['purple'] if m<35 else C['orange'] for m in maes_s]
                bars_mf = ax_mf.barh(fams_s,maes_s,color=cms,alpha=0.85,height=0.5)
                ax_mf.bar_label(bars_mf,fmt='%.1f%%',padding=5,fontsize=10,fontweight='bold')
                ax_mf.axvline(20,color=C['green'],linestyle='--',linewidth=1,alpha=0.7,label='Excellent (<20%)')
                ax_mf.axvline(35,color=C['orange'],linestyle='--',linewidth=1,alpha=0.7,label='Acceptable (<35%)')
                ax_mf.set_xlabel('Erreur relative (%)'); ax_mf.invert_yaxis()
                ax_mf.legend(fontsize=9)
                plt.tight_layout(); st.pyplot(fig_mf); plt.close()

            with col_r:
                st.markdown("### Synthèse décisionnelle")
                for fam in fams_s:
                    rel = MF[fam]['mae_rel']
                    if rel<20:   icon,txt,col = "🟢","Automatisable",C['green']
                    elif rel<35: icon,txt,col = "🟡","Valider manuellement",C['purple']
                    else:        icon,txt,col = "🔴","Enrichir les features",C['orange']
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;justify-content:space-between;
                                padding:7px 12px;background:white;border-radius:8px;
                                border:0.5px solid #E5E5E5;margin-bottom:6px;">
                        <span style="font-size:13px;font-weight:500;">{icon} {fam[:22]}</span>
                        <span style="font-size:12px;color:{col};font-weight:500;">{rel:.1f}% · {txt}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Prédictions par famille")
            sel_fam_mf = st.selectbox("Famille", fams_av, label_visibility="collapsed")
            if sel_fam_mf in MF:
                r_mf = MF[sel_fam_mf]
                fig_p,ax_p = plt.subplots(figsize=(13,3.5))
                ax_p.plot(r_mf['y_te'].index, r_mf['y_te'].values,
                          label='Réel', color=C['blue'], linewidth=1.2)
                ax_p.plot(r_mf['y_te'].index, r_mf['pred'],
                          label=f"Prédit (MAE={r_mf['mae']:.0f} · {r_mf['mae_rel']:.1f}%)",
                          color=C['green'], linewidth=1.1)
                ax_p.fill_between(r_mf['y_te'].index, r_mf['y_te'].values,
                                  r_mf['pred'], alpha=0.08, color=C['orange'])
                ax_p.set_ylabel('Ventes'); ax_p.legend(fontsize=10)
                ax_p.set_title(f"{sel_fam_mf} · Magasin {sel_store}", fontsize=12)
                plt.tight_layout(); st.pyplot(fig_p); plt.close()
    else:
        st.info("Les résultats multi-familles ne sont pas disponibles.\nLance le notebook 06 pour les générer.")
        st.code("# Dans notebooks/06_multi_family.ipynb\n# Exécuter toutes les cellules\n# models/multi_family_results.pkl sera créé automatiquement")


# ════════════════════════════════════════════════════════════
# TAB 5 — INTERPRÉTABILITÉ
# ════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Interprétabilité")

    # Choisir quel modèle interpréter
    interp_model = st.selectbox(
        "Modèle à interpréter",
        [k for k in available if 'XGBoost' in k or 'Ensemble' in k],
        key='interp'
    )

    # Extraire le modèle XGBoost du bon endroit
    try:
        if 'XGBoost (notebook 02)' in interp_model:
            m_interp = models['XGBoost (notebook 02)']
        else:
            ens = models['Ensemble (notebook 05)']
            m_interp = ens['xgb']

        imp_values = m_interp.feature_importances_
        imp_features = model_feature_names(m_interp) or FEAT
        if len(imp_features) != len(imp_values):
            imp_features = FEAT[:len(imp_values)] if len(FEAT) >= len(imp_values) \
                else [f'feature_{i}' for i in range(len(imp_values))]
        imp = pd.Series(imp_values, index=imp_features)
    except:
        st.warning("Importance non disponible pour ce modèle.")
        st.stop()

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("### Top 15 features")
        top15 = imp.nlargest(15).sort_values()
        fig_fi,ax_fi = plt.subplots(figsize=(6,5.5))
        ci = [C['green'] if i>=13 else C['purple'] if i>=10 else '#CCCCCC'
              for i in range(len(top15))]
        top15.plot(kind='barh',ax=ax_fi,color=ci[::-1],alpha=0.85)
        ax_fi.set_xlabel('Importance')
        ax_fi.set_title('Vert = top 2 · Violet = top 5', fontsize=10, color='#888')
        plt.tight_layout(); st.pyplot(fig_fi); plt.close()

    with col_s2:
        st.markdown("### Résidus")
        pred_interp = predict_with_model(interp_model, models, X_te, FEAT)
        res = y_te.values - pred_interp
        mae_interp = mean_absolute_error(y_te, pred_interp)
        fig_r,axes_r = plt.subplots(2,1,figsize=(6,5.5))
        axes_r[0].scatter(y_te.index,res,alpha=0.35,s=10,color=C['purple'])
        axes_r[0].axhline(0,color='black',linewidth=0.8,linestyle='--')
        axes_r[0].axhline( mae_interp,color=C['green'],linewidth=0.8,linestyle=':')
        axes_r[0].axhline(-mae_interp,color=C['green'],linewidth=0.8,linestyle=':')
        axes_r[0].set_ylabel('Erreur'); axes_r[0].set_title('Résidus', fontsize=11)
        axes_r[1].hist(res,bins=40,color=C['purple'],alpha=0.7,edgecolor='white')
        axes_r[1].axvline(0,color='black',linewidth=1)
        axes_r[1].set_xlabel('Erreur'); axes_r[1].set_ylabel('Fréquence')
        axes_r[1].set_title('Distribution des erreurs', fontsize=11)
        plt.tight_layout(); st.pyplot(fig_r); plt.close()

    st.markdown("---")
    st.markdown("### Insights clés")
    top3 = imp.nlargest(3).index.tolist()
    descs = {
        'rolling_min_7d':   "Le creux des 7 derniers jours est le meilleur signal du niveau de ventes.",
        'transactions':     "Le trafic client prédit mieux les ventes que les promotions.",
        'rolling_mean_28d': "La tendance mensuelle stabilise les prédictions.",
        'trend':            "La croissance progressive des ventes sur 4 ans.",
        'fourier_sin_1':    "La saisonnalité annuelle capturée par Fourier.",
        'dow_sin':          "L'encodage cyclique performe mieux qu'un encodage naïf.",
        'lag_7d':           "Les ventes de la même semaine précédente.",
        'rolling_mean_7d':  "La tendance des 7 derniers jours.",
    }
    ci3 = st.columns(3)
    for i,feat in enumerate(top3):
        with ci3[i]:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">#{i+1} Feature</div>
                <div style="font-size:15px;font-weight:600;color:{C['blue']};margin-bottom:5px;">{feat}</div>
                <div style="font-size:12px;color:#555;line-height:1.5;">{descs.get(feat,'Feature clé identifiée par XGBoost.')}</div>
            </div>
            """, unsafe_allow_html=True)
