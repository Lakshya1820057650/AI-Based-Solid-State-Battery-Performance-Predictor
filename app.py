"""
AI-Based Solid State Battery Performance Predictor
Streamlit App — deployable to Streamlit Community Cloud
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Battery Performance Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #e6edf3; }

    .metric-card {
        background: linear-gradient(135deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #58a6ff;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-unit {
        font-size: 0.8rem;
        color: #8b949e;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f0f6fc;
        border-left: 3px solid #58a6ff;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
        font-family: 'IBM Plex Mono', monospace;
    }
    .stSlider > div > div > div { background: #58a6ff !important; }
    .predict-result {
        background: linear-gradient(135deg, #0d2137, #112240);
        border: 1px solid #58a6ff44;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-title {
        font-size: 0.7rem;
        color: #58a6ff;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 0.3rem;
    }
    .result-val {
        font-size: 2.4rem;
        font-weight: 600;
        color: #e6edf3;
        font-family: 'IBM Plex Mono', monospace;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA & MODEL (cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def generate_dataset(n_samples=1200, random_state=42):
    np.random.seed(random_state)
    n = n_samples
    band_gap           = np.random.uniform(0.0, 6.0, n)
    formation_energy   = np.random.uniform(-4.0, 0.5, n)
    density            = np.random.uniform(1.5, 8.0, n)
    volume             = np.random.uniform(20, 500, n)
    nsites             = np.random.randint(2, 30, n).astype(float)
    ionic_radius_mean  = np.random.uniform(0.3, 1.8, n)
    electronegativity  = np.random.uniform(0.8, 3.5, n)
    oxidation_state    = np.random.uniform(-3.0, 5.0, n)
    bulk_modulus       = np.random.uniform(10, 300, n)
    shear_modulus      = np.random.uniform(5, 200, n)
    e_above_hull       = np.abs(np.random.normal(0.05, 0.08, n))

    energy_density = (
        250 - 30*e_above_hull - 15*band_gap + 20*np.abs(formation_energy)
        + 10*bulk_modulus/100 - 8*density + 5*oxidation_state
        + np.random.normal(0, 15, n)
    )
    energy_density = np.clip(energy_density, 80, 500)

    electrochemical_stability = (
        3.0 + 0.4*band_gap - 0.3*e_above_hull + 0.2*electronegativity
        - 0.1*np.abs(formation_energy) + np.random.normal(0, 0.3, n)
    )
    electrochemical_stability = np.clip(electrochemical_stability, 0.5, 7.0)

    df = pd.DataFrame({
        "band_gap": band_gap, "formation_energy": formation_energy,
        "density": density, "volume": volume, "nsites": nsites,
        "ionic_radius_mean": ionic_radius_mean, "electronegativity": electronegativity,
        "oxidation_state": oxidation_state, "bulk_modulus": bulk_modulus,
        "shear_modulus": shear_modulus, "e_above_hull": e_above_hull,
        "energy_density": energy_density,
        "electrochemical_stability": electrochemical_stability,
    })
    return df


@st.cache_data(show_spinner=False)
def feature_engineering(df):
    df = df.copy()
    df["bulk_shear_ratio"]  = df["bulk_modulus"] / (df["shear_modulus"] + 1e-6)
    df["vol_per_site"]      = df["volume"] / df["nsites"]
    df["stability_flag"]    = (df["e_above_hull"] < 0.1).astype(int)
    df["band_gap_sq"]       = df["band_gap"] ** 2
    df["formation_density"] = df["formation_energy"] * df["density"]
    return df


@st.cache_resource(show_spinner=False)
def train_models(df_hash):
    df = feature_engineering(generate_dataset())
    feature_cols = [c for c in df.columns if c not in ["energy_density", "electrochemical_stability"]]
    results = {}
    for target in ["energy_density", "electrochemical_stability"]:
        X = df[feature_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        model = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=4,
            min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        results[target] = {
            "model": model, "scaler": scaler,
            "X_test": X_test_s, "y_test": y_test, "y_pred": y_pred,
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "cv": cross_val_score(model, X_train_s, y_train, cv=5, scoring="r2").mean(),
            "feature_cols": feature_cols,
            "importances": pd.Series(model.feature_importances_, index=feature_cols),
        }
    return results


@st.cache_data(show_spinner=False)
def screen_top_materials(df, top_n=10):
    df = df.copy()
    df["ed_norm"]   = (df["energy_density"] - df["energy_density"].min()) / (df["energy_density"].max() - df["energy_density"].min())
    df["es_norm"]   = (df["electrochemical_stability"] - df["electrochemical_stability"].min()) / (df["electrochemical_stability"].max() - df["electrochemical_stability"].min())
    df["stab_norm"] = 1 - (df["e_above_hull"] - df["e_above_hull"].min()) / (df["e_above_hull"].max() - df["e_above_hull"].min())
    df["composite_score"] = 0.5*df["ed_norm"] + 0.3*df["es_norm"] + 0.2*df["stab_norm"]
    top = df.nlargest(top_n, "composite_score")[
        ["band_gap","formation_energy","density","e_above_hull",
         "energy_density","electrochemical_stability","composite_score"]
    ].reset_index(drop=True)
    top.index += 1
    return top


# ─────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────

def dark_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")
    return fig, ax


def plot_actual_vs_predicted(res_ed, res_es):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax, res, lbl, unit, color in zip(
            axes,
            [res_ed, res_es],
            ["Energy Density", "Electrochemical Stability"],
            ["Wh/kg", "V"],
            ["#58a6ff", "#3fb950"]):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.scatter(res["y_test"], res["y_pred"], alpha=0.35, color=color, edgecolors="none", s=18)
        mn = min(res["y_test"].min(), res["y_pred"].min())
        mx = max(res["y_test"].max(), res["y_pred"].max())
        ax.plot([mn, mx], [mn, mx], "--", color="#f0f6fc", lw=1.2, alpha=0.6, label="Ideal")
        ax.set_xlabel(f"Actual ({unit})", color="#8b949e")
        ax.set_ylabel(f"Predicted ({unit})", color="#8b949e")
        ax.set_title(f"Actual vs Predicted — {lbl}", color="#e6edf3")
        ax.legend(facecolor="#0d1117", labelcolor="#8b949e", edgecolor="#30363d")
        ax.text(0.05, 0.92, f"R² = {res['r2']:.3f}", transform=ax.transAxes,
                fontsize=10, color=color, fontfamily="monospace")
    plt.tight_layout()
    return fig


def plot_feature_importance(importances, title):
    fig, ax = dark_fig(figsize=(9, 6))
    importances.sort_values().plot.barh(ax=ax, color="#58a6ff", edgecolor="#0d1117", linewidth=0.5)
    ax.set_title(title, color="#e6edf3")
    ax.set_xlabel("Importance Score", color="#8b949e")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.3, ax=ax, annot_kws={"size": 7},
                linecolor="#0d1117")
    ax.set_title("Feature Correlation Heatmap", color="#e6edf3", pad=12)
    ax.tick_params(colors="#8b949e")
    plt.tight_layout()
    return fig


def plot_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    for ax, col, lbl, color in zip(
            axes,
            ["energy_density", "electrochemical_stability"],
            ["Energy Density (Wh/kg)", "Electrochemical Stability (V)"],
            ["#58a6ff", "#3fb950"]):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        sns.histplot(df[col], bins=40, kde=True, ax=ax, color=color)
        ax.set_title(f"Distribution — {lbl}", color="#e6edf3")
        ax.set_xlabel(lbl, color="#8b949e")
        ax.set_ylabel("Count", color="#8b949e")
    plt.tight_layout()
    return fig


def plot_scatter_2d(df):
    fig, ax = dark_fig(figsize=(9, 5))
    sc = ax.scatter(df["energy_density"], df["electrochemical_stability"],
                    c=df["band_gap"], cmap="plasma", alpha=0.45, s=16, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Band Gap (eV)", color="#8b949e")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
    ax.set_xlabel("Energy Density (Wh/kg)")
    ax.set_ylabel("Electrochemical Stability (V)")
    ax.set_title("Energy Density vs Stability  (colored by Band Gap)")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔋 Battery Predictor")
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", ["📊 Dashboard", "🔮 Predict Material", "🏆 Top Candidates", "📈 Analytics"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**About**")
    st.caption("Predicts energy density (Wh/kg) and electrochemical stability window (V) of solid-state battery materials using Random Forest regression on Materials Project descriptors.")
    st.caption("Model trained on 1,200 synthetic samples with physics-inspired target generation.")

# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────

with st.spinner("Training models on first load…"):
    df_raw = generate_dataset()
    df     = feature_engineering(df_raw)
    models = train_models("fixed_seed_42")  # stable cache key

res_ed = models["energy_density"]
res_es = models["electrochemical_stability"]

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────

if page == "📊 Dashboard":
    st.markdown("# 🔋 Solid-State Battery Performance Predictor")
    st.markdown("*ML-powered screening of electrode & electrolyte materials using Materials Project descriptors*")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Energy Density R²", f"{res_ed['r2']:.3f}", ""),
        ("Stability R²", f"{res_es['r2']:.3f}", ""),
        ("ED RMSE", f"{res_ed['rmse']:.1f}", "Wh/kg"),
        ("ES RMSE", f"{res_es['rmse']:.3f}", "V"),
    ]
    for col, (label, val, unit) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
    st.pyplot(plot_actual_vs_predicted(res_ed, res_es))

    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", len(df))
        st.metric("Features (after engineering)", len(res_ed["feature_cols"]))
    with col2:
        st.metric("Train / Test Split", "80% / 20%")
        st.metric("CV Folds", "5")

    st.markdown("<div class='section-header'>Sample Data</div>", unsafe_allow_html=True)
    st.dataframe(df_raw.head(20).style.background_gradient(cmap="Blues", subset=["energy_density", "electrochemical_stability"]), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────

elif page == "🔮 Predict Material":
    st.markdown("# 🔮 Predict New Material")
    st.markdown("Adjust the material descriptors below to predict performance.")
    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("<div class='section-header'>Electronic / Thermodynamic</div>", unsafe_allow_html=True)
        band_gap         = st.slider("Band Gap (eV)", 0.0, 6.0, 2.5, 0.1)
        formation_energy = st.slider("Formation Energy (eV/atom)", -4.0, 0.5, -1.5, 0.05)
        e_above_hull     = st.slider("E Above Hull (eV/atom)", 0.0, 0.5, 0.03, 0.005)
        electronegativity = st.slider("Mean Electronegativity", 0.8, 3.5, 2.0, 0.05)
        oxidation_state  = st.slider("Avg Oxidation State", -3.0, 5.0, 1.5, 0.1)

    with col_r:
        st.markdown("<div class='section-header'>Structural / Mechanical</div>", unsafe_allow_html=True)
        density         = st.slider("Density (g/cm³)", 1.5, 8.0, 3.5, 0.1)
        volume          = st.slider("Unit Cell Volume (Å³)", 20.0, 500.0, 150.0, 5.0)
        nsites          = st.slider("Number of Sites", 2, 30, 8, 1)
        ionic_radius    = st.slider("Mean Ionic Radius (Å)", 0.3, 1.8, 0.9, 0.05)
        bulk_modulus    = st.slider("Bulk Modulus (GPa)", 10.0, 300.0, 120.0, 5.0)
        shear_modulus   = st.slider("Shear Modulus (GPa)", 5.0, 200.0, 70.0, 5.0)

    if st.button("⚡ Predict Performance", type="primary", use_container_width=True):
        input_base = pd.DataFrame([{
            "band_gap": band_gap, "formation_energy": formation_energy,
            "density": density, "volume": volume, "nsites": float(nsites),
            "ionic_radius_mean": ionic_radius, "electronegativity": electronegativity,
            "oxidation_state": oxidation_state, "bulk_modulus": bulk_modulus,
            "shear_modulus": shear_modulus, "e_above_hull": e_above_hull,
        }])
        # Feature engineering on single row
        inp = input_base.copy()
        inp["bulk_shear_ratio"]  = inp["bulk_modulus"] / (inp["shear_modulus"] + 1e-6)
        inp["vol_per_site"]      = inp["volume"] / inp["nsites"]
        inp["stability_flag"]    = int(e_above_hull < 0.1)
        inp["band_gap_sq"]       = inp["band_gap"] ** 2
        inp["formation_density"] = inp["formation_energy"] * inp["density"]

        feat_cols = res_ed["feature_cols"]
        inp = inp[feat_cols]

        pred_ed = res_ed["model"].predict(res_ed["scaler"].transform(inp))[0]
        pred_es = res_es["model"].predict(res_es["scaler"].transform(inp))[0]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="predict-result">
                <div class="result-title">⚡ Energy Density</div>
                <div class="result-val">{pred_ed:.1f} <span style="font-size:1rem;color:#8b949e">Wh/kg</span></div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="predict-result">
                <div class="result-title">🛡 Electrochemical Stability</div>
                <div class="result-val">{pred_es:.2f} <span style="font-size:1rem;color:#8b949e">V</span></div>
            </div>""", unsafe_allow_html=True)

        # Composite score
        df_ed_range = df["energy_density"]
        df_es_range = df["electrochemical_stability"]
        ed_norm  = (pred_ed - df_ed_range.min()) / (df_ed_range.max() - df_ed_range.min())
        es_norm  = (pred_es - df_es_range.min()) / (df_es_range.max() - df_es_range.min())
        stab_n   = max(0, 1 - e_above_hull / 0.5)
        score    = 0.5*ed_norm + 0.3*es_norm + 0.2*stab_n
        st.markdown(f"""
        <div style="margin-top:1rem; padding:0.8rem 1.2rem; background:#1c2128; border-radius:8px; border:1px solid #30363d;">
            <span style="font-family:'IBM Plex Mono',monospace; color:#8b949e; font-size:0.75rem;">COMPOSITE SCORE</span>
            <span style="font-family:'IBM Plex Mono',monospace; color:#f0f6fc; font-size:1.4rem; margin-left:1rem;">{score:.3f}</span>
            <span style="color:#8b949e; font-size:0.8rem;"> / 1.000</span>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: TOP CANDIDATES
# ─────────────────────────────────────────────

elif page == "🏆 Top Candidates":
    st.markdown("# 🏆 Top Candidate Materials")
    st.markdown("Ranked by composite score: **50% energy density + 30% stability + 20% thermodynamic stability**")
    st.markdown("---")

    n_top = st.slider("Number of candidates", 5, 50, 10, 5)
    top   = screen_top_materials(df, top_n=n_top)

    st.dataframe(
        top.style
           .background_gradient(cmap="YlGn", subset=["composite_score"])
           .background_gradient(cmap="Blues", subset=["energy_density"])
           .background_gradient(cmap="Greens", subset=["electrochemical_stability"])
           .format({"composite_score": "{:.3f}", "energy_density": "{:.1f}",
                    "electrochemical_stability": "{:.2f}", "e_above_hull": "{:.4f}",
                    "band_gap": "{:.2f}", "formation_energy": "{:.3f}", "density": "{:.2f}"}),
        use_container_width=True, height=420
    )

    csv = top.to_csv().encode("utf-8")
    st.download_button("⬇ Download CSV", csv, "top_candidates.csv", "text/csv", use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────

elif page == "📈 Analytics":
    st.markdown("# 📈 Analytics")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Distributions", "Correlation", "2D Scatter"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Energy Density**")
            st.pyplot(plot_feature_importance(res_ed["importances"], "Feature Importance — Energy Density"))
        with col2:
            st.markdown("**Electrochemical Stability**")
            st.pyplot(plot_feature_importance(res_es["importances"], "Feature Importance — Electrochemical Stability"))

    with tab2:
        st.pyplot(plot_distributions(df))

    with tab3:
        st.pyplot(plot_correlation_heatmap(df))

    with tab4:
        st.pyplot(plot_scatter_2d(df))
