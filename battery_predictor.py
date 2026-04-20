"""
AI-Based Solid State Battery Performance Predictor

Predicts energy density and electrochemical stability of solid-state battery
materials using material descriptors from the Materials Project database.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

import os

# ─────────────────────────────────────────────
# 1. DATA GENERATION (simulates Materials Project descriptors)
# ─────────────────────────────────────────────

def generate_dataset(n_samples=1200, random_state=42):
    """
    Generates a synthetic dataset mimicking Materials Project descriptors
    for solid-state battery electrode/electrolyte materials.

    Features (material descriptors):
        - band_gap           : Electronic band gap (eV)
        - formation_energy   : Formation energy per atom (eV/atom)
        - density            : Crystal density (g/cm³)
        - volume             : Unit cell volume (Å³)
        - nsites             : Number of sites in unit cell
        - ionic_radius_mean  : Mean ionic radius of constituent elements (Å)
        - electronegativity  : Mean Pauling electronegativity
        - oxidation_state    : Average oxidation state
        - bulk_modulus       : Bulk modulus (GPa)
        - shear_modulus      : Shear modulus (GPa)
        - e_above_hull       : Energy above convex hull (eV/atom) — stability proxy

    Targets:
        - energy_density         : Theoretical energy density (Wh/kg)
        - electrochemical_stability : Electrochemical stability window (V)
    """
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

    # Physics-inspired target generation with noise
    energy_density = (
        250
        - 30  * e_above_hull
        - 15  * band_gap
        + 20  * np.abs(formation_energy)
        + 10  * bulk_modulus / 100
        - 8   * density
        + 5   * oxidation_state
        + np.random.normal(0, 15, n)
    )
    energy_density = np.clip(energy_density, 80, 500)

    electrochemical_stability = (
        3.0
        + 0.4  * band_gap
        - 0.3  * e_above_hull
        + 0.2  * electronegativity
        - 0.1  * np.abs(formation_energy)
        + np.random.normal(0, 0.3, n)
    )
    electrochemical_stability = np.clip(electrochemical_stability, 0.5, 7.0)

    df = pd.DataFrame({
        "band_gap"              : band_gap,
        "formation_energy"      : formation_energy,
        "density"               : density,
        "volume"                : volume,
        "nsites"                : nsites,
        "ionic_radius_mean"     : ionic_radius_mean,
        "electronegativity"     : electronegativity,
        "oxidation_state"       : oxidation_state,
        "bulk_modulus"          : bulk_modulus,
        "shear_modulus"         : shear_modulus,
        "e_above_hull"          : e_above_hull,
        "energy_density"        : energy_density,
        "electrochemical_stability": electrochemical_stability,
    })
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df):
    """Add derived features to improve model performance."""
    df = df.copy()
    df["bulk_shear_ratio"]      = df["bulk_modulus"] / (df["shear_modulus"] + 1e-6)
    df["vol_per_site"]          = df["volume"] / df["nsites"]
    df["stability_flag"]        = (df["e_above_hull"] < 0.1).astype(int)
    df["band_gap_sq"]           = df["band_gap"] ** 2
    df["formation_density"]     = df["formation_energy"] * df["density"]
    return df


# ─────────────────────────────────────────────
# 3. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(df, target_col, label):
    feature_cols = [c for c in df.columns if c not in
                    ["energy_density", "electrochemical_stability"]]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train_s, y_train, cv=5,
                           scoring="r2").mean()

    print(f"\n{'='*50}")
    print(f"  Target: {label}")
    print(f"{'='*50}")
    print(f"  R²  Score  : {r2:.4f}")
    print(f"  CV  R²     : {cv:.4f}")
    print(f"  RMSE       : {rmse:.4f}")
    print(f"  MAE        : {mae:.4f}")

    return model, scaler, X_test_s, y_test, y_pred, feature_cols


# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────

def save_plots(df, model_ed, scaler_ed, X_test_ed, y_test_ed, y_pred_ed,
               model_es, scaler_es, X_test_es, y_test_es, y_pred_es,
               feature_cols, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # ── (A) Correlation heatmap ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.4, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_heatmap.png", dpi=150)
    plt.close()
    print("  Saved: correlation_heatmap.png")

    # ── (B) Actual vs Predicted — Energy Density ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, y_t, y_p, lbl, unit in zip(
            axes,
            [y_test_ed, y_test_es],
            [y_pred_ed, y_pred_es],
            ["Energy Density", "Electrochemical Stability"],
            ["Wh/kg", "V"]):
        ax.scatter(y_t, y_p, alpha=0.4, edgecolors="k", linewidths=0.3, s=25)
        mn, mx = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal fit")
        ax.set_xlabel(f"Actual ({unit})", fontsize=11)
        ax.set_ylabel(f"Predicted ({unit})", fontsize=11)
        ax.set_title(f"Actual vs Predicted — {lbl}", fontsize=12)
        ax.legend()
        r2 = r2_score(y_t, y_p)
        ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
                fontsize=10, color="navy")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/actual_vs_predicted.png", dpi=150)
    plt.close()
    print("  Saved: actual_vs_predicted.png")

    # ── (C) Feature Importance — Energy Density ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    importances = pd.Series(model_ed.feature_importances_, index=feature_cols)
    importances.sort_values().plot.barh(ax=ax, color="steelblue", edgecolor="k",
                                        linewidth=0.5)
    ax.set_title("Feature Importances — Energy Density (Random Forest)", fontsize=13)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance_energy_density.png", dpi=150)
    plt.close()
    print("  Saved: feature_importance_energy_density.png")

    # ── (D) Distribution of targets ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, lbl, color in zip(
            axes,
            ["energy_density", "electrochemical_stability"],
            ["Energy Density (Wh/kg)", "Electrochemical Stability (V)"],
            ["royalblue", "coral"]):
        sns.histplot(df[col], bins=40, kde=True, ax=ax, color=color)
        ax.set_title(f"Distribution of {lbl}", fontsize=12)
        ax.set_xlabel(lbl)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/target_distributions.png", dpi=150)
    plt.close()
    print("  Saved: target_distributions.png")

    # ── (E) Stability vs Energy Density (scatter) ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(df["energy_density"], df["electrochemical_stability"],
                    c=df["band_gap"], cmap="viridis", alpha=0.5, s=20,
                    edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Band Gap (eV)")
    ax.set_xlabel("Energy Density (Wh/kg)")
    ax.set_ylabel("Electrochemical Stability (V)")
    ax.set_title("Energy Density vs Electrochemical Stability\n(colored by Band Gap)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/energy_vs_stability.png", dpi=150)
    plt.close()
    print("  Saved: energy_vs_stability.png")

    print(f"\n  All plots saved to '{out_dir}/' folder.")


# ─────────────────────────────────────────────
# 5. MATERIAL SCREENER
# ─────────────────────────────────────────────

def screen_top_materials(df, top_n=10):
    """
    Rank materials by a composite score:
    high energy density + wide stability window + low e_above_hull
    """
    df = df.copy()
    df["ed_norm"]  = (df["energy_density"] - df["energy_density"].min()) / \
                     (df["energy_density"].max() - df["energy_density"].min())
    df["es_norm"]  = (df["electrochemical_stability"] - df["electrochemical_stability"].min()) / \
                     (df["electrochemical_stability"].max() - df["electrochemical_stability"].min())
    df["stab_norm"] = 1 - (df["e_above_hull"] - df["e_above_hull"].min()) / \
                          (df["e_above_hull"].max() - df["e_above_hull"].min())

    df["composite_score"] = 0.5 * df["ed_norm"] + \
                            0.3 * df["es_norm"] + \
                            0.2 * df["stab_norm"]

    top = df.nlargest(top_n, "composite_score")[
        ["band_gap", "formation_energy", "density",
         "e_above_hull", "energy_density",
         "electrochemical_stability", "composite_score"]
    ].reset_index(drop=True)
    top.index += 1
    return top


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI-Based Solid State Battery Performance Predictor")
    print("="*60)

    # Step 1 — Generate data
    print("\n[1] Generating dataset from Materials Project descriptors...")
    df_raw = generate_dataset(n_samples=1200)
    print(f"    Dataset shape : {df_raw.shape}")
    print(f"    Columns       : {list(df_raw.columns)}")

    # Step 2 — Feature engineering
    print("\n[2] Applying feature engineering...")
    df = feature_engineering(df_raw)
    print(f"    New shape     : {df.shape}")

    # Step 3 — Save dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/battery_materials_dataset.csv", index=False)
    print("    Saved: data/battery_materials_dataset.csv")

    # Step 4 — Train models
    print("\n[3] Training Random Forest models...")
    model_ed, scaler_ed, X_test_ed, y_test_ed, y_pred_ed, feat_cols = \
        train_and_evaluate(df, "energy_density", "Energy Density (Wh/kg)")

    model_es, scaler_es, X_test_es, y_test_es, y_pred_es, _ = \
        train_and_evaluate(df, "electrochemical_stability",
                           "Electrochemical Stability (V)")

    # Step 5 — Plots
    print("\n[4] Generating visualizations...")
    save_plots(df, model_ed, scaler_ed, X_test_ed, y_test_ed, y_pred_ed,
               model_es, scaler_es, X_test_es, y_test_es, y_pred_es, feat_cols)

    # Step 6 — Top materials
    print("\n[5] Top 10 candidate materials by composite score:")
    top_materials = screen_top_materials(df, top_n=10)
    print(top_materials.to_string())
    top_materials.to_csv("data/top_candidate_materials.csv")
    print("\n    Saved: data/top_candidate_materials.csv")

    print("\n" + "="*60)
    print("  Pipeline complete.")
    print("="*60 + "\n")
