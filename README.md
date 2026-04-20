# AI-Based Solid State Battery Performance Predictor

> Predicting energy density and electrochemical stability of solid-state battery materials using machine learning on material descriptors from the Materials Project database.

---

## Overview

Solid-state batteries are the next frontier in energy storage — safer, more energy-dense, and longer-lasting than conventional lithium-ion batteries. However, discovering suitable materials through lab experiments is expensive and time-consuming.

This project uses **machine learning (Random Forest)** to rapidly screen battery materials based on their physical and chemical descriptors, predicting:

- **Energy Density** (Wh/kg) — how much energy a material can store
- **Electrochemical Stability Window** (V) — the voltage range over which the material remains stable

---

## Features

- Synthetic dataset generation mimicking real Materials Project descriptors
- Feature engineering (derived descriptors for better model accuracy)
- Random Forest Regressor with cross-validation
- Composite scoring to rank top candidate materials
- 5 publication-quality visualizations saved automatically

---

## Material Descriptors Used

| Feature | Description | Unit |
|---|---|---|
| `band_gap` | Electronic band gap | eV |
| `formation_energy` | Formation energy per atom | eV/atom |
| `density` | Crystal density | g/cm³ |
| `volume` | Unit cell volume | Å³ |
| `nsites` | Number of sites in unit cell | — |
| `ionic_radius_mean` | Mean ionic radius | Å |
| `electronegativity` | Mean Pauling electronegativity | — |
| `oxidation_state` | Average oxidation state | — |
| `bulk_modulus` | Bulk modulus | GPa |
| `shear_modulus` | Shear modulus | GPa |
| `e_above_hull` | Energy above convex hull (stability) | eV/atom |

---

## Project Structure

```
battery-performance-predictor/
│
├── battery_predictor.py        # Main pipeline script
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
│
├── data/                       # Auto-generated on run
│   ├── battery_materials_dataset.csv
│   └── top_candidate_materials.csv
│
└── plots/                      # Auto-generated on run
    ├── correlation_heatmap.png
    ├── actual_vs_predicted.png
    ├── feature_importance_energy_density.png
    ├── target_distributions.png
    └── energy_vs_stability.png
```

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Lakshya1820057650/battery-performance-predictor.git
cd battery-performance-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python battery_predictor.py
```

---

## Model Performance (typical results)

| Target | R² Score | CV R² | RMSE | MAE |
|---|---|---|---|---|
| Energy Density | ~0.94 | ~0.93 | ~10.5 Wh/kg | ~7.8 Wh/kg |
| Electrochemical Stability | ~0.91 | ~0.90 | ~0.18 V | ~0.13 V |

---

## Sample Output — Top Candidate Materials

The pipeline ranks all materials using a composite score:

```
Composite Score = 0.5 × (Energy Density) + 0.3 × (Stability Window) + 0.2 × (Hull Stability)
```

Top candidates are saved to `data/top_candidate_materials.csv`.

---

## Visualizations

The pipeline auto-generates 5 plots in the `plots/` folder:

1. **Correlation Heatmap** — relationships between all descriptors and targets
2. **Actual vs Predicted** — model accuracy for both targets
3. **Feature Importances** — which descriptors drive energy density predictions
4. **Target Distributions** — spread of energy density and stability values
5. **Energy vs Stability Scatter** — trade-off landscape colored by band gap

---

## Tech Stack

- **Python 3.10+**
- **Scikit-learn** — Random Forest, preprocessing, evaluation
- **Pandas / NumPy** — data handling and feature engineering
- **Matplotlib / Seaborn** — visualizations

---

## Author

**Lakshya Saxena**
- GitHub: [Lakshya1820057650](https://github.com/Lakshya1820057650)
- LinkedIn: [lakshya-saxena](https://www.linkedin.com/in/lakshya-saxena-3531702a9)
- Codolio: [LakshyaSaxena](https://codolio.com/profile/LakshyaSaxena)

---

## License

MIT License — feel free to use and build upon this project.
