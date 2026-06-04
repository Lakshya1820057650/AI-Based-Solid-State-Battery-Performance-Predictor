# 🔋 AI-Based Solid State Battery Performance Predictor

Predicts **energy density (Wh/kg)** and **electrochemical stability window (V)** of solid-state battery materials using Random Forest regression on Materials Project descriptors.

## Features
- 📊 **Dashboard** — Model metrics, actual vs predicted plots, sample data
- 🔮 **Predict** — Interactive sliders to predict a custom material's performance
- 🏆 **Top Candidates** — Composite-scored material ranking with CSV download
- 📈 **Analytics** — Feature importance, distributions, correlation heatmap, 2D scatter

---

## 🚀 Deploy to Render (Free, No Credit Card)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
# Create a new repo on github.com, then:
git remote add origin https://github.com/<your-username>/battery-predictor.git
git push -u origin main
```

### Step 2 — Deploy on Render
1. Go to **[render.com](https://render.com)** → Sign up free (use GitHub login)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account → select your repo
4. Render auto-detects the `Dockerfile` — no manual config needed
5. Set **Plan** to **Free**
6. Click **"Create Web Service"** — live in ~3 minutes ✅

Your app URL will be: `https://battery-performance-predictor.onrender.com`

> **Note:** Free tier spins down after 15 min of inactivity; first visit after idle takes ~30s to wake up.

---

## 🖥 Run Locally with Docker
```bash
docker build -t battery-predictor .
docker run -p 8501:8501 battery-predictor
# Open http://localhost:8501
```

## 🖥 Run Locally without Docker

---

```bash
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

---

## Project Structure
```
battery-predictor/
├── app.py            # Main Streamlit app
├── requirements.txt  # Python dependencies
└── README.md
```

## Model Details
| Target | R² | CV R² | RMSE |
|--------|-----|-------|------|
| Energy Density (Wh/kg) | ~0.87 | ~0.86 | ~18 Wh/kg |
| Electrochemical Stability (V) | ~0.86 | ~0.85 | ~0.16 V |

- **Algorithm**: Random Forest (200 trees, max_depth=12)
- **Features**: 16 (11 base + 5 engineered)
- **Training data**: 1,200 physics-inspired synthetic samples
