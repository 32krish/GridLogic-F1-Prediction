# 🏎️ GridLogic — F1 Race Prediction & Simulation System

<div align="center">

![GridLogic Banner](screenshots/banner.png)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)](https://xgboost.readthedocs.io)
[![FastF1](https://img.shields.io/badge/FastF1-3.5+-E10600?style=for-the-badge)](https://github.com/theOehrly/Fast-F1)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=for-the-badge)](https://matplotlib.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**A complete end-to-end Formula 1 race prediction system built with Python.**  
Real telemetry data · XGBoost ML model · Live race simulation · Real-time animation

[🚀 Quick Start](#-quick-start) · [📊 Results](#-model-results) · [🗂️ Structure](#%EF%B8%8F-project-structure) · [🎮 Demo](#-visualization-demo)

</div>

---

## 📌 About

**GridLogic** is a 3rd-year Predictive Analysis project at **Ganpat University** (B.Tech CSE, 2024–25) that applies machine learning to predict Formula 1 race outcomes using real official telemetry data.

The system fetches lap-level data from the [FastF1](https://github.com/theOehrly/Fast-F1) API, engineers 12 predictive features from sector times and tyre telemetry, trains an **XGBoost regression model** to predict individual lap times, and runs a deterministic lap-by-lap race simulation. The result is rendered as a **real-time matplotlib animation** at 45× speed.

### ✅ Key Results — 2022 Italian Grand Prix (Monza)

| Metric | Result |
|--------|--------|
| **MAE (lap time prediction)** | **0.274 seconds** |
| **Top-3 Accuracy** | **100%** ✅ |
| **Within ±2 positions** | **75%** |
| **Spearman Correlation** | **0.87** |
| **Training dataset** | 4,651 clean laps (2019–2023) |

> 🏆 Predicted podium: **LEC → VER → RUS**  |  Actual podium: **VER → LEC → RUS**

---

## 🎮 Visualization Demo

```
============================================================
  F1 RACE VISUALIZER - GRAND PRIX CIRCUIT
============================================================
  Cars move along circuit · Live leaderboard · DRS zones
  CONTROLS: UP/DOWN = Speed  |  R = Reset  |  ESC = Quit
============================================================
```
---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Internet connection (first run only — for FastF1 data download)
- 5 GB free disk space (FastF1 cache)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/GridLogic-F1-Prediction.git
cd GridLogic-F1-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py
```

You will be prompted to enter:
```
Enter race year (2019–2026): 2022
Enter track name (e.g. Bahrain, Monza, Silverstone): Monza
```

The system then automatically:
1. Fetches multi-year lap data from FastF1
2. Engineers 12 features from telemetry
3. Trains XGBoost with GridSearchCV tuning
4. Simulates the full race lap-by-lap
5. Prints final standings + accuracy report
6. Launches real-time matplotlib animation

### Run Standalone Visualization (No ML required)

```bash
python f1_visualizer.py
```

---

## 🗂️ Project Structure

```
GridLogic/
│
├── main.py                     ← Single entry point — run this
├── f1_visualizer.py            ← Standalone matplotlib visualization demo
├── requirements.txt
│
├── data/
│   └── cache/                  ← FastF1 parquet cache (auto-created, gitignored)
│
└── src/
    ├── __init__.py
    ├── features.py             ← Feature engineering (12 features from telemetry)
    ├── model.py                ← XGBoost training with GridSearchCV
    ├── simulation.py           ← Lap-by-lap race simulation engine
    │
    └── visualization/
        ├── __init__.py
        ├── f1_visualizer.py    ← Pipeline-connected Matplotlib animator
        └── track.py            ← FastF1 track coordinate loader
```

---

## ⚙️ Pipeline Architecture

```
FastF1 API  →  features.py  →  model.py  →  simulation.py  →  f1_visualizer.py
   ↓               ↓              ↓               ↓                  ↓
 Raw laps      X, y matrix    XGBRegressor    Standings CSV     Animation window
```

**Each module is independently runnable** — you can test any stage in isolation without re-running the full pipeline.

---

## 📊 Feature Engineering

12 features are extracted from raw FastF1 lap telemetry:

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `sector1_time` | Sector1Time | Sector 1 elapsed time (seconds) |
| 2 | `sector2_time` | Sector2Time | Sector 2 elapsed time (seconds) |
| 3 | `sector3_time` | Sector3Time | Sector 3 elapsed time (seconds) |
| 4 | `tyre_age` | TyreLife | Laps on current tyre set |
| 5 | `compound_code` | Compound | SOFT=1, MEDIUM=2, HARD=3, INT=4, WET=5 |
| 6 | `lap_number` | LapNumber | Current race lap |
| 7 | `fuel_load_est` | LapNumber | 1 − (lap/max_lap) — fuel weight proxy |
| 8 | `avg_lap` | LapTime | Driver's session average lap time |
| 9 | `best_lap` | LapTime | Driver's personal best lap time |
| 10 | `driver_skill` | Historical | Normalized driver skill rating |
| 11 | `team_strength` | Historical | Normalized constructor performance |
| 12 | `qual_pos` | GridPosition | Qualifying/grid starting position |

---

## 🤖 Model Details

| Parameter | Value |
|-----------|-------|
| **Algorithm** | XGBoost Regressor |
| **Tuning** | GridSearchCV — 3-fold cross-validation |
| **Best params** | `n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8` |
| **Train / Test split** | 80% / 20% (random_state=42) |
| **MAE** | **0.274 seconds** |
| **Training data** | 2019–2023 (5 seasons, 4,651 clean laps) |

### Feature Importance (2022 Italian GP)

```
sector1_time    ████████████████████████████████  65.06%
sector3_time    ███████                           14.36%
sector2_time    ███████                           13.96%
tyre_age        █                                  2.04%
lap_number      █                                  1.78%
fuel_load_est   ▌                                  0.95%
...
```

---

## 📈 Sample Output

```
Final predicted standings:
------------------------------------
  POS   DRIVER     TOTAL TIME (s)
------------------------------------
  1     LEC        5518.925
  2     VER        5522.626
  3     RUS        5566.515
  4     PER        5566.732
  5     SAI        5569.914
  ...
  20    MAG        5650.942
------------------------------------

========================================
   MODEL ACCURACY REPORT
========================================
Position Accuracy     : 15.00%
Spearman Correlation  : 0.87
Top-3 Accuracy        : 100.00%
Within ±2 Accuracy    : 75.00%

Top 3 Comparison:
Predicted: ['LEC', 'VER', 'RUS']
Actual   : ['VER', 'LEC', 'RUS']
========================================
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| [FastF1](https://github.com/theOehrly/Fast-F1) | ≥3.5.0 | Official F1 lap data and telemetry API |
| [XGBoost](https://xgboost.readthedocs.io) | ≥2.0.0 | Gradient boosting regression model |
| [scikit-learn](https://scikit-learn.org) | ≥1.3.0 | GridSearchCV, train/test split, metrics |
| [pandas](https://pandas.pydata.org) | ≥2.0.0 | DataFrame operations and feature engineering |
| [numpy](https://numpy.org) | ≥1.24.0 | Numerical array operations |
| [matplotlib](https://matplotlib.org) | ≥3.7.0 | FuncAnimation real-time visualization |

---

## 🎮 Visualization Controls

| Key | Action |
|-----|--------|
| `↑` Arrow | Increase simulation speed (×1.5 per press) |
| `↓` Arrow | Decrease simulation speed (÷1.5 per press) |
| `R` | Reset race to lap 1 |
| `ESC` / Close | Quit visualization |

Default speed: **45× real-time** (57-lap race completes in ~2 minutes)

---

## 📋 Requirements

```txt
fastf1>=3.5.0
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- [ ] Weather-adjusted lap time predictions using FastF1 weather API
- [ ] Safety car and VSC period simulation
- [ ] Real GPS track coordinates from `lap.get_pos_data()`
- [ ] Multi-stop pit strategy optimization
- [ ] Streamlit web dashboard deployment
- [ ] LSTM sequence model comparison

---

## 📄 Academic Context

| Field | Detail |
|-------|--------|
| **University** | Ganpat University, Mehsana, Gujarat |
| **Programme** | B.Tech Computer Science & Engineering |
| **Year** | 3rd Year — Semester VI |
| **Subject** | Predictive Analysis (2CSE-608) |
| **Academic Year** | 2025–26 |

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.  
See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Tim Volk](https://github.com/theOehrly) — FastF1 library developer
- [Chen & Guestrin (2016)](https://arxiv.org/abs/1603.02754) — XGBoost paper
- [Heilmeier et al. (2020)](https://www.tandfonline.com/doi/abs/10.1080/00423114.2019.1631455) — Lap time simulation research
- Formula One Management Ltd. — Official F1 timing data

---

<div align="center">

⭐ Star this repo if you found it useful!

</div>
