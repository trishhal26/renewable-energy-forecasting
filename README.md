# 🌞💨 Multi-Source Renewable Energy Forecasting

Forecasting power output for **solar** and **wind** energy sources using machine learning — built as a single, comparative project rather than two disconnected notebooks.

> **Status: 🚧 In Progress** — Repository structure and project plan are set up. Full implementation begins after 20 Sept 2026 (post-exams). Watch this repo for updates!

---

## 📌 Project Overview

Most renewable-energy forecasting projects online tackle solar *or* wind in isolation. This project instead asks: **how differently should we approach forecasting for two fundamentally different energy sources?**

- **Solar power output** tends to follow strong daily/seasonal cyclic patterns (sunrise–sunset, season) — comparatively easier to model.
- **Wind power output** is noisier and non-linear (tied to a turbine's power curve, sudden gusts, direction changes) — a harder forecasting problem.

By building both pipelines with a **shared methodology** (same preprocessing philosophy, same evaluation approach) but source-specific modeling choices, the project aims to show:
1. Solid end-to-end ML pipeline skills (cleaning → feature engineering → modeling → evaluation).
2. The ability to reason about *why* one domain is harder to predict than another — not just fit a model and report accuracy.

---

## 🎯 Goals

- [ ] Build a solar power output forecasting model (regression)
- [ ] Build a wind power output forecasting model (regression)
- [ ] Shared utility functions for preprocessing/plotting across both
- [ ] Baseline models (Linear Regression) → stronger models (Random Forest / XGBoost)
- [ ] Optional: LSTM/time-series model for one source to show sequence modeling
- [ ] Comparative analysis: predictability of solar vs. wind, feature importance, error analysis
- [ ] Clean visualizations: predicted vs. actual, feature importance, residual plots
- [ ] (Stretch) Simple Streamlit demo for interactive forecasting

---

## 🗂️ Repository Structure

```
renewable-energy-forecasting/
├── data/
│   ├── raw/              # Original downloaded datasets (not committed if large — see .gitignore)
│   └── processed/        # Cleaned/feature-engineered datasets
├── notebooks/            # Exploratory analysis & modeling notebooks
├── solar/                # Solar forecasting pipeline (scripts/modules)
├── wind/                 # Wind forecasting pipeline (scripts/modules)
├── utils/                # Shared preprocessing, evaluation, and plotting functions
├── models/                # Saved trained models
├── images/               # Plots and visualizations used in README/reports
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 📊 Planned Datasets

| Source | Dataset | Link |
|---|---|---|
| Wind | Wind Turbine SCADA Dataset | Kaggle |
| Wind | Wind Toolkit | NREL |
| Wind/Solar | Grid generation data | ENTSO-E |
| Solar | National Solar Radiation Database (NSRDB) | NREL |

*(Final dataset selection to be confirmed during implementation phase.)*

---

## 🛠️ Planned Tech Stack

- **Language:** Python
- **Core libraries:** pandas, NumPy, scikit-learn
- **Modeling:** XGBoost / Random Forest, optionally LSTM (PyTorch/Keras)
- **Visualization:** matplotlib, seaborn
- **Explainability:** SHAP (feature importance)
- **(Stretch) Deployment:** Streamlit

---

## 🗓️ Timeline

| Phase | Status |
|---|---|
| Project scoping & repo setup | ✅ Done |
| Data collection & EDA | ⏳ Starting after 20 Sept 2026 |
| Solar forecasting pipeline | ⏳ Planned |
| Wind forecasting pipeline | ⏳ Planned |
| Comparative analysis & writeup | ⏳ Planned |
| (Stretch) Streamlit demo | ⏳ Planned |

---

## 👤 About

Built by Trisha Haldar as a portfolio project while completing an AI/ML certificate program (IIT Patna x Masai School). Part of a broader set of projects demonstrating applied ML, deep learning, and LLM/RAG skills — see other pinned repositories on this profile.

---

*This README will be updated with results, plots, and a full write-up as the project progresses.*
