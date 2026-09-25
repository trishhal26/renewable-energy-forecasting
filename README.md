# 🌞💨 Multi-Source Renewable Energy Forecasting

Forecasting power output for **solar** and **wind** energy sources using machine learning — built as a single, comparative project rather than two disconnected notebooks.

## 📌 Project Overview

This project predicts **solar and wind power output** from weather and operating conditions using classical ML — a "pure machine learning" project (regression, feature engineering, model evaluation) without any deep learning or LLM components.

- **Core objective:** Given current weather/sensor readings (irradiation, temperature, wind speed, etc.), predict power output — for both a solar plant and a wind turbine — and compare which is easier to predict and why.
- **Stretch goal:** Extend the solar and/or wind model into a genuine *forecast* (predicting tomorrow's output from today's data) using lag-based time-series features, rather than same-timestamp prediction.

## 🎯 Goals

- [ ] Solar power output prediction (regression)
- [ ] Wind power output prediction (regression)
- [ ] Shared utility functions for preprocessing/plotting
- [ ] Baseline models (Linear Regression) → stronger models (Random Forest / XGBoost)
- [ ] Comparative analysis: predictability of solar vs. wind, feature importance, error analysis
- [ ] Clean visualizations: predicted vs. actual, feature importance, residual plots
- [ ] **(Stretch)** Lag-based forecasting: predict next-day output from historical patterns
- [ ] **(Stretch)** Simple Streamlit demo

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

## 📊 Datasets Used

| Source | Dataset | Link |
|---|---|---|
| Solar | Solar Power Generation Data (2 plants, generation + weather sensor readings) | Kaggle |
| Wind | Wind Turbine SCADA Dataset | Kaggle |

*Note: the solar dataset is from plants in India; the wind dataset is from a turbine in Turkey. They're used independently for a same-timestamp output-prediction task, not combined into one location-specific model.*
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
| Data collection & EDA | 🔄 In progress |
| Solar power prediction pipeline | ⏳ Planned |
| Wind power prediction pipeline | ⏳ Planned |
| Comparative analysis & writeup | ⏳ Planned |
| (Stretch) Lag-based forecasting | ⏳ Planned |
| (Stretch) Streamlit demo | ⏳ Planned |

---

## 👤 About

Built by Trisha Haldar as a portfolio project while completing an AI/ML certificate program (IIT Patna x Masai School). Part of a broader set of projects demonstrating applied ML, deep learning, and LLM/RAG skills — see other pinned repositories on this profile.

---

*This README will be updated with results, plots, and a full write-up as the project progresses.*
