# ⚡ Renewable Energy Forecasting & Asset Health Check

A machine learning project on real solar and wind data, built around two questions an energy operator asks every day:

1. **Wind — Forecasting:** *How much power will the turbine produce in the next hour?*
2. **Solar — Health Check:** *Are all the inverters producing what they should, and if not, how much energy are we losing?*

Each dataset is used for what it is best suited to. The wind dataset has a full year of history, which makes it right for time-series forecasting and deep learning. The solar dataset has ~22 inverters working under the same weather, which makes it right for spotting underperforming equipment.

> **Status: 🔄 In Progress** — Data collection and exploratory data analysis complete. Feature engineering and modeling underway.

---

## 🧭 Project at a Glance

| | 💨 Wind | ☀️ Solar |
|---|---|---|
| **Question** | How much power will we produce next? | Which inverters are underperforming, and what does it cost? |
| **Task** | Time-series forecasting | Underperformance (anomaly) detection |
| **Models** | Naive baseline → XGBoost → LSTM | Expected-output regression → residual analysis |
| **Output** | Next-hour power forecast + model comparison | Inverter ranking + lost energy (kWh) + estimated cost (₹) |
| **Why this dataset** | Full year of 10-min data (~50K rows): enough history for forecasting and for training an LSTM | ~22 inverters under identical weather: weak inverters stand out against their neighbours |

---

## 💨 Part 1 — Wind Power Forecasting

### What we are doing
Forecasting the turbine's power output for the **next hour**, using only information available at the time of prediction.

- **Features:** past power output (lags), past wind speed, wind direction, rolling averages, and time features (hour, day, month)
- **Models, from simplest to most complex:**
  1. **Naive baseline** — "next hour will look like the last hour." Every model must beat this to be useful.
  2. **XGBoost** — a tree-based model trained on lag and time features.
  3. **LSTM (PyTorch)** — a deep learning model that learns directly from sequences of past observations.
- **Evaluation:** time-based train/test split (train on earlier months, test on later months), compared using MAE and RMSE.

### Why
- **Naive baseline first:** a forecast only has value if it beats the simplest possible guess.
- **XGBoost vs. LSTM:** tests whether a deep learning model's extra complexity actually pays off over a strong tree-based model on this data. Whichever wins, the comparison is the result.
- **Time-based split:** this is time-series data. A random split would let the model see the future during training and overstate its accuracy.

### What we are not doing, and why
- **No weather forecast inputs:** real forecasting systems use predicted wind speed, but historical forecasts for this turbine's location and dates aren't available. The model forecasts from past observations only.
- **No long-range forecasting:** accuracy from past observations alone drops quickly beyond short horizons, so the focus stays on the next hour.

---

## ☀️ Part 2 — Solar Asset Health Check

### What we are doing
Finding inverters that produce less than they should, and estimating what that shortfall costs.

1. **Learn expected output:** a regression model predicts how much AC power an inverter *should* produce from irradiation, ambient temperature, module temperature, and time of day.
2. **Compare expected vs. actual:** the gap between the two (the residual) is the signal. A small gap means normal operation; a large, persistent shortfall points to a possible fault, dirty panels, or downtime.
3. **Rank and quantify:** rank inverters by how consistently they underproduce compared with others under identical weather, and convert the shortfall into **lost energy (kWh)** and **estimated cost (₹)** using a stated tariff assumption.

### Why
- **Same weather, many inverters:** all inverters in a plant share one weather sensor, so an inverter that keeps lagging behind its neighbours is a strong, fair signal of a problem.
- **Residual-based detection:** simple, explainable, and widely used in industrial monitoring. Every flag traces back to "the model expected X, the inverter produced Y".
- **Ending at a cost figure:** turns a model output into a maintenance decision: *which inverter to check first*.

### What we are not doing, and why
- **No solar forecasting:** the solar dataset covers only ~34 days. That's too short to forecast reliably or to train an LSTM without overfitting, so the data is used where it's strongest: comparing inverters.

---

## 📈 Key Findings So Far (EDA)

- **Wind:** clean data with no missing values. Output is noisy with no daily pattern. Wind speed vs. power follows the expected S-shaped curve: near zero at low speeds, a steep rise, then a flat plateau at rated capacity. A few small negative power values appear, consistent with the turbine drawing grid power while idling; these are handled during preprocessing.
- **Solar:** clean data with no missing values. Output follows a clear daily cycle: zero at night, peaking around midday.

---

## ⚠️ Known Limitations

- **Forecasts use past observations only** (no weather forecasts), so they set a realistic baseline rather than production-grade accuracy.
- **Short solar history (~34 days)** limits the health check to that period and rules out seasonal analysis.
- **Different locations:** the wind (Turkey) and solar (India) datasets are analysed independently, not combined.
- **Cost estimates** depend on an assumed electricity tariff and are indicative, not exact.

---

## 📊 Datasets

| Part | Dataset | Details |
|---|---|---|
| Wind | [Wind Turbine SCADA Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset) (Kaggle) | 1 turbine in Turkey, full year 2018, 10-min intervals. Active power, wind speed, wind direction, theoretical power curve |
| Solar | [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) (Kaggle) | 2 plants in India, ~34 days, 15-min intervals. Inverter-level generation + plant-level weather sensor data |

*Raw data is not committed to this repo (see `.gitignore`). Download it from the links above into `data/raw/wind/` and `data/raw/solar/`.*

---

## 🗂️ Repository Structure

```
renewable-energy-forecasting/
├── data/
│   ├── raw/
│   │   ├── solar/        # Plant 1 & 2 generation + weather CSVs (not committed)
│   │   └── wind/         # Turbine SCADA CSV (not committed)
│   └── processed/        # Cleaned / feature-engineered data
├── notebooks/
│   ├── 01_solar_eda.ipynb
│   └── 02_wind_eda.ipynb
├── solar/                # Solar health-check pipeline
├── wind/                 # Wind forecasting pipeline
├── utils/                # Shared preprocessing, evaluation, plotting
├── models/               # Saved trained models
├── images/               # Plots used in this README
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- **Python** · pandas · NumPy · scikit-learn · XGBoost
- **Deep learning:** PyTorch (LSTM)
- **Visualization:** matplotlib, seaborn
- **Explainability:** SHAP
- **Deployment:** Streamlit

---

## 🗓️ Roadmap

| Phase | Status |
|---|---|
| Project scoping & repo setup | ✅ Done |
| Data collection & EDA (wind + solar) | ✅ Done |
| Wind: feature engineering (lags, rolling, time features) | 🔄 In progress |
| Wind: naive baseline + XGBoost forecasting | ⏳ Planned |
| Wind: LSTM forecasting + comparison with XGBoost | ⏳ Planned |
| Solar: expected-output model + inverter ranking | ⏳ Planned |
| Solar: lost energy & cost estimation | ⏳ Planned |
| Streamlit dashboard (Wind Forecast tab + Solar Health tab) | ⏳ Planned |
| **(Stretch)** Wind: day-ahead forecasting | ⏳ Planned |
| **(Stretch)** Wind: energy loss vs. theoretical power curve | ⏳ Planned |
| **(Stretch)** Prediction intervals via quantile regression | ⏳ Planned |

---

## 👤 About

Built by **Trisha Haldar** as a portfolio project during an AI/ML certificate program (IIT Patna × Masai School).

*This README is updated as the project progresses.*