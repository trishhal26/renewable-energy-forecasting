"""
Feature pipeline for 1-hour-ahead wind power forecasting.

Turns raw 10-minute turbine readings into exactly the features the XGBoost model was
trained on (notebooks 03 and 04). Used by the dashboard's replay mode, and checked
against the training features in notebook 07.
"""

import numpy as np
import pandas as pd

HORIZON_STEPS = 6     # 6 steps x 10 min = forecast 1 hour ahead
HISTORY_STEPS = 144   # the longest lag is 24 hours (144 steps), so that much history is needed


def build_features(readings):
    """
    Build model features from raw readings.

    readings: DataFrame indexed by timestamp (10-minute steps) with columns
        'power' (kW, negative values already clipped to 0),
        'wind_speed' (m/s), and 'wind_direction' (degrees).

    Returns one row of features per timestamp. Rows without enough history,
    or affected by a data gap, contain NaN and must not be used for a forecast.
    """
    readings = readings.asfreq("10min")   # guarantee evenly spaced rows, so shifts mean exact time lags
    power = readings["power"]
    wind_speed = readings["wind_speed"]

    f = pd.DataFrame(index=readings.index)
    f["power"] = power
    f["wind_speed"] = wind_speed

    # Lag features (notebook 03)
    for lag in [1, 2, 3, 6, 12, 144]:
        f[f"power_lag_{lag}"] = power.shift(lag)
    for lag in [1, 3, 6]:
        f[f"wind_speed_lag_{lag}"] = wind_speed.shift(lag)

    # Rolling features over the past 1 hour (6 steps) and 3 hours (18 steps)
    for window in [6, 18]:
        f[f"power_roll_mean_{window}"] = power.rolling(window).mean()
        f[f"power_roll_std_{window}"] = power.rolling(window).std()
        f[f"wind_speed_roll_mean_{window}"] = wind_speed.rolling(window).mean()

    # Cyclical encodings
    rad = np.deg2rad(readings["wind_direction"])
    f["wind_dir_sin"] = np.sin(rad)
    f["wind_dir_cos"] = np.cos(rad)
    hour = f.index.hour + f.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    month = f.index.month
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Trend features (notebook 04)
    f["power_diff_1"] = f["power"] - f["power_lag_1"]
    f["power_diff_6"] = f["power"] - f["power_lag_6"]
    f["wind_speed_diff_1"] = f["wind_speed"] - f["wind_speed_lag_1"]
    f["wind_speed_diff_6"] = f["wind_speed"] - f["wind_speed_lag_6"]

    return f
