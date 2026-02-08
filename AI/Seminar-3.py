#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# Seminar 3: Building Neural Networks 
#
# Dataset: UCI Air Quality (AirQualityUCI)
#
# Goals:
#   - Load a real-world dataset that contains messy values
#   - Perform minimal but realistic cleaning (handle sensor missing codes)
#   - Avoid time-series data leakage using a temporal train/test split
#   - Standardise features correctly (fit scaler on train only)
#   - Build a strong baseline (Linear Regression)
#   - Train a simple Neural Network 
#   - Evaluate with MAE and R2, and interpret plots 
# ============================================================


# ============================================================
# Imports and setup
# ============================================================

# We import libraries for:
# - numerical computing and data handling (numpy, pandas)
# - plotting (matplotlib)
# - preprocessing and evaluation (scikit-learn)
# - neural networks (TensorFlow / Keras)

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from tensorflow.keras import layers, models


# Set random seed for reproducibility (so you get similar results)
tf.keras.utils.set_random_seed(42)


# ============================================================
# Load dataset (UCI Air Quality)
# ============================================================

# This dataset is famous for being "messy":
# - often ';' separated
# - often uses ',' as the decimal separator
# - sometimes includes extra empty columns (e.g., "Unnamed" or an all-NaN column)
# In real business analytics, data rarely comes perfectly clean.
# A key skill is being able to load, inspect, and clean it safely.

csv_path = "AirQualityUCI.csv"

df = pd.read_csv(
    csv_path,
    sep=";",
    decimal=",",
    engine="python"
)


# After loading, printing df.head() is a sanity check
print(df.head())  



# ============================================================
# Select features and target
# ============================================================

# We define our prediction task:
# - Target: "NO2(GT)" (a measure of NO2 concentration)
# - Features: sensor signals and environmental conditions

feature_cols = [
    "CO(GT)",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]
target_col = "NO2(GT)"

use_cols = feature_cols + [target_col]
df = df[use_cols].copy()

print(df.head())



# ============================================================
# Minimal cleaning 
# ============================================================

# In this dataset, "-200" is widely used as a "missing value code".
# If we treat -200 as a real measurement, we will mislead the model badly.
df = df.replace(-200, np.nan)

# Drop rows with missing values after cleaning
df = df.dropna().reset_index(drop=True)

# After dropping NaNs, you have fewer rows, but higher quality data.



# ============================================================
# Data split
# ============================================================

# The aim of this step is to split data into two sub-sets, 
# one for training and one for testing.
# Machine learning model must work on unseen data.

X = df[feature_cols].astype(float).to_numpy()
y = df[target_col].astype(float).to_numpy()

n = len(df)
test_ratio = 0.25
test_size = int(np.floor(n * test_ratio))
train_size = n - test_size

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

print("Total rows:", n)
print("Train rows:", len(X_train), "Test rows:", len(X_test))


# ============================================================
#  Normalise X (fit on train only)
# ============================================================

# StandardScaler transforms each feature to:
# - mean = 0
# - standard deviation = 1
#
# Why we do this:
# - Linear regression can behave better with comparable feature scales
# - Neural networks train faster and more stably with standardised inputs
#
# Critical rule:
# - Fit the scaler ONLY on the training data
# - Then apply the same transformation to test data

scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)


# ============================================================
# Neural Network model (simple MLP)
# ============================================================

# We build a small multi-layer perceptron (MLP):
# - Input layer matches number of features
# - Hidden layers use ReLU to model non-linear patterns
# - Output is a single value (regression)

model = models.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
)

history = model.fit(
    X_train_s, y_train,
    epochs=10,
    batch_size=32,
    shuffle=False,
    verbose=1
)

y_pred_nn = model.predict(X_test_s, verbose=0).reshape(-1)

mae_nn = mean_absolute_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print("\nNeural Network performance (Time-series split, predict NO2(GT)):")
print("MAE:", mae_nn)
print("R2:", r2_nn)

# How to interpret MAE:
# - average absolute prediction error (same units as NO2)
# - lower is better


# Common mistakes - take away knowledge:
# - using a big network for a small dataset (you could try to add more layers into MLP)
# - running too few epochs and concluding “NN doesn’t work” (you could change epoch to 5 and see the results)
# - forgetting standardisation 


# ============================================================
# 7) Baseline model: Linear Regression
# ============================================================

# We build up a baseline model for comparison purpose
# Linear regression learns a linear relationship:
# y ≈ b0 + b1*x1 + ... + bk*xk

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression performance (Time-series split, predict NO2(GT)):")
print("MAE:", mae_lr)
print("R2:", r2_lr)

# Interpretation:
# - If NN is slightly better than LR, that suggests useful non-linearity exists.
# - If NN is worse, that does NOT necessily mean NN is “bad” — it may mean:
#   (1) dataset is small/noisy
#   (2) features already linearly explain most signal
#   (3) training time is too short
#   (4) network is not tuned




# ============================================================
# 9) Plots: Linear Regression vs Neural Network
# ============================================================

# Metrics summarise performance, but plots show how the model fails:
# - are predictions systematically biased?
# - are errors larger at high values?
# - do we see outliers dominating?

# ---------- Predicted vs Actual (Neural Network)
plt.figure()
plt.scatter(y_test, y_pred_nn)
plt.xlabel("Actual NO2(GT)")
plt.ylabel("Predicted NO2(GT)")
plt.title("Neural Network: Predicted vs Actual (NO2)")
plt.show()

# ---------- Predicted vs Actual (Linear Regression)
plt.figure()
plt.scatter(y_test, y_pred_lr)
plt.xlabel("Actual NO2(GT)")
plt.ylabel("Predicted NO2(GT)")
plt.title("Linear Regression: Predicted vs Actual (NO2)")
plt.show()

# Interpretation:
# - Ideally points lie near a 45-degree line (perfect prediction).
# - Big spread means large errors.



# ============================================================
# Practical Questions 
# ============================================================

# ------------------------------------------------------------
# Q1:
# What are the key modelling assumptions behind Linear Regression and Neural Networks?
# ------------------------------------------------------------

# ------------------------------------------------------------
# Q2:
# What does MAE mean in this context, and why is it useful?
# ------------------------------------------------------------


# ------------------------------------------------------------
# Q3:
# Which model (Linear Regression vs Neural Network) do you expect to be more stable if we slightly change
# the training data (or random seed)? Why?
# Hint: Please change the random seed and re-run the code, and observe the difference across runs.
# ------------------------------------------------------------


# ------------------------------------------------------------
# Q4:
# If you need to explain your prediction to a non-technical manager, which model would you choose and why?
# ------------------------------------------------------------

# ------------------------------------------------------------
# Q5:
# We can use domain knowledge to create additional engineered features (interaction between temperature and humidity).
# If we add a useful engineered feature, which model is MORE likely to benefit: Linear Regression or NN?
#
# Hint:
# Please copy and paste the following code just before the data split section, then re-run the notebook.
# Observe the performance change for BOTH Linear Regression and the Neural Network.
# ------------------------------------------------------------

# >>> Exercise snippet (copy/paste into the earlier section, just before the data split) >>>
df["T_RH_interaction"] = df["T"] * df["RH"]
df["T_RH_interaction"] = pd.to_numeric(df["T_RH_interaction"], errors="coerce")

feature_cols.append("T_RH_interaction")
df = df.dropna().reset_index(drop=True)
# <<< End of exercise snippet <<<

