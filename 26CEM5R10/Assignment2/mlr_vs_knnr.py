# ============================================================
# Assignment 2: Performance Comparison of MLR and KNNR
# Dataset: Corn (maize) farm data from Kaggle
# Goal: predict corn "Yield" from farm/farmer characteristics
# ============================================================

# ---------- STEP 1: Import the tools (libraries) we need ----------
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------- STEP 2: Load the dataset ----------
# (Upload corn_data.csv to Colab first - see instructions)
df = pd.read_csv("corn_data.csv")

print("Rows and columns:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])

# ---------- STEP 3: Choose target (Y) and inputs (X) ----------
target = "Yield"                     # what we want to predict

# 'Farmer' is just an ID and 'Crop' is "corn" in every row,
# so they carry no useful information -> remove them.
X = df.drop(columns=[target, "Farmer", "Crop"])
y = df[target]

# Separate columns into numbers and text categories
numeric_cols = X.select_dtypes(include="number").columns.tolist()
category_cols = X.select_dtypes(exclude="number").columns.tolist()
print("\nNumeric columns:", numeric_cols)
print("Category (text) columns:", category_cols)

# ---------- STEP 4: Prepare the data (preprocessing) ----------
# Numbers: fill missing values with the median, then scale them
#          (scaling is very important for KNN).
# Text:    fill missing values with the most common value, then
#          convert to 0/1 columns (one-hot encoding).
numeric_pipe = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
category_pipe = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", category_pipe, category_cols),
])

# ---------- STEP 5: Split into training (80%) and testing (20%) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining rows:", len(X_train), "| Testing rows:", len(X_test))

# ---------- STEP 6: Define the two models ----------
models = {
    "MLR (Multiple Linear Regression)": Pipeline([
        ("prep", preprocess),
        ("model", LinearRegression()),
    ]),
    "KNNR (K-Nearest Neighbors, k=5)": Pipeline([
        ("prep", preprocess),
        ("model", KNeighborsRegressor(n_neighbors=5)),
    ]),
}

# ---------- STEP 7: Train, predict, and measure everything ----------
results = []
predictions = {}

for name, model in models.items():
    # --- Training: measure time, CPU time and memory ---
    tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    model.fit(X_train, y_train)

    train_time = time.perf_counter() - wall_start
    train_cpu = time.process_time() - cpu_start
    _, train_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Prediction: measure time and memory ---
    tracemalloc.start()
    wall_start = time.perf_counter()

    y_pred = model.predict(X_test)

    predict_time = time.perf_counter() - wall_start
    _, predict_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    predictions[name] = y_pred

    # --- Accuracy metrics on the TEST data ---
    results.append({
        "Model": name,
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "Train time (s)": train_time,
        "Train CPU time (s)": train_cpu,
        "Predict time (s)": predict_time,
        "Train peak memory (MB)": train_peak_mem / 1024 / 1024,
        "Predict peak memory (MB)": predict_peak_mem / 1024 / 1024,
    })

# ---------- STEP 8: Show the comparison table ----------
results_df = pd.DataFrame(results).set_index("Model")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("\n================ RESULTS ================")
print(results_df.round(4).T)          # .T flips the table so it is easy to read
results_df.round(4).to_csv("comparison_results.csv")

# ---------- STEP 9: Make charts ----------
short = ["MLR", "KNNR"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Chart 1: accuracy metrics
results_df[["RMSE", "MAE"]].set_axis(short).plot(kind="bar", ax=axes[0], rot=0)
axes[0].set_title("Error (lower is better)")

# Chart 2: R2
axes[1].bar(short, results_df["R2"], color=["tab:blue", "tab:orange"])
axes[1].set_title("R2 score (higher is better)")

# Chart 3: time
results_df[["Train time (s)", "Predict time (s)"]].set_axis(short).plot(
    kind="bar", ax=axes[2], rot=0)
axes[2].set_title("Execution time (seconds)")

plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150)
plt.show()

# Chart 4: actual vs predicted
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.6)
    lims = [y.min(), y.max()]
    ax.plot(lims, lims, "r--")          # perfect prediction line
    ax.set_xlabel("Actual Yield")
    ax.set_ylabel("Predicted Yield")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# ---------- STEP 10: Automatic conclusion ----------
best_acc = results_df["R2"].idxmax()
fastest = results_df["Train time (s)"].idxmin()
print("\n================ CONCLUSION ================")
print("Most accurate model (highest R2):", best_acc)
print("Fastest to train:", fastest)
