import os
import time
import psutil
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("insurance.csv")


# ============================================================
# 2. DATASET INSPECTION
# ============================================================

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["expenses"])
y = df["expenses"]


# ============================================================
# 4. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "age",
    "bmi",
    "children"
]

categorical_features = [
    "sex",
    "smoker",
    "region"
]


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 6. MULTIPLE LINEAR REGRESSION PREPROCESSING
# ============================================================

mlr_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 7. MULTIPLE LINEAR REGRESSION MODEL
# ============================================================

mlr_model = Pipeline(
    steps=[
        ("preprocessor", mlr_preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ============================================================
# 8. KNN REGRESSION PREPROCESSING
# ============================================================

knn_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 9. KNN REGRESSION MODEL
# ============================================================

knn_model = Pipeline(
    steps=[
        ("preprocessor", knn_preprocessor),
        (
            "regressor",
            KNeighborsRegressor(
                n_neighbors=5
            )
        )
    ]
)


# ============================================================
# 10. PROCESS INFORMATION
# ============================================================

process = psutil.Process(os.getpid())


# ============================================================
# 11. MLR TRAINING, PREDICTION AND RESOURCE MEASUREMENT
# ============================================================

mlr_start_wall = time.perf_counter()
mlr_start_cpu = time.process_time()
mlr_start_memory = process.memory_info().rss

mlr_model.fit(X_train, y_train)

y_pred_mlr = mlr_model.predict(X_test)

mlr_end_wall = time.perf_counter()
mlr_end_cpu = time.process_time()
mlr_end_memory = process.memory_info().rss

mlr_wall_time = mlr_end_wall - mlr_start_wall
mlr_cpu_time = mlr_end_cpu - mlr_start_cpu

mlr_memory_change = (
    mlr_end_memory - mlr_start_memory
)


# ============================================================
# 12. MLR PERFORMANCE METRICS
# ============================================================

r2_mlr = r2_score(
    y_test,
    y_pred_mlr
)

rmse_mlr = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_mlr
    )
)

mae_mlr = mean_absolute_error(
    y_test,
    y_pred_mlr
)


# ============================================================
# 13. KNN REGRESSION TRAINING, PREDICTION
#     AND RESOURCE MEASUREMENT
# ============================================================

knn_start_wall = time.perf_counter()
knn_start_cpu = time.process_time()
knn_start_memory = process.memory_info().rss

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

knn_end_wall = time.perf_counter()
knn_end_cpu = time.process_time()
knn_end_memory = process.memory_info().rss

knn_wall_time = knn_end_wall - knn_start_wall
knn_cpu_time = knn_end_cpu - knn_start_cpu

knn_memory_change = (
    knn_end_memory - knn_start_memory
)


# ============================================================
# 14. KNN REGRESSION PERFORMANCE METRICS
# ============================================================

r2_knn = r2_score(
    y_test,
    y_pred_knn
)

rmse_knn = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_knn
    )
)

mae_knn = mean_absolute_error(
    y_test,
    y_pred_knn
)


# ============================================================
# 15. CREATE FINAL COMPARISON TABLE
# ============================================================

results = pd.DataFrame({
    "Model": ["MLR", "KNNR"],

    "R2": [
        r2_mlr,
        r2_knn
    ],

    "RMSE": [
        rmse_mlr,
        rmse_knn
    ],

    "MAE": [
        mae_mlr,
        mae_knn
    ],

    "Wall_Time_s": [
        mlr_wall_time,
        knn_wall_time
    ],

    "CPU_Time_s": [
        mlr_cpu_time,
        knn_cpu_time
    ],

    "Memory_Change_MB": [
        mlr_memory_change / (1024 ** 2),
        knn_memory_change / (1024 ** 2)
    ]
})


# ============================================================
# 16. DISPLAY FINAL RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("MLR vs KNNR PERFORMANCE COMPARISON")
print("=" * 80)

print(results.to_string(index=False))


# ============================================================
# 17. SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

results.to_csv(
    "results/comparison_results.csv",
    index=False
)

print("\nResults saved successfully.")
print("File: results/comparison_results.csv")