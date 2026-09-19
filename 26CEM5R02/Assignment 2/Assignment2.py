# ============================================================
# ASSIGNMENT 2
# MLR vs KNNR - Insurance Premium Prediction
# ============================================================

import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# 10. TRAIN MODELS AND GENERATE PREDICTIONS
# ============================================================

mlr_model.fit(
    X_train,
    y_train
)

y_pred_mlr = mlr_model.predict(
    X_test
)


knn_model.fit(
    X_train,
    y_train
)

y_pred_knn = knn_model.predict(
    X_test
)


# ============================================================
# 11. CALCULATE PERFORMANCE METRICS
# ============================================================

# MLR
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


# KNNR
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
# 12. COMPUTATIONAL RESOURCE MEASUREMENT
# ============================================================

process = psutil.Process(
    os.getpid()
)

os.makedirs(
    "results",
    exist_ok=True
)

benchmark_file = (
    "results/computational_benchmark.csv"
)


# ------------------------------------------------------------
# If computational results already exist, reuse them.
# This prevents the comparison table from changing every
# time the script is executed.
# ------------------------------------------------------------

if os.path.exists(benchmark_file):

    benchmark = pd.read_csv(
        benchmark_file
    )

    mlr_wall_time = benchmark.loc[
        benchmark["Model"] == "MLR",
        "Wall_Time_s"
    ].iloc[0]

    mlr_cpu_time = benchmark.loc[
        benchmark["Model"] == "MLR",
        "CPU_Time_s"
    ].iloc[0]

    mlr_memory_change = benchmark.loc[
        benchmark["Model"] == "MLR",
        "Memory_Change_MB"
    ].iloc[0]

    knn_wall_time = benchmark.loc[
        benchmark["Model"] == "KNNR",
        "Wall_Time_s"
    ].iloc[0]

    knn_cpu_time = benchmark.loc[
        benchmark["Model"] == "KNNR",
        "CPU_Time_s"
    ].iloc[0]

    knn_memory_change = benchmark.loc[
        benchmark["Model"] == "KNNR",
        "Memory_Change_MB"
    ].iloc[0]

    print(
        "\nExisting computational benchmark found."
    )

    print(
        "Previously measured computational values "
        "are being reused."
    )


# ------------------------------------------------------------
# Otherwise perform the benchmark for the first time.
# ------------------------------------------------------------

else:

    print(
        "\nPerforming computational benchmark..."
    )

    benchmark_runs = 10

    mlr_wall_times = []
    mlr_cpu_times = []
    mlr_memory_changes = []

    knn_wall_times = []
    knn_cpu_times = []
    knn_memory_changes = []


    # --------------------------------------------------------
    # MLR BENCHMARK
    # --------------------------------------------------------

    for i in range(benchmark_runs):

        benchmark_mlr = Pipeline(
            steps=[
                (
                    "preprocessor",
                    mlr_preprocessor
                ),
                (
                    "regressor",
                    LinearRegression()
                )
            ]
        )

        memory_before = (
            process.memory_info().rss
        )

        wall_start = time.perf_counter()

        cpu_start = time.process_time()

        benchmark_mlr.fit(
            X_train,
            y_train
        )

        benchmark_mlr.predict(
            X_test
        )

        cpu_end = time.process_time()

        wall_end = time.perf_counter()

        memory_after = (
            process.memory_info().rss
        )

        mlr_wall_times.append(
            wall_end - wall_start
        )

        mlr_cpu_times.append(
            cpu_end - cpu_start
        )

        mlr_memory_changes.append(
            (
                memory_after -
                memory_before
            ) / (1024 ** 2)
        )


    # --------------------------------------------------------
    # KNNR BENCHMARK
    # --------------------------------------------------------

    for i in range(benchmark_runs):

        benchmark_knn = Pipeline(
            steps=[
                (
                    "preprocessor",
                    knn_preprocessor
                ),
                (
                    "regressor",
                    KNeighborsRegressor(
                        n_neighbors=5
                    )
                )
            ]
        )

        memory_before = (
            process.memory_info().rss
        )

        wall_start = time.perf_counter()

        cpu_start = time.process_time()

        benchmark_knn.fit(
            X_train,
            y_train
        )

        benchmark_knn.predict(
            X_test
        )

        cpu_end = time.process_time()

        wall_end = time.perf_counter()

        memory_after = (
            process.memory_info().rss
        )

        knn_wall_times.append(
            wall_end - wall_start
        )

        knn_cpu_times.append(
            cpu_end - cpu_start
        )

        knn_memory_changes.append(
            (
                memory_after -
                memory_before
            ) / (1024 ** 2)
        )


    # --------------------------------------------------------
    # USE MEAN OF BENCHMARK RUNS
    # --------------------------------------------------------

    mlr_wall_time = np.mean(
        mlr_wall_times
    )

    mlr_cpu_time = np.mean(
        mlr_cpu_times
    )

    mlr_memory_change = np.mean(
        mlr_memory_changes
    )


    knn_wall_time = np.mean(
        knn_wall_times
    )

    knn_cpu_time = np.mean(
        knn_cpu_times
    )

    knn_memory_change = np.mean(
        knn_memory_changes
    )


    # --------------------------------------------------------
    # SAVE COMPUTATIONAL BENCHMARK
    # --------------------------------------------------------

    benchmark = pd.DataFrame({

        "Model": [
            "MLR",
            "KNNR"
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
            mlr_memory_change,
            knn_memory_change
        ]
    })


    benchmark.to_csv(
        benchmark_file,
        index=False
    )

    print(
        "Computational benchmark completed "
        "and saved."
    )


# ============================================================
# 13. FINAL COMPARISON TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [
        "MLR",
        "KNNR"
    ],

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
        mlr_memory_change,
        knn_memory_change
    ]
})


# ============================================================
# 14. DISPLAY FINAL RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("MLR vs KNNR PERFORMANCE COMPARISON")
print("=" * 90)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 15. SAVE FINAL RESULTS
# ============================================================

results.to_csv(
    "results/comparison_results.csv",
    index=False
)

print(
    "\nResults saved successfully."
)

print(
    "File: results/comparison_results.csv"
)


# ============================================================
# GRAPH SETTINGS
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["font.size"] = 12

plt.rcParams["axes.titlesize"] = 16

plt.rcParams["axes.labelsize"] = 13

plt.rcParams["xtick.labelsize"] = 12

plt.rcParams["ytick.labelsize"] = 12

plt.rcParams["legend.fontsize"] = 11


# ============================================================
# 16. ACTUAL VS PREDICTED
#     TWO SEPARATE GRAPHS IN ONE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6.5)
)


# ------------------------------------------------------------
# COMMON AXIS LIMITS
# ------------------------------------------------------------

all_values = np.concatenate([
    np.asarray(y_test),
    np.asarray(y_pred_mlr),
    np.asarray(y_pred_knn)
])

plot_min = np.floor(
    all_values.min() / 5000
) * 5000

plot_max = np.ceil(
    all_values.max() / 5000
) * 5000


# ------------------------------------------------------------
# MLR GRAPH
# ------------------------------------------------------------

axes[0].scatter(
    y_test,
    y_pred_mlr,
    s=42,
    alpha=0.65,
    edgecolors="black",
    linewidths=0.4
)

axes[0].plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

axes[0].set_xlim(
    plot_min,
    plot_max
)

axes[0].set_ylim(
    plot_min,
    plot_max
)

axes[0].set_title(
    "Multiple Linear Regression",
    fontweight="bold",
    pad=12
)

axes[0].set_xlabel(
    "Actual Expenses"
)

axes[0].set_ylabel(
    "Predicted Expenses"
)

axes[0].legend(
    loc="upper left",
    frameon=True
)

axes[0].grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)


# ------------------------------------------------------------
# KNNR GRAPH
# ------------------------------------------------------------

axes[1].scatter(
    y_test,
    y_pred_knn,
    s=42,
    alpha=0.65,
    edgecolors="black",
    linewidths=0.4
)

axes[1].plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

axes[1].set_xlim(
    plot_min,
    plot_max
)

axes[1].set_ylim(
    plot_min,
    plot_max
)

axes[1].set_title(
    "K-Nearest Neighbors Regression",
    fontweight="bold",
    pad=12
)

axes[1].set_xlabel(
    "Actual Expenses"
)

axes[1].set_ylabel(
    "Predicted Expenses"
)

axes[1].legend(
    loc="upper left",
    frameon=True
)

axes[1].grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)


fig.suptitle(
    "Actual vs Predicted Expenses",
    fontsize=18,
    fontweight="bold",
    y=1.02
)

plt.tight_layout()

plt.savefig(
    "results/actual_vs_predicted_comparison.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. R² COMPARISON
# ============================================================

models = [
    "MLR",
    "KNNR"
]

r2_values = [
    r2_mlr,
    r2_knn
]

fig, ax = plt.subplots(
    figsize=(8, 6)
)

bars = ax.bar(
    models,
    r2_values,
    width=0.55
)

ax.set_title(
    "R² Comparison of MLR and KNNR",
    fontsize=17,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Regression Model"
)

ax.set_ylabel(
    "R²"
)

ax.set_ylim(
    0,
    1.10
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)


# Fixed, aligned value labels
for bar, value in zip(
    bars,
    r2_values
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.025,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )


plt.tight_layout()

plt.savefig(
    "results/r2_comparison.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. RMSE AND MAE COMPARISON
# ============================================================

metrics = [
    "RMSE",
    "MAE"
]

mlr_errors = [
    rmse_mlr,
    mae_mlr
]

knn_errors = [
    rmse_knn,
    mae_knn
]

x = np.arange(
    len(metrics)
)

width = 0.34

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars_mlr = ax.bar(
    x - width / 2,
    mlr_errors,
    width,
    label="MLR"
)

bars_knn = ax.bar(
    x + width / 2,
    knn_errors,
    width,
    label="KNNR"
)

ax.set_title(
    "RMSE and MAE Comparison",
    fontsize=17,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Error Metric"
)

ax.set_ylabel(
    "Error in Expenses"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    metrics
)

ax.legend(
    frameon=True
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)


# ------------------------------------------------------------
# FIXED VALUE LABEL ALIGNMENT
# ------------------------------------------------------------

for bar in bars_mlr:

    value = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 500,
        f"{value:,.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


for bar in bars_knn:

    value = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 500,
        f"{value:,.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


# Add sufficient space above labels
largest_error = max(
    mlr_errors + knn_errors
)

ax.set_ylim(
    0,
    largest_error * 1.18
)

plt.tight_layout()

plt.savefig(
    "results/error_comparison.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 19. COMPUTATIONAL TIME COMPARISON
# ============================================================

wall_times = [
    mlr_wall_time,
    knn_wall_time
]

cpu_times = [
    mlr_cpu_time,
    knn_cpu_time
]

x = np.arange(
    len(models)
)

width = 0.34

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars_wall = ax.bar(
    x - width / 2,
    wall_times,
    width,
    label="Wall Time"
)

bars_cpu = ax.bar(
    x + width / 2,
    cpu_times,
    width,
    label="CPU Time"
)

ax.set_title(
    "Computational Time Comparison",
    fontsize=17,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Regression Model"
)

ax.set_ylabel(
    "Time (seconds)"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    models
)

ax.legend(
    frameon=True
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)


# ------------------------------------------------------------
# DETERMINE LABEL OFFSET
# ------------------------------------------------------------

largest_time = max(
    wall_times + cpu_times
)

time_offset = max(
    largest_time * 0.05,
    0.0001
)


# ------------------------------------------------------------
# ADD ALIGNED LABELS
# ------------------------------------------------------------

for bar in bars_wall:

    value = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + time_offset,
        f"{value:.6f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


for bar in bars_cpu:

    value = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + time_offset,
        f"{value:.6f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


ax.set_ylim(
    0,
    largest_time * 1.25
)

plt.tight_layout()

plt.savefig(
    "results/time_comparison.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. MEMORY CHANGE COMPARISON
# ============================================================

memory_values = [
    mlr_memory_change,
    knn_memory_change
]

fig, ax = plt.subplots(
    figsize=(8, 6)
)

bars = ax.bar(
    models,
    memory_values,
    width=0.55
)

ax.set_title(
    "Memory Change Comparison",
    fontsize=17,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Regression Model"
)

ax.set_ylabel(
    "Memory Change (MB)"
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)


# ------------------------------------------------------------
# MEMORY LABEL ALIGNMENT
# ------------------------------------------------------------

largest_memory = max(
    memory_values
)

if largest_memory > 0:

    memory_offset = (
        largest_memory * 0.05
    )

    for bar, value in zip(
        bars,
        memory_values
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + memory_offset,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_ylim(
        0,
        largest_memory * 1.25
    )

else:

    for bar, value in zip(
        bars,
        memory_values
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_ylim(
        0,
        1
    )


plt.tight_layout()

plt.savefig(
    "results/memory_comparison.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 21. FINAL GRAPH LIST
# ============================================================

print("\n")
print("=" * 70)
print("GRAPHS SAVED SUCCESSFULLY")
print("=" * 70)

print(
    "1. results/actual_vs_predicted_comparison.png"
)

print(
    "2. results/r2_comparison.png"
)

print(
    "3. results/error_comparison.png"
)

print(
    "4. results/time_comparison.png"
)

print(
    "5. results/memory_comparison.png"
)

print(
    "\nAll graphs use Times New Roman and 600 DPI."
)

print(
    "Computational benchmark is stored in:"
)

print(
    "results/computational_benchmark.csv"
)