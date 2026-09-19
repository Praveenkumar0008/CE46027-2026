import pandas as pd
import numpy as np
import time
import psutil
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\Dataset\Student_Performance.csv"
)

print("DATASET INFORMATION")
print("-------------------")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. DATA PREPROCESSING
# --------------------------------------------------

# Convert categorical variable into numerical values
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({
    "Yes": 1,
    "No": 0
})

# Define input variables
X = df[
    [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced"
    ]
]

# Define target variable
y = df["Performance Index"]


# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN-TEST SPLIT")
print("----------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. SYSTEM RESOURCE INFORMATION
# --------------------------------------------------

process = psutil.Process(os.getpid())


# ==================================================
# 5. MULTIPLE LINEAR REGRESSION
# ==================================================

memory_before_mlr = process.memory_info().rss / (1024 ** 2)

start_wall_mlr = time.perf_counter()
start_cpu_mlr = time.process_time()

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

y_pred_mlr = mlr_model.predict(X_test)

end_cpu_mlr = time.process_time()
end_wall_mlr = time.perf_counter()

memory_after_mlr = process.memory_info().rss / (1024 ** 2)


# MLR metrics
mlr_r2 = r2_score(y_test, y_pred_mlr)
mlr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlr))
mlr_mae = mean_absolute_error(y_test, y_pred_mlr)

mlr_wall_time = end_wall_mlr - start_wall_mlr
mlr_cpu_time = end_cpu_mlr - start_cpu_mlr
mlr_memory = memory_after_mlr - memory_before_mlr


# ==================================================
# 6. KNN REGRESSION
# ==================================================

# Standardization is important for distance-based KNN
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


memory_before_knn = process.memory_info().rss / (1024 ** 2)

start_wall_knn = time.perf_counter()
start_cpu_knn = time.process_time()

knn_model = KNeighborsRegressor(
    n_neighbors=5
)

knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)

end_cpu_knn = time.process_time()
end_wall_knn = time.perf_counter()

memory_after_knn = process.memory_info().rss / (1024 ** 2)


# KNN metrics
knn_r2 = r2_score(y_test, y_pred_knn)
knn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
knn_mae = mean_absolute_error(y_test, y_pred_knn)

knn_wall_time = end_wall_knn - start_wall_knn
knn_cpu_time = end_cpu_knn - start_cpu_knn
knn_memory = memory_after_knn - memory_before_knn


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("\n")
print("=" * 60)
print("PERFORMANCE COMPARISON: MLR vs KNNR")
print("=" * 60)

print("\nMULTIPLE LINEAR REGRESSION")
print("--------------------------")
print("R²:", mlr_r2)
print("RMSE:", mlr_rmse)
print("MAE:", mlr_mae)
print("Execution Time (seconds):", mlr_wall_time)
print("CPU Time (seconds):", mlr_cpu_time)
print("Memory Change (MB):", mlr_memory)


print("\nK-NEAREST NEIGHBORS REGRESSION")
print("------------------------------")
print("Number of Neighbors (K): 5")
print("R²:", knn_r2)
print("RMSE:", knn_rmse)
print("MAE:", knn_mae)
print("Execution Time (seconds):", knn_wall_time)
print("CPU Time (seconds):", knn_cpu_time)
print("Memory Change (MB):", knn_memory)


# --------------------------------------------------
# 8. CREATE COMPARISON TABLE
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Multiple Linear Regression",
        "K-Nearest Neighbors Regression"
    ],
    "R2": [
        mlr_r2,
        knn_r2
    ],
    "RMSE": [
        mlr_rmse,
        knn_rmse
    ],
    "MAE": [
        mlr_mae,
        knn_mae
    ],
    "Execution_Time_Seconds": [
        mlr_wall_time,
        knn_wall_time
    ],
    "CPU_Time_Seconds": [
        mlr_cpu_time,
        knn_cpu_time
    ],
    "Memory_Change_MB": [
        mlr_memory,
        knn_memory
    ]
})


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

output_path = (
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems"
    r"\Assignment 2\output\model_comparison.csv"
)

comparison.to_csv(
    output_path,
    index=False
)

print("\n")
print("=" * 60)
print("FINAL COMPARISON TABLE")
print("=" * 60)

print(comparison.to_string(index=False))

print("\nResults saved successfully to:")
print(output_path)
