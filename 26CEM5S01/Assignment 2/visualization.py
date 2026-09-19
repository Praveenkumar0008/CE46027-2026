import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\Dataset\Student_Performance.csv"
)


# --------------------------------------------------
# 2. DATA PREPROCESSING
# --------------------------------------------------

df["Extracurricular Activities"] = df["Extracurricular Activities"].map({
    "Yes": 1,
    "No": 0
})

X = df[
    [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced"
    ]
]

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


# --------------------------------------------------
# 4. MLR MODEL
# --------------------------------------------------

mlr_model = LinearRegression()

mlr_model.fit(X_train, y_train)

y_pred_mlr = mlr_model.predict(X_test)


# --------------------------------------------------
# 5. KNNR MODEL
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsRegressor(
    n_neighbors=5
)

knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)


# --------------------------------------------------
# 6. ACTUAL VS PREDICTED VALUES
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_mlr,
    alpha=0.5,
    label="MLR"
)

plt.scatter(
    y_test,
    y_pred_knn,
    alpha=0.5,
    label="KNNR (K=5)"
)

min_value = min(y_test.min(), y_pred_mlr.min(), y_pred_knn.min())
max_value = max(y_test.max(), y_pred_mlr.max(), y_pred_knn.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Actual vs Predicted Performance Index")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\output\actual_vs_predicted.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 7. PERFORMANCE METRICS
# --------------------------------------------------

metrics = pd.read_csv(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\output\model_comparison.csv"
)


# --------------------------------------------------
# 8. R-SQUARED COMPARISON
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    metrics["Model"],
    metrics["R2"]
)

plt.ylabel("R²")
plt.title("R² Comparison of MLR and KNNR")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\output\r2_comparison.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 9. RMSE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    metrics["Model"],
    metrics["RMSE"]
)

plt.ylabel("RMSE")
plt.title("RMSE Comparison of MLR and KNNR")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\output\rmse_comparison.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 10. MAE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    metrics["Model"],
    metrics["MAE"]
)

plt.ylabel("MAE")
plt.title("MAE Comparison of MLR and KNNR")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\output\mae_comparison.png",
    dpi=300
)

plt.show()


print("\nVisualization completed successfully.")

print("\nFigures saved in:")
print(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\output"
)
