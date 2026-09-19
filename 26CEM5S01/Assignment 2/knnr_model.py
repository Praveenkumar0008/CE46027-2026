import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Load the dataset
df = pd.read_csv(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\Dataset\Student_Performance.csv"
)

# Convert categorical variable into numerical values
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({
    "Yes": 1,
    "No": 0
})

# Define predictor variables
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


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Standardize the predictor variables
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create the KNNR model
knnr_model = KNeighborsRegressor(
    n_neighbors=5
)

# Train the model
knnr_model.fit(X_train_scaled, y_train)


# Make predictions
y_pred = knnr_model.predict(X_test_scaled)


# Calculate performance metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)


# Display results
print("K-NEAREST NEIGHBORS REGRESSION RESULTS")
print("---------------------------------------")

print("Number of neighbors (K):", 5)

print("R²:", r2)
print("RMSE:", rmse)
print("MAE:", mae)
