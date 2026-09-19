import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

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

# Create the MLR model
mlr_model = LinearRegression()

# Train the model
mlr_model.fit(X_train, y_train)

# Make predictions
y_pred = mlr_model.predict(X_test)

# Calculate performance metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Display results
print("MULTIPLE LINEAR REGRESSION RESULTS")
print("-----------------------------------")

print("R²:", r2)
print("RMSE:", rmse)
print("MAE:", mae)

print("\nIntercept:", mlr_model.intercept_)

print("\nCoefficients:")
for feature, coefficient in zip(X.columns, mlr_model.coef_):
    print(feature, ":", coefficient)
