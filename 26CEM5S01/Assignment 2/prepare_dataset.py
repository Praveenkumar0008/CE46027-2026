import pandas as pd

# Load the dataset
df = pd.read_csv(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\Dataset\Student_Performance.csv"
)

# Convert categorical variable into numerical values
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({
    "Yes": 1,
    "No": 0
})

# Define predictor variables (X)
X = df[
    [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced"
    ]
]

# Define target variable (y)
y = df["Performance Index"]

# Display the prepared data
print("Prepared predictor variables (X):")
print(X.head())

print("\nTarget variable (y):")
print(y.head())

print("\nPredictor data types:")
print(X.dtypes)

print("\nNumber of predictor variables:", X.shape[1])
print("Number of observations:", X.shape[0])

print("\nMissing values in X:")
print(X.isnull().sum())

print("\nMissing values in y:")
print(y.isnull().sum())
