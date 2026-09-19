import pandas as pd
from sklearn.model_selection import train_test_split

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

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Display the results
print("Training data:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nTesting data:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nTotal observations:", len(X))
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
