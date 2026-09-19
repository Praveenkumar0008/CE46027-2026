import pandas as pd

df = pd.read_csv(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\Dataset\Student_Performance.csv"
)

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())
