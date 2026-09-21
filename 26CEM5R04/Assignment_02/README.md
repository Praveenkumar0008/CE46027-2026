
# Assignment 2 - Building Energy Efficiency

## Objective

This project compares Multiple Linear Regression (MLR) and
K-Nearest Neighbors Regression (KNNR) for predicting building
Heating Load and Cooling Load.

## Dataset

The dataset contains 768 observations and 8 input features.

### Input Features

- Relative Compactness
- Surface Area
- Wall Area
- Roof Area
- Overall Height
- Orientation
- Glazing Area
- Glazing Area Distribution

### Target Variables

- Heating Load
- Cooling Load

## Models

1. Multiple Linear Regression (MLR)
2. K-Nearest Neighbors Regression (KNNR)

## Data Split

The dataset was divided into:
- 80% training data
- 20% testing data
- random_state = 42

## KNNR Settings

- K = 5
- Weight = distance
- StandardScaler used for feature scaling

## Evaluation Metrics

- R² Score
- RMSE
- MAE
- Execution Time
- Peak Memory Usage

## Output Files

- comparison_results.csv
- actual_vs_predicted_values.csv
- actual_vs_predicted_comparison.png
- error_comparison.png
- r2_comparison.png
- time_comparison.png
- memory_comparison.png

## Dataset Source

Kaggle dataset URL:
PASTE-YOUR-KAGGLE-URL-HERE

## GitHub

GitHub ID:
PASTE-YOUR-GITHUB-ID-HERE
