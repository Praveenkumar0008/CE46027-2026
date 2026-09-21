# Assignment 2: Performance Comparison of MLR and KNNR

## 1. Dataset

The Building Energy Efficiency dataset was obtained from Kaggle.

The dataset contains building characteristics and energy-load measurements.

## 2. Objective

The objective of this assignment is to compare the performance of:

- Multiple Linear Regression (MLR)
- K-Nearest Neighbors Regression (KNNR)

The models were evaluated using R², RMSE, MAE, computational resource utilization, and execution time.

## 3. Target Variable

Heating Load was selected as the dependent variable.

## 4. Input Features

The following eight variables were used as input features:

1. Relative Compactness
2. Surface Area
3. Wall Area
4. Roof Area
5. Overall Height
6. Orientation
7. Glazing Area
8. Glazing Area Distribution

Cooling Load was not used as an input feature because it is another output variable in the dataset.

## 5. Methodology

The dataset was divided into 80% training data and 20% testing data.

A random state of 42 was used to ensure reproducibility.

MLR was trained using the original feature values.

For KNNR, the input features were standardized using StandardScaler because KNN is a distance-based algorithm.

KNNR was implemented with K = 5.

## 6. Evaluation Metrics

The models were evaluated using:

- R²
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Training execution time
- Prediction execution time
- Memory utilization

## 7. Results

The obtained results are provided in:

`MLR_KNNR_Comparison.csv`

## 8. Conclusion

The performance of MLR and KNNR was compared based on prediction accuracy and computational requirements.

The results show how the two regression methods perform on the selected Building Energy Efficiency dataset.