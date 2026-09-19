# Assignment 2: Performance Comparison of MLR and KNNR

## 1. Objective

The objective of this assignment is to compare the performance of
Multiple Linear Regression (MLR) and K-Nearest Neighbors Regression
(KNNR) using an Insurance Premium Prediction dataset.

The comparison is based on prediction accuracy, computational resource
utilization, and execution time.

------------------------------------------------------------------------

## 2. Dataset

### Dataset Name

Insurance Premium Prediction

### Dataset Source

Kaggle

### Dataset URL

https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction/versions/1

### Dataset Description

The dataset contains information related to individuals and their
insurance expenses.

The predictor variables used in this assignment are:

-   age
-   sex
-   bmi
-   children
-   smoker
-   region

The target variable is:

-   expenses

The dataset contains 1,338 observations and 7 columns.

------------------------------------------------------------------------

## 3. Software and Tools Used

The following software and Python libraries were used:

-   Python
-   Anaconda
-   Spyder
-   pandas
-   NumPy
-   scikit-learn
-   psutil

------------------------------------------------------------------------

## 4. Data Preprocessing

The dataset was first loaded using pandas and inspected for its
dimensions, column names, missing values, data types, and statistical
summary.

The predictor variables were divided into numerical and categorical
features.

### Numerical Features

-   age
-   bmi
-   children

### Categorical Features

-   sex
-   smoker
-   region

Categorical variables were converted into numerical representations
using One-Hot Encoding.

For KNN Regression, the numerical features were standardized using
StandardScaler because KNN uses distance-based calculations.

------------------------------------------------------------------------

## 5. Train-Test Split

The dataset was divided into training and testing subsets using an 80:20
ratio.

The following settings were used:

-   Training data: 80%
-   Testing data: 20%
-   random_state: 42

The same train-test split was used for both MLR and KNNR to provide a
consistent basis for comparison.

------------------------------------------------------------------------

## 6. Multiple Linear Regression

Multiple Linear Regression was used to model the relationship between
the input variables and insurance expenses.

The model was implemented using the `LinearRegression` class from
scikit-learn.

The categorical variables were one-hot encoded before being supplied to
the regression model.

------------------------------------------------------------------------

## 7. K-Nearest Neighbors Regression

K-Nearest Neighbors Regression was used as the second regression model.

The value of k was set to:

**k = 5**

The numerical variables were standardized before applying KNN
Regression, and categorical variables were converted using One-Hot
Encoding.

------------------------------------------------------------------------

## 8. Evaluation Metrics

The performance of both models was evaluated using the following
metrics:

### R²

R² measures the proportion of variation in the target variable that is
explained by the regression model.

### RMSE

Root Mean Squared Error measures the square root of the average squared
difference between actual and predicted values.

### MAE

Mean Absolute Error measures the average absolute difference between
actual and predicted values.

------------------------------------------------------------------------

## 9. Computational Performance

In addition to prediction accuracy, computational performance was
measured for both models.

The following measures were recorded:

-   Wall-clock execution time
-   CPU process time
-   Memory usage/change

The Python `time` and `psutil` libraries were used to obtain these
measurements.

------------------------------------------------------------------------

## 10. Results

The final results obtained from the Python program are shown below.

  -------------------------------------------------------------------------------
  Model           R²          RMSE           MAE  Wall Time   CPU Time     Memory
                                                        (s)        (s)     Change
                                                                             (MB)
  ------- ---------- ------------- ------------- ---------- ---------- ----------
  MLR       0.783573   5796.556336   4181.561524   0.018850   0.015625   0.015625

  KNNR      0.698759   6838.664775   3886.129575   0.027067   0.187500   0.000000
  -------------------------------------------------------------------------------

The values in the table were obtained by executing `assignment2.py`
using the selected Insurance Premium Prediction dataset.

------------------------------------------------------------------------

## 11. Code Workflow

The Python program follows these steps:

1.  Load the insurance dataset.
2.  Inspect the dataset.
3.  Separate predictor variables and target variable.
4.  Identify numerical and categorical features.
5.  Split the dataset into training and testing data.
6.  Prepare the preprocessing pipeline for MLR.
7.  Train the Multiple Linear Regression model.
8.  Generate MLR predictions.
9.  Calculate MLR performance metrics.
10. Prepare the preprocessing pipeline for KNNR.
11. Standardize numerical features and encode categorical features.
12. Train the KNN Regression model with k = 5.
13. Generate KNNR predictions.
14. Calculate KNNR performance metrics.
15. Measure execution time and computational resource usage.
16. Create a comparison table.
17. Save the final results as `comparison_results.csv`.

------------------------------------------------------------------------

## 12. Output File

The program automatically creates a `results` folder and saves the final
comparison table as:

`results/comparison_results.csv`

The CSV file contains the performance results of both MLR and KNNR.

------------------------------------------------------------------------

## 13. How to Run the Program

### Step 1

Install Anaconda.

### Step 2

Activate the Python environment used for the assignment.

### Step 3

Install the required Python libraries.

### Step 4

Place `insurance.csv` in the same directory as `assignment2.py`.

### Step 5

Run `assignment2.py`.

### Step 6

The final model comparison is displayed in the console and saved to:

`results/comparison_results.csv`

