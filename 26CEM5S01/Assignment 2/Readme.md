# Performance Comparison of MLR and KNNR

## Assignment 2

This project presents a performance comparison between Multiple Linear Regression (MLR) and K-Nearest Neighbors Regression (KNNR) using a Student Performance dataset. The models are evaluated using prediction accuracy metrics and computational resource measurements.

---

## 1. Dataset

### Dataset Name
Student Performance

### Source
Kaggle

### Dataset URL
https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression

### Dataset Description

The dataset contains 10,000 student records and includes academic and lifestyle-related variables. The objective is to predict the Performance Index using selected input variables.

### Dataset Size

- Number of observations: 10,000
- Number of variables: 6

### Variables

| Variable | Description |
|---|---|
| Hours Studied | Number of hours studied |
| Previous Scores | Previous academic score |
| Extracurricular Activities | Participation in extracurricular activities |
| Sleep Hours | Number of hours of sleep |
| Sample Question Papers Practiced | Number of sample question papers practiced |
| Performance Index | Target variable |

---

## 2. Objective

The objective of this assignment is to compare Multiple Linear Regression (MLR) and K-Nearest Neighbors Regression (KNNR) based on:

- R²
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Execution time
- CPU time
- Memory utilization

---

## 3. Software and Libraries

### Software

- Python 3.11.5
- Python IDLE
- GitHub

### Python Libraries

- pandas
- NumPy
- scikit-learn
- matplotlib
- psutil

---

## 4. Data Preprocessing

The dataset was first loaded using pandas.

The `Extracurricular Activities` variable is categorical and was converted into numerical form:

- Yes = 1
- No = 0

The following five variables were used as predictors:

1. Hours Studied
2. Previous Scores
3. Extracurricular Activities
4. Sleep Hours
5. Sample Question Papers Practiced

The target variable was:

- Performance Index

The dataset contained 10,000 observations and no missing values.

---

## 5. Train-Test Split

The dataset was divided into training and testing datasets using an 80:20 ratio.

- Training samples: 8,000
- Testing samples: 2,000
- Test size: 20%
- Random state: 42

The same train-test split was used for both models to ensure a consistent comparison.

---

## 6. Multiple Linear Regression

Multiple Linear Regression was implemented using the `LinearRegression` algorithm from scikit-learn.

The model estimates the relationship between the Performance Index and the five predictor variables.

The model was trained using the 8,000 training observations and evaluated using the 2,000 testing observations.

---

## 7. K-Nearest Neighbors Regression

K-Nearest Neighbors Regression was implemented using `KNeighborsRegressor`.

The number of neighbors was set to:

**K = 5**

Since KNN is distance-based, the predictor variables were standardized using `StandardScaler`.

The scaler was fitted only on the training data and subsequently applied to the test data.

---

## 8. Evaluation Metrics

### R²

R² measures the proportion of variation in the target variable explained by the model. Higher values indicate greater explanatory performance.

### RMSE

Root Mean Square Error measures the square root of the average squared prediction error. Lower values indicate smaller prediction errors.

### MAE

Mean Absolute Error measures the average absolute difference between observed and predicted values. Lower values indicate smaller prediction errors.

---

## 9. Computational Resource Measurement

The computational performance of the models was assessed using:

- Wall-clock execution time
- CPU processing time
- Process-level memory change

The `time` and `psutil` Python libraries were used for these measurements.

The reported resource values represent the measurements obtained during the experimental run on the computer used for this assignment.

---

## 10. Results

| Parameter | MLR | KNNR (K=5) |
|---|---:|---:|
| R² | 0.988983 | 0.976898 |
| RMSE | 2.020552 | 2.925932 |
| MAE | 1.611121 | 2.360400 |
| Execution Time (seconds) | 0.022669 | 0.016406 |
| CPU Time (seconds) | 0.015625 | 0.000000* |
| Memory Change (MB) | 1.609375 | 0.523438 |

*The measured CPU time for KNNR was below the reporting resolution of the timer and was recorded as 0.000000 seconds.

---

## 11. Results Interpretation

The MLR model achieved an R² value of 0.988983, while KNNR achieved an R² value of 0.976898. The RMSE values were 2.020552 for MLR and 2.925932 for KNNR. The corresponding MAE values were 1.611121 and 2.360400.

The measured prediction errors were therefore lower for MLR for this particular test dataset and train-test split.

In terms of the measured computational resources, KNNR recorded a lower wall-clock execution time and lower process-level memory change during the experimental run. The execution time was 0.016406 seconds for KNNR compared with 0.022669 seconds for MLR. The measured memory change was 0.523438 MB for KNNR and 1.609375 MB for MLR.

Because the execution times are very small, the computational measurements can vary depending on system conditions and should be interpreted as observations from this experimental run.

---

## 12. Visualizations

### Actual vs Predicted Performance Index

The scatter plot compares the actual Performance Index values with the predicted values from MLR and KNNR. The diagonal line represents perfect prediction.

### R² Comparison

The R² comparison illustrates the predictive performance of the two regression models.

### RMSE Comparison

The RMSE comparison illustrates the magnitude of prediction errors produced by the two models.

### MAE Comparison

The MAE comparison illustrates the average absolute prediction error of the models.

### Execution Time Comparison

The execution-time graph compares the measured wall-clock time required by the two models.

### Memory Utilization Comparison

The memory comparison shows the measured process-level memory change during model execution.

---

## 13. Project Structure

```text
Assignment 2/
│
├── Dataset/
│   └── Student_Performance.csv
│
├── output/
│   ├── model_comparison.csv
│   ├── actual_vs_predicted.png
│   ├── r2_comparison.png
│   ├── rmse_comparison.png
│   ├── mae_comparison.png
│   ├── execution_time_comparison.png
│   └── memory_comparison.png
│
├── py_files/
│   ├── check_dataset.py
│   ├── prepare_dataset.py
│   ├── split_dataset.py
│   ├── mlr_model.py
│   ├── knnr_model.py
│   ├── model_comparison.py
│   └── visualization.py
│
└── README.md
