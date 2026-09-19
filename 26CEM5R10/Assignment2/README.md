Assignment 2: Performance Comparison of MLR and KNNR
Name: Sahla Abdul Salam
Roll number: 26CEM5R10
GitHub ID: Sahla-AbdulSalam
Dataset: Corn (maize) farm survey data, Kaggle — https://www.kaggle.com/datasets/japondo/corn-farming-data
1. Objective
Compare Multiple Linear Regression (MLR) and K-Nearest Neighbors Regression (KNNR) on the corn dataset in terms of
(a) accuracy (R², RMSE, MAE) and (b) computational resource use and execution time.
2. Dataset
422 rows (farmers) and 22 columns.
Target variable: `Yield` (corn yield).
Input variables: household size, acreage, fertilizer amount, number of laborers, latitude, longitude, and categorical
variables such as county, education, gender, age bracket, power source, water source, credit source, crop insurance,
farm records, and advisory details.
`Farmer` (an ID) and `Crop` (always "corn") were removed because they carry no predictive information.
Missing values: `Acreage` (71) and `Education` (26).
3. Steps followed
Created a GitHub account.
Downloaded the corn dataset from Kaggle and entered my GitHub ID and dataset URL in the Google Sheet.
Used Google Colab (Python environment) to run the code.
Wrote a Python script (`mlr_vs_knnr.py`) that does the following:
Imports libraries: pandas, numpy, matplotlib, scikit-learn.
Loads `corn_data.csv`.
Selects `Yield` as the target and the remaining useful columns as inputs.
Preprocesses the data:
Missing numbers are filled with the median; missing categories with the most frequent value.
Numeric columns are standardized (essential for KNN, which is distance-based).
Text columns are converted to 0/1 columns (one-hot encoding).
Splits the data: 80% training, 20% testing (`random_state=42` so results are reproducible).
Trains MLR and KNNR (k = 5) using identical preprocessing.
Measures accuracy on the test set: R², RMSE, MAE.
Measures resources: wall-clock time, CPU time (training) and peak memory (training and prediction).
Saves a results table (`comparison_results.csv`) and charts (`metrics_comparison.png`, `actual_vs_predicted.png`).
Uploaded the work to the `nitw-gis/CE46027-2026` repository in `<roll number>/Assignment2`.
4. Metrics explained
R²: fraction of variation in yield explained by the model (closer to 1 is better).
RMSE: typical prediction error, in yield units; penalizes large errors (lower is better).
MAE: average absolute prediction error (lower is better).
5. Results
Test set: 85 rows (20% of data). Training set: 337 rows (80%).
Metric	MLR	KNNR (k=5)
R²	0.8656	0.5448
RMSE	49.6242	91.3394
MAE	38.3124	70.4941
Train time (s)	0.1772	0.1486
Train CPU time (s)	0.1055	0.0834
Predict time (s)	0.1356	0.5634
Train peak memory (MB)	0.4218	0.2920
Predict peak memory (MB)	0.1000	0.2259
Charts: `metrics_comparison.png` (errors, R², times) and `actual_vs_predicted.png` (predicted vs actual yield).
Note: timing values change slightly on every run because they depend on the computer.
6. Discussion
Accuracy: MLR was clearly more accurate. It explained about 87% of the variation in yield (R² = 0.87) versus about 54% for KNNR, and its errors were smaller (RMSE 49.6 vs 91.3; MAE 38.3 vs 70.5).
Speed: Training times were similar (both under 0.2 s). KNNR was slower at prediction (0.56 s vs 0.14 s) because it must compare every test point with all training points.
Memory: KNNR used slightly less memory during training, but more during prediction. Differences are very small because the dataset is small.
Why MLR did better: yield in this dataset appears to depend fairly linearly on inputs such as fertilizer amount and acreage, which suits MLR. KNNR can be hurt when there are many input columns (especially the many one-hot encoded columns), because distances between points become less meaningful.
Conclusion: For this dataset MLR is the better choice: more accurate, faster at prediction, and easier to interpret.
7. How to run
Open Google Colab, upload `corn_data.csv` (folder icon on the left → upload).
Paste the code from `mlr_vs_knnr.py` into a cell and run it.
8. Files in this folder
`mlr_vs_knnr.py` – the Python code
`README.md` – this document
`comparison_results.csv`, `metrics_comparison.png`, `actual_vs_predicted.png` – outputs
