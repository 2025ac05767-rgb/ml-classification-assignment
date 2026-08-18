# Breast Cancer Classification Using Machine Learning

## A. Problem Statement

The objective of this project is to predict whether a breast tumor is malignant or benign  build and compare multiple machine learning classification models

Five classification algorithms are implemented and evaluated using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

The models are compared using Accuracy, AUC, Precision, Recall, F1 Score, and
Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is also developed to allow test data to be
uploaded and evaluated using any of the trained models.


## B. Dataset Description

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset

Dataset used through: scikit-learn `load_breast_cancer()`
Original source: UCI Machine Learning Repository

The dataset contains measurements computed from digitized images of fine needle
aspirates of breast masses. These measurements describe characteristics of the
cell nuclei present in the images.

Number of instances: 569

Number of input features: 30

Classification type: Binary classification

Target classes:

- Malignant
- Benign

The 30 numerical features include measurements related to radius, texture,
perimeter, area, smoothness, compactness, concavity, symmetry and other
characteristics.

The dataset was divided into training and testing sets using an 80:20 split.
Stratified sampling was used to approximately preserve the class distribution
in both sets.

StandardScaler was applied for Logistic Regression and kNN because these
algorithms are affected by differences in feature scales.


## C. GitHub Repository

GitHub Repository:
https://github.com/2025ac05767-rgb/ml-classification-assignment/tree/main/ML_Assignment_2




## Live Streamlit Application

Streamlit Application:
https://ml-classification-assignment-pa9pcggssvtowj6cddnasx.streamlit.app/


## D. Models Used and Performance Comparison

The following five machine learning classification models were trained on the
same dataset and evaluated on the same test set.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |


## E. Observations
| ML Model                 | Observation about Model Performance                                                                                                                                                 |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**  | Logistic Regression performed the best among all the models. It achieved 98.25% accuracy and also had the highest F1 and MCC scores.                                                |
| **Decision Tree**        | Decision Tree had the lowest accuracy among the five models at 92.11%. It performed reasonably well, but the other models gave better results on this dataset.                      |
| **kNN**                  | kNN performed well with 95.61% accuracy and 97.22% recall. Since kNN uses distance between data points, I used feature scaling before training the model.                           |
| **Gaussian Naive Bayes** | Naive Bayes achieved 93.86% accuracy. Even though it makes a simple assumption that features are independent for a given class, it still performed fairly well on this dataset.     |
| **Random Forest**        | Random Forest achieved 95.61% accuracy and performed better than the Decision Tree. Using multiple decision trees instead of a single tree helped it give more reliable results.    |
| **Overall Winner**       | Logistic Regression is the best model for this dataset based on my results. It achieved the highest accuracy of 98.25% and also performed well across the other evaluation metrics. |



## Streamlit Application Features

The deployed Streamlit application provides:

- CSV test data upload
- Selection between all five trained classification models
- Accuracy, AUC, Precision, Recall, F1 Score and MCC
- Confusion Matrix
- Classification Report
- Display of uploaded test data


## Project Structure

    ML_Assignment_2/
    |
    |-- app.py
    |-- train_models.py
    |-- requirements.txt
    |-- README.md
    |-- test_data.csv
    |
    `-- model/
        |-- logistic_regression.pkl
        |-- decision_tree.pkl
        |-- knn.pkl
        |-- naive_bayes.pkl
        |-- random_forest.pkl
        `-- scaler.pkl


## How to Run the Application

Install the required dependencies:

    pip install -r requirements.txt

Run the Streamlit application:

    streamlit run app.py

Upload `test_data.csv` through the application and select a model from the
dropdown to view its evaluation results.