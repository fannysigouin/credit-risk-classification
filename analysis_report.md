# Module 20 Challenge: Analysis Report

## Overview of the Analysis

The purpose of this analysis was to produce a machine learning model capable of identiying the creditworthiness of borrowers, by segmenting loans as 'healthy' or 'high-risk'. This was accomplished using a dataset of historical lending activity from a peer-to-peer lending services company. The dataset, `lending_data.csv`, contained multiple data points such as the size of the loan, the interest rate, the borrower's income, their debt-to-income ratio, their number of accounts, the number of derogatory marks on their account, their total debt, and the status of their loan (0 meaning healthy and 1 meaning high-risk). Upon initially reviewing the dataset, specifically the loan status, 75,000 healthy loans and 2,500 high-risk loans were identified.

The analysis began first by reading the data from the CSV file into a pandas dataframe. After separating the target variable (loan status) from the dataset's features, the data was split into training and testing datasets using the `train_test_split` model from `sklearn`. Then, a `LogisticRegression` model, also from `sklearn`, was instantiated and fitted with the training data. Predictions of loan status were then made using the testing data.

The last step of the analysis was to calculate the balanced accuracy score of the model as well as generate a confusion matrix and a classification report to evaluate the model's performance. 

## Results

The model has a fairly high balanced accuracy score of 0.95. The classification report produced indicates that the model is able to predict both types of loans (healthy and high-risk) with a high degree of precision and accuracy. Healthy loans (`0`) have a very high accuracy and recall of 1.0 and 0.99, while high-risk loans (`1`) have slightly lower accuracy and recall of 0.85 and 0.91.

## Summary

Overall, the model performs well and is able to predict both types of loans with a high degree of accuracy and precision, although it is better at predicting healthy loans than high-risk loans. However, I would still recommend using this model as it has a very high overall accuracy of 0.99.
