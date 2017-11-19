# Project 1: Pattern Classification and Machine Learning

One Paragraph of project description goes here

## Data filtering

The outliers have been replaced by 0 for all the data : test and training sets. 

## Features

For each feature we generate a polynomial of degree 9 using the function build_poly in scripts/algorithms.py

## Structure

### data

Put the train.csv and test.csv in the data folder.

### implementations.py

Contains all 6 methods:

least_squares_GD : (Linear Regression using Gradient Descent)

least_squares_SGD : (Linear Regression using Stochastic Gradient Descent)

least_squares : (Linear Regression using Normal Equations)

ridge_regression : (Ridge Regression using Normal Equations)

logistic_regression : (Logistic Regression using Gradient Descent)

reg_logistic_regression : (Regularized Logistic Regression using Gradient Descent)


### run.py

Uses ridge regression to create the submission : Go to the scripts folder and use the command "python run.py" in the terminal.

You must have python 3 in order for the code to work correctly.

### proj1_helpers.py

Contains functions to load the data and create the predictions/submission.

### algorithms.py

Contains utilitary functions such as build_poly to create the polynomials used for regression.

### Group members:

Skander Hajri

Guillaume Mollard

Adrien Ruault

### Team name on Kaggle : "?"