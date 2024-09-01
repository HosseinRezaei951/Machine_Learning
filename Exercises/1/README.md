# Polynomial Regression and Model Complexity Visualization

This project demonstrates the effect of increasing the polynomial degree on model complexity and its impact on training and testing data. The exercise involves fitting polynomial regression models with varying degrees to a dataset, predicting outputs, and visualizing the results. The exercise aims to illustrate how the complexity of the model changes with the polynomial degree, potentially leading to underfitting or overfitting.

## Overview

In this exercise, polynomial regression models of varying degrees (from 1 to 9) are applied to a dataset. The models are trained using different subsets of data, and their performance is evaluated on a test set. The results are visualized using scatter plots and prediction lines, allowing for an intuitive understanding of how model complexity influences performance.

### Code Breakdown

The code performs the following steps for each polynomial degree:

1. **Data Generation**:
   - **Training Data**: Random samples from a uniform distribution, scaled to a range between 0 and 1.
   - **Testing Data**: A separate set of random samples, also scaled.

2. **Polynomial Feature Expansion**:
   - For each polynomial degree, the input data is expanded using polynomial features up to that degree.

3. **Model Training**:
   - A polynomial regression model is trained using the expanded features and the training data.
   - A small regularization term (`lambda = 0.001`) is used to prevent overfitting.

4. **Prediction**:
   - The trained model is used to predict outputs for the test data.

5. **Visualization**:
   - Scatter plots of the training and testing data are generated.
   - The model's prediction line is plotted for each polynomial degree.

6. **Saving Results**:
   - The plots are saved as images in the `Results` directory.

### Key Components

- **Phi_1**: Linear model (degree 1).
- **Phi_2 to Phi_9**: Polynomial models of increasing complexity, from degree 2 to degree 9.

### Example Plot

Each plot shows:
- **Blue Points**: Testing data.
- **Red Points**: Training data.
- **Green Line**: Prediction line for the given polynomial degree.

## How to Run the Code

1. **Dependencies**:
   - Ensure you have `matplotlib` and `numpy` installed.
   - Install dependencies via pip if necessary:
     ```
     pip install matplotlib numpy
     ```

2. **Execution**:
   - Run the Python script.
   - The script will generate and save a series of plots in the `Results` directory.

3. **Results**:
   - Navigate to the `Results` directory to view the saved plots. Each plot represents a different polynomial degree and training size.

## Conclusion

This exercise provides hands-on experience with polynomial regression, demonstrating how model complexity affects performance on both training and test datasets. By visually analyzing the results, one can better understand the trade-offs between underfitting and overfitting as the model's complexity increases.