# Bayesian Linear Regression with Polynomial Features and Uncertainty Visualization

This exercise applies Bayesian linear regression to a sinusoidal dataset using polynomial basis expansion. The objective is to explore how model complexity affects predictions and the associated uncertainty by varying the degree of polynomial features. The exercise demonstrates the predictive performance and uncertainty estimation of the model for different levels of polynomial expansion, with visualizations that highlight the model's confidence in its predictions.

## Overview

The code implements Bayesian linear regression with polynomial basis expansions of degree 3, 5, and 10. The model is trained on synthetic sinusoidal data, and its predictions are evaluated on a separate test dataset. The uncertainty in the predictions is quantified and visualized, providing insight into how confident the model is across different polynomial complexities.


### Code Breakdown

The code performs the following steps for each polynomial degree (3, 5, and 10) and for various training set sizes:

1. **Data Generation**:
   - **Training Data**: A small subset of random samples (sizes 3, 5, and 10) is selected from a uniform distribution between 0 and 1. The sinusoidal target values are generated with some added noise.
   - **Testing Data**: A larger set of samples is generated similarly for evaluating the model.

2. **Polynomial Feature Expansion**:
   - Polynomial features are created for each degree (3, 5, and 10), allowing the model to capture increasingly complex relationships in the data.

3. **Bayesian Linear Regression**:
   - Bayesian linear regression is performed by computing the posterior distribution of the model parameters, incorporating prior assumptions about the parameter distribution.
   - The model predicts the mean output and calculates the uncertainty (standard deviation) for each test point.

4. **Prediction and Uncertainty Estimation**:
   - For each polynomial degree, the model’s mean prediction and uncertainty intervals (±3 standard deviations) are calculated and visualized.
   - The uncertainty intervals provide a range within which the true output is expected to lie, reflecting the model's confidence.

5. **Visualization**:
   - Three subplots are created for each degree (Phi_3, Phi_5, Phi_10). Each plot includes:
     - **Training Data**: Black scatter points.
     - **True Function**: Green line representing the actual sinusoidal function.
     - **Mean Prediction**: Blue line showing the model’s predicted mean.
     - **Uncertainty Interval**: Red lines indicating the prediction intervals (mean ± 3 standard deviations).

6. **Saving Results**:
   - The plots are saved as images in the `Results` directory, with filenames indicating the training set size and iteration number.

### Key Components

- **Polynomial Degrees**:
  - **Phi_3**: A cubic polynomial model.
  - **Phi_5**: A quintic polynomial model.
  - **Phi_10**: A decic polynomial model, capturing even more complex relationships.
  
- **Uncertainty Visualization**:
  - The red bands around the mean predictions illustrate where the model is less certain about its predictions, highlighting areas with higher or lower confidence.

## How to Run the Code

1. **Dependencies**:
   - Ensure you have the following libraries installed:
     ```
     pip install matplotlib numpy scikit-learn
     ```

2. **Execution**:
   - Run the Python script.
   - The script will generate and save a series of plots in the `Results` directory, corresponding to different polynomial degrees and training set sizes.

3. **Results**:
   - Navigate to the `Results` directory to view the saved plots. Each set of plots allows you to compare the model’s predictions and uncertainties across different polynomial complexities and training data sizes.

## Conclusion

This exercise provides hands-on experience with Bayesian linear regression and uncertainty quantification, illustrating how polynomial degree impacts model complexity and confidence. The visualizations help to understand the trade-offs involved in choosing model complexity, with higher degrees providing more flexibility but also potentially more uncertainty in regions with sparse data.