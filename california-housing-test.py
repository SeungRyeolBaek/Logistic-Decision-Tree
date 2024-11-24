import numpy as np
import random
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from LogisticSplitDecisionTree import LogisticDecisionTreeRegressor
import matplotlib.pyplot as plt

# Fix the seed for reproducibility
random.seed(0)
np.random.seed(0)

# Load California Housing data
data = fetch_california_housing()
X, y = data.data, data.target

# Split the data once, to use the same test set for all models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# List of max_depth values to test
max_depth_list = [2,4,6,8,10,12]

# Initialize dictionaries to store results
logistic_scores = {}
original_scores = {}
logistic_depths = {}
original_depths = {}

# Iterate over different max_depth values
for depth in max_depth_list:
    # Custom LogisticDecisionTreeRegressor
    reg_logistic = LogisticDecisionTreeRegressor(max_depth=depth, random_state=0)
    reg_logistic.fit(X_train, y_train)
    print("LogisticDecisionTreeRegressor done")
    logistic_score = reg_logistic.score(X_test, y_test)
    logistic_scores[depth] = logistic_score

    # Original DecisionTreeRegressor
    reg_original = DecisionTreeRegressor(max_depth=depth, random_state=0)
    reg_original.fit(X_train, y_train)
    print("DecisionTreeRegressor done")
    original_score = reg_original.score(X_test, y_test)
    original_scores[depth] = original_score

    # Calculate actual tree depth for max_depth=None
    if depth is None:
        original_depths[depth] = reg_original.get_depth()

    # Predict on the test set
    y_pred_logistic = reg_logistic.predict(X_test)
    y_pred_original = reg_original.predict(X_test)

    # Plotting predicted vs actual values
    plt.figure(figsize=(12, 6))

    # Scatter plot for LogisticDecisionTreeRegressor
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_logistic, alpha=0.5, color='blue')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'LogisticDecisionTreeRegressor\nmax_depth={depth}, R²={logistic_score:.4f}')

    # Scatter plot for DecisionTreeRegressor
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_original, alpha=0.5, color='green')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'DecisionTreeRegressor\nmax_depth={depth}, R²={original_score:.4f}')

    plt.tight_layout()
    plt.savefig(f'california-housing-result/comparison_regression_depth_{depth}.png')
    plt.close()

# Print R² scores
print("R² Scores for LogisticDecisionTreeRegressor:")
for depth, score in logistic_scores.items():
    print(f"max_depth={depth}, R²={score:.4f}")

print("\nR² Scores for DecisionTreeRegressor:")
for depth, score in original_scores.items():
    print(f"max_depth={depth}, R²={score:.4f}")
