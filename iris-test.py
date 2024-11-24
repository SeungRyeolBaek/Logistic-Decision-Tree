import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from LogisticSplitDecisionTree import LogisticDecisionTreeClassifier, plot_decision_tree
import matplotlib.pyplot as plt

# Seed 고정
random.seed(0)
np.random.seed(0)

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Custom LogisticDecisionTreeClassifier
clf_logistic = LogisticDecisionTreeClassifier(random_state=0)

clf_logistic.fit(X_train, y_train)
logistic_score = clf_logistic.score(X_test, y_test)
print("Custom LogisticDecisionTreeClassifier score:", logistic_score)

# Original DecisionTreeClassifier for comparison
clf_original = DecisionTreeClassifier(random_state=0)

clf_original.fit(X_train, y_train)
original_score = clf_original.score(X_test, y_test)

print("Original DecisionTreeClassifier score:", original_score)

# Plotting the custom tree
plot_decision_tree(
    clf_logistic,
    filled=True,
    rounded=True,
    fontsize=10,
    filename="logistic_split_decision_tree_iris.png"
)

# Original DecisionTreeClassifier tree plot
plt.figure(figsize=(12, 8))
plot_tree(clf_original, filled=True)
plt.title("Original Decision Tree")
plt.savefig("original_decision_tree_iris.png")  # 파일로 저장
plt.close()