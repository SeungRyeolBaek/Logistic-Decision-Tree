import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from LogisticSplitDecisionTree import LogisticDecisionTreeClassifier, plot_decision_tree
import matplotlib.pyplot as plt

# 시드 고정
random.seed(0)
np.random.seed(0)

# 데이터 로드
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names  # 특성 이름 가져오기

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 커스텀 LogisticDecisionTreeClassifier
clf_logistic = LogisticDecisionTreeClassifier(random_state=0)

clf_logistic.fit(X_train, y_train)
logistic_score = clf_logistic.score(X_test, y_test)
print("Custom LogisticDecisionTreeClassifier score:", logistic_score)

# 기존 DecisionTreeClassifier와 비교
clf_original = DecisionTreeClassifier(random_state=0)

clf_original.fit(X_train, y_train)
original_score = clf_original.score(X_test, y_test)

print("Original DecisionTreeClassifier score:", original_score)

# 커스텀 트리 시각화
plot_decision_tree(
    clf_logistic,
    filled=True,
    rounded=True,
    fontsize=10,
    filename="logistic_split_decision_tree_breast_cancer.png"
)

# Original DecisionTreeClassifier 트리 시각화
plt.figure(figsize=(20, 10))
plot_tree(clf_original, filled=True, feature_names=feature_names, class_names=data.target_names)
plt.title("Original Decision Tree (Breast Cancer Dataset)")
plt.savefig("original_decision_tree_breast_cancer.png")
plt.close()