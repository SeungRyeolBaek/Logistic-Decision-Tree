import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines
import warnings
from sklearn.exceptions import ConvergenceWarning

class Node:
    def __init__(self, *, feature_index=None, threshold=None, left=None, right=None,
                 value=None, impurity=None, samples=None):
        """
        A tree node for both classification and regression.

        Parameters:
        - feature_index: Index of the feature to split on.
        - threshold: Threshold value for the split.
        - left: Left child node.
        - right: Right child node.
        - value: Predicted value at the leaf node.
        - impurity: Impurity measure at the node.
        - samples: Number of samples at the node.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For classification: class counts; For regression: mean value
        self.impurity = impurity
        self.samples = samples
        
class LogisticDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        num_samples = y.size
        num_samples_per_class = [np.sum(y == c) for c in self.classes_]
        impurity = self._gini(y)
        node = Node(
            impurity=impurity,
            samples=num_samples,
            value=num_samples_per_class,
        )
        if self.max_depth == None:
            self.max_depth = num_samples
        if (depth < self.max_depth and num_samples >= self.min_samples_split and impurity > 0):
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
                return node
        # At leaf nodes, node.value remains as num_samples_per_class
        return node
    
    def _best_split(self, X, y):
        m, n_features = X.shape
        if m <= 1:
            return None, None
        best_idx, best_thr = None, None
        best_impurity = 1.0  # Maximum Gini impurity
        for idx in range(n_features):
            X_column = X[:, idx]
            logistic_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            try:
                logistic_model.fit(X_column.reshape(-1, 1), y)
            except Exception:
                continue  # Skip feature if logistic regression fails
            coef = logistic_model.coef_[0][0]
            intercept = logistic_model.intercept_[0]
            if coef == 0:
                continue  # Avoid division by zero
            threshold = -intercept / coef
            if np.isnan(threshold) or np.isinf(threshold):
                continue
            left_indices = X_column < threshold
            right_indices = ~left_indices
            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue
            y_left = y[left_indices]
            y_right = y[right_indices]
            impurity = self._weighted_gini(y_left, y_right)
            if impurity < best_impurity:
                best_impurity = impurity
                best_idx = idx
                best_thr = threshold
        return best_idx, best_thr
    
    def _gini(self, y):
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in self.classes_)
    
    def _weighted_gini(self, y_left, y_right):
        m = len(y_left) + len(y_right)
        gini_left = self._gini(y_left)
        gini_right = self._gini(y_right)
        return (len(y_left) * gini_left + len(y_right) * gini_right) / m
    
    def predict(self, X):
        X = check_array(X)
        return np.array([self._predict(inputs) for inputs in X])
    
    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        # Return the class with the most samples
        return self.classes_[np.argmax(node.value)]

class LogisticDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth=None, min_samples_split=2, random_state=None):
        """
        A decision tree regressor that uses logistic regression for splitting.

        Parameters:
        - max_depth: The maximum depth of the tree.
        - min_samples_split: The minimum number of samples required to split an internal node.
        - random_state: Controls the randomness of the estimator.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_2d=True)
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        num_samples = y.size
        max_depth = self.max_depth if self.max_depth is not None else np.inf
        value = np.mean(y)
        impurity = self._mse(y)
        node = Node(
            impurity=impurity,
            samples=num_samples,
            value=value,  # Node's mean target value
        )
        # Check split conditions
        if (depth < max_depth and num_samples >= self.min_samples_split and impurity > 0):
            idx, thr, split_impurity = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        m, n_features = X.shape
        if m <= 1:
            return None, None, None
        best_idx, best_thr = None, None
        best_impurity = np.inf

        # Calculate the median of y
        y_median = np.median(y)
        y_temp = (y >= y_median).astype(int)

        for idx in range(n_features):
            X_column = X[:, idx]
            logistic_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    logistic_model.fit(X_column.reshape(-1, 1), y_temp)
            except Exception:
                continue  # Skip feature if logistic regression fails

            coef = logistic_model.coef_[0][0]
            intercept = logistic_model.intercept_[0]
            if coef == 0:
                continue  # Avoid division by zero
            threshold = -intercept / coef
            if np.isnan(threshold) or np.isinf(threshold):
                continue

            left_indices = X_column < threshold
            right_indices = ~left_indices
            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue

            y_left = y[left_indices]
            y_right = y[right_indices]
            impurity = self._weighted_mse(y_left, y_right)

            if impurity < best_impurity:
                best_impurity = impurity
                best_idx = idx
                best_thr = threshold

        return best_idx, best_thr, best_impurity if best_idx is not None else None

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y)

    def _weighted_mse(self, y_left, y_right):
        m = len(y_left) + len(y_right)
        if m == 0:
            return 0
        mse_left = self._mse(y_left) * len(y_left)
        mse_right = self._mse(y_right) * len(y_right)
        return (mse_left + mse_right) / m  # Weighted average MSE

    def predict(self, X):
        X = check_array(X)
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value  # Leaf node's mean target value

def plot_decision_tree(model, filled=False, rounded=True, precision=3, fontsize=12, filename=None):
    """
    Plots the decision tree using matplotlib, ensuring nodes at the same level are spread out.
    """
    from collections import defaultdict

    # Helper function to get the total number of leaves under a node
    def get_num_leaves(node):
        if node.left is None and node.right is None:
            return 1
        num_leaves = 0
        if node.left:
            num_leaves += get_num_leaves(node.left)
        if node.right:
            num_leaves += get_num_leaves(node.right)
        return num_leaves

    # Helper function to assign x positions to nodes
    def assign_positions(node, current_depth=0, positions={}, x_offset=0):
        if node is None:
            return x_offset
        if current_depth not in positions:
            positions[current_depth] = []
        # Assign positions to left subtree
        x_offset = assign_positions(node.left, current_depth + 1, positions, x_offset)
        # Assign position to current node
        num_leaves = get_num_leaves(node)
        x = x_offset + num_leaves / 2
        positions[current_depth].append((x, node))
        x_offset += num_leaves
        # Assign positions to right subtree
        x_offset = assign_positions(node.right, current_depth + 1, positions, x_offset)
        return x_offset

    # Assign positions to all nodes
    positions = defaultdict(list)
    assign_positions(model.tree_, positions=positions)

    # Flatten positions and create a mapping from node to position
    node_positions = {}
    for depth, nodes in positions.items():
        for x, node in nodes:
            node_positions[node] = (x, -depth)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw nodes and edges
    for node, (x, y) in node_positions.items():
        # Draw edges
        if node.left:
            child_x, child_y = node_positions[node.left]
            ax.plot([x, child_x], [y, child_y], 'k-', linewidth=1)
        if node.right:
            child_x, child_y = node_positions[node.right]
            ax.plot([x, child_x], [y, child_y], 'k-', linewidth=1)

        # Prepare node label
        impurity_name = "gini" if hasattr(model, "classes_") else "mse"
        if isinstance(node.value, list):
            # 분류의 경우: 클래스별 샘플 수 표시
            value_text = f"value = {node.value}"
        else:
            # 회귀의 경우: 평균값 표시
            value_text = f"value = {node.value:.{precision}f}"
        label = (f"{impurity_name} = {node.impurity:.{precision}f}\n"
                 f"samples = {node.samples}\n"
                 f"{value_text}")
        if node.left is None and node.right is None:
            if filled:
                color = 'lightgreen'
            else:
                color = 'white'
        else:
            label = (f"x[{node.feature_index}] < {node.threshold:.{precision}f}\n" + label)
            if filled:
                color = 'lightblue'
            else:
                color = 'white'

        # Draw node box
        bbox_props = dict(boxstyle="round,pad=0.5", fc=color, ec='black', lw=1) if rounded else dict(boxstyle="square,pad=0.5", fc=color, ec='black', lw=1)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, bbox=bbox_props)

    # Adjust plot
    ax.axis('off')
    fig.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        
def hierarchy_pos(G, root, width=1.5, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Positions nodes in a hierarchy.

    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723

    Parameters:
    - G: The graph
    - root: The root node
    - width: Horizontal space allocated
    - vert_gap: Vertical gap between levels
    - vert_loc: Vertical location of root
    - xcenter: Horizontal location of root
    """
    def _hierarchy_pos(G, node, leftmost, rightmost, vert_loc, pos):
        children = list(G.successors(node))
        if not children:
            pos[node] = (leftmost, vert_loc)
            return leftmost, rightmost
        else:
            xmin = leftmost
            xmax = rightmost
            total_children = len(children)
            child_gap = (rightmost - leftmost) / total_children
            for i, child in enumerate(children):
                new_left = leftmost + i * child_gap
                new_right = new_left + child_gap
                xmid = (new_left + new_right) / 2
                pos[child] = (xmid, vert_loc - vert_gap)
                _hierarchy_pos(G, child, new_left, new_right, vert_loc - vert_gap, pos)
            pos[node] = ((leftmost + rightmost) / 2, vert_loc)
            return leftmost, rightmost
    pos = {}
    _hierarchy_pos(G, root, 0, width, vert_loc, pos)
    return pos