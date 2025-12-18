from typing import Optional
import numpy as np
from TreeNode import TreeNode
from collections import Counter
import random

class DecisionTree:
    def __init__(self,
                 max_depth_pct: float,
                 min_samples_split_pct: float,
                 min_samples_leaf_pct: float,
                 min_impurity_decrease: float,
                 n_features: int,
                 n_classes: int):
        # store hyperparams as floats (percent-of-total-samples for integer hyperparams)
        self.max_depth_pct = float(max_depth_pct)
        self.min_samples_split_pct = float(min_samples_split_pct)
        self.min_samples_leaf_pct = float(min_samples_leaf_pct)
        self.min_impurity_decrease = float(min_impurity_decrease)

        # fitted tree root and the sample indices used for its last fit
        self.root: Optional[TreeNode] = None
        self.sample_indices: Optional[np.ndarray] = None

        self.n_features = n_features
        self.n_classes = n_classes

    # convert percent hyperparams to ints using total_samples
    def _compute_int_hyperparams(self, total_samples: int):
        max_depth = max(1, int(round(self.max_depth_pct * total_samples)))
        min_samples_split = max(2, int(round(self.min_samples_split_pct * total_samples)))
        min_samples_leaf = max(1, int(round(self.min_samples_leaf_pct * total_samples)))
        return max_depth, min_samples_split, min_samples_leaf

    def fit(self, X: np.ndarray, y: np.ndarray, use_indices: Optional[np.ndarray] = None):
        n_total = X.shape[0]

        # determine sample indices to use (store them as row indices into X/y)
        if use_indices is not None:
            idxs = np.array(use_indices, dtype=int)
        else:
            idxs = np.random.choice(n_total, size=n_total, replace=True)

        self.sample_indices = idxs  # store bootstrap row indices

        X_sub = X[idxs]
        y_sub = y[idxs]

        # compute integer hyperparams from percent representations
        max_depth_int, min_samples_split, min_samples_leaf = self._compute_int_hyperparams(n_total)

        # build tree using stored sample
        self.root = self._build_tree(X_sub, y_sub, max_depth_int, X.shape[1], self.n_classes,
                                     depth=0,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     min_impurity_decrease=self.min_impurity_decrease)

    def predict(self, X):
        if self.root is None:
            raise ValueError("Tree not fitted")
        return np.array([self.root.predict_one(x) for x in X])

    def height(self) -> int:
        if self.root is None:
            return 0
        return self.root.height()

    def num_leaves(self) -> int:
        if self.root is None:
            return 0
        return self.root.count_leaves()

    # --- CART style helpers ---
    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        counts = np.array(list(Counter(y).values()))
        p = counts / counts.sum()
        return 1.0 - np.sum(p ** 2)

    @staticmethod
    def _impurity_after_split(left_y, right_y):
        n = len(left_y) + len(right_y)
        if n == 0:
            return 0.0
        return (len(left_y) / n) * DecisionTree._gini(left_y) + (len(right_y) / n) * DecisionTree._gini(right_y)

    def _build_tree(self, X, y, max_depth: int, n_features: int, n_classes: int,
                    depth=0, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0):
        # terminal conditions
        if len(y) == 0:
            return TreeNode(value=0)
        if len(set(y)) == 1:
            return TreeNode(value=Counter(y).most_common(1)[0][0])
        if depth >= max_depth or len(y) < min_samples_split:
            return TreeNode(value=Counter(y).most_common(1)[0][0])

        parent_gini = DecisionTree._gini(y)
        feature_subset = random.sample(range(n_features), max(1, int(np.sqrt(n_features))))

        best_feat = None
        best_thresh = None
        best_impurity = None

        for f in feature_subset:
            thresh_candidates = np.unique(X[:, f])
            if len(thresh_candidates) == 0:
                continue
            # limit threshold candidates for speed
            if len(thresh_candidates) > 20:
                thresh_candidates = np.random.choice(thresh_candidates, 20, replace=False)
            for t in thresh_candidates:
                left_mask = X[:, f] <= t
                right_mask = X[:, f] > t
                if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                    continue
                impurity_after = DecisionTree._impurity_after_split(y[left_mask], y[right_mask])
                impurity_decrease = parent_gini - impurity_after
                if impurity_decrease <= min_impurity_decrease:
                    continue
                if best_feat is None or impurity_after < best_impurity:
                    best_feat = f
                    best_thresh = t
                    best_impurity = impurity_after

        if best_feat is None:
            return TreeNode(value=Counter(y).most_common(1)[0][0])

        left_node = self._build_tree(X[X[:, best_feat] <= best_thresh], y[X[:, best_feat] <= best_thresh],
                                     max_depth, n_features, n_classes, depth+1,
                                     min_samples_split, min_samples_leaf, min_impurity_decrease)
        right_node = self._build_tree(X[X[:, best_feat] > best_thresh], y[X[:, best_feat] > best_thresh],
                                      max_depth, n_features, n_classes, depth+1,
                                      min_samples_split, min_samples_leaf, min_impurity_decrease)

        return TreeNode(best_feat, best_thresh, left_node, right_node)
