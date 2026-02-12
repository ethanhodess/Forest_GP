from typing import Optional, Union, Dict, List
import numpy as np
from TreeNode import TreeNode
from collections import Counter
import random
import math
import copy

class DecisionTree:

    # allowed sets
    _allowed_max_features: List[Union[str, None]] = ["sqrt", "log2", None]
    _allowed_criteria: List[str] = ["gini", "entropy", "log_loss"]
    _allowed_class_weights: List[Union[str, None]] = [None, "balanced"]

    def __init__(
        self,
        max_depth_gene: float,
        min_samples_split_gene: float,
        min_samples_leaf_gene: float,
        min_impurity_gene: float,
        n_features: int,
        n_classes: int,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        max_leaf_nodes: Optional[int] = None,
        class_weight: Optional[Union[Dict[int, float], str]] = None,
        criterion: str = "gini",
        mutation_rate: float = 0.1
    ):
        # numeric 
        self.max_depth_gene = float(max_depth_gene)
        self.min_samples_split_gene = float(min_samples_split_gene)
        self.min_samples_leaf_gene = float(min_samples_leaf_gene)
        self.min_impurity_gene = float(min_impurity_gene)

        # categorical 
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.criterion = criterion

        # fitted tree root and sample indices
        self.root: Optional[TreeNode] = None
        self.sample_indices: Optional[np.ndarray] = None

        self.n_features = n_features
        self.n_classes = n_classes
        self._leaf_count = 0
        self.mutation_rate = mutation_rate

    # custom decode
    def _decode_hyperparams(self, n_samples: int):
        max_depth = int(2 + self.max_depth_gene * 10)          # [2,12]
        min_samples_leaf = int(1 + self.min_samples_leaf_gene * 30)  # [1,31]
        min_samples_split = max(2, int(2 * self.min_samples_split_gene * 50))
        min_impurity_decrease = self.min_impurity_gene * 0.05

        return (
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            self.max_features,
            self.max_leaf_nodes,
            self.class_weight,
            self.criterion
        )

    # --- fit method ---
    def fit(self, X: np.ndarray, y: np.ndarray, use_indices: Optional[np.ndarray] = None):
        n_total = X.shape[0]

        idxs = np.array(use_indices, dtype=int) if use_indices is not None else np.random.choice(n_total, size=n_total, replace=True)
        self.sample_indices = idxs

        X_sub, y_sub = X[idxs], y[idxs]

        max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, \
        max_features, max_leaf_nodes, class_weight, criterion = self._decode_hyperparams(n_total)

        self._leaf_count = 0  # reset leaf counter
        self.root = self._build_tree(
            X_sub, y_sub, max_depth, X.shape[1], self.n_classes,
            depth=0,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            criterion=criterion
        )

    # --- predict ---
    def predict(self, X):
        if self.root is None:
            raise ValueError("Tree not fitted")
        return np.array([self.root.predict_one(x) for x in X])

    def height(self) -> int:
        return self.root.height() if self.root else 0

    def num_leaves(self) -> int:
        return self.root.count_leaves() if self.root else 0

    # --- build tree (supports all criteria, max_features, class_weight) ---
    def _build_tree(
        self,
        X, y,
        max_depth: int,
        n_features: int,
        n_classes: int,
        depth=0,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        class_weight=None,
        criterion="gini"
    ):
        # compute global weights
        if class_weight == "balanced":
            counts = Counter(y)
            total = len(y)
            class_weights = {c: total / (len(counts) * cnt) for c, cnt in counts.items()}
            sample_weight = np.array([class_weights[yi] for yi in y])
        else:
            sample_weight = np.ones(len(y))

        # terminal conditions
        if len(y) == 0 or len(set(y)) == 1 or depth >= max_depth or len(y) < min_samples_split or (max_leaf_nodes and self._leaf_count >= max_leaf_nodes):
            self._leaf_count += 1
            return TreeNode(value=Counter(y).most_common(1)[0][0] if len(y) else 0)

        def impurity(y_local, w_local):
            total_weight = w_local.sum()
            if total_weight == 0:
                return 0.0

            counts = {}
            for yi, wi in zip(y_local, w_local):
                counts[yi] = counts.get(yi, 0.0) + wi

            probs = np.array(list(counts.values())) / total_weight

            if criterion == "gini":
                return 1.0 - np.sum(probs ** 2)
            elif criterion == "entropy":
                return -np.sum(probs * np.log2(probs + 1e-12))
            elif criterion == "log_loss":
                return -np.sum(probs * np.log(probs + 1e-12))
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

        parent_impurity = impurity(y, sample_weight)

        # max_features handling
        if max_features is None:
            k = n_features
        elif max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(max_features, float):
            k = max(1, int(max_features * n_features))
        else:
            k = int(max_features)

        feature_subset = random.sample(range(n_features), k)

        # find best split
        best_feat = best_thresh = None
        best_impurity_after = None

        for f in feature_subset:
            values = np.unique(X[:, f])
            if len(values) < 2:
                continue

            # midpoints
            thresholds = (values[:-1] + values[1:]) / 2.0

            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask
                if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                    continue

                w_left = sample_weight[left_mask]
                w_right = sample_weight[right_mask]

                impurity_after = (
                    (w_left.sum() / sample_weight.sum()) * impurity(y[left_mask], w_left)
                    + (w_right.sum() / sample_weight.sum()) * impurity(y[right_mask], w_right)
                )

                decrease = parent_impurity - impurity_after
                if decrease <= min_impurity_decrease:
                    continue

                if best_feat is None or impurity_after < best_impurity_after:
                    best_feat, best_thresh, best_impurity_after = f, t, impurity_after

        if best_feat is None:
            self._leaf_count += 1
            return TreeNode(value=Counter(y).most_common(1)[0][0])

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_node = self._build_tree(
            X[left_mask], y[left_mask], max_depth, n_features, n_classes,
            depth+1, min_samples_split, min_samples_leaf, min_impurity_decrease,
            max_features, max_leaf_nodes, class_weight, criterion
        )
        right_node = self._build_tree(
            X[right_mask], y[right_mask], max_depth, n_features, n_classes,
            depth+1, min_samples_split, min_samples_leaf, min_impurity_decrease,
            max_features, max_leaf_nodes, class_weight, criterion
        )

        return TreeNode(best_feat, best_thresh, left_node, right_node)


    def _mutate_from_parent(self, parent):
        child = copy.deepcopy(parent)
        bootstrap_mutate = (random.random() < self.mutation_rate)

        # numeric genes
        for attr in ["max_depth_gene", "min_samples_split_gene", "min_samples_leaf_gene", "min_impurity_gene"]:
            if random.random() < self.mutation_rate:
                old = getattr(child, attr)
                new = np.clip(old + np.random.normal(0.0, 0.1), 0.0, 1.0)
                setattr(child, attr, float(new))

        # integer gene
        if random.random() < self.mutation_rate and child.max_leaf_nodes is not None:
            delta = random.randint(-3, 3)
            child.max_leaf_nodes = max(1, child.max_leaf_nodes + delta)

        # categorical genes (coin-flip)
        if random.random() < self.mutation_rate:
            child.criterion = random.choice(self._allowed_criteria)
        if random.random() < self.mutation_rate:
            child.max_features = random.choice(self._allowed_max_features)
        if random.random() < self.mutation_rate:
            child.class_weight = random.choice(self._allowed_class_weights)

        return child, bootstrap_mutate
