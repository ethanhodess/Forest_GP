from typing import Optional

class TreeNode:
    def __init__(self,
                 feature: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None,
                 value: Optional[int] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  

    def is_leaf(self) -> bool:
        return self.value is not None

    def height(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + max(self.left.height(), self.right.height())

    def predict_one(self, x):
        if self.is_leaf():
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left.predict_one(x)
        else:
            return self.right.predict_one(x)

    def count_leaves(self) -> int:
        if self.is_leaf():
            return 1
        return self.left.count_leaves() + self.right.count_leaves()
