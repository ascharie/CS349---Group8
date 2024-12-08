class Node:
    def __init__(self, label = None, attribute = None, is_leaf = False):
        self.label = label
        self.children = {}
        self.attribute = attribute # best attribute to split on
        self.is_leaf = is_leaf
        
    def add_child(self, val, child):
        self.children[val] = child

    def show(self, level=0):
        indent = '     ' * level

        if not self.is_leaf:
            print(indent, 'attribute:', self.attribute)
            for value, child in self.children.items():
                print(indent, '--', value)
                child.show(level + 1)
        else:
            print(indent, 'label:', self.label)

import numpy as np
import pandas as pd

class Discretizer:
    def __init__(self, method="equal-width", n_bins=5, cols=None):
        self.method = method
        self.n_bins = n_bins
        self.cols = cols
        self.bins_ = {}

    def fit(self, X):
        self.bins_ = {}

        columns = self.cols
        
        for col in columns:
            if self.method == "equal-width":
                self.bins_[col] = np.linspace(X[col].min(), X[col].max(), self.n_bins + 1)
            elif self.method == "equal-frequency":
                self.bins_[col] = pd.qcut(X[col], q=self.n_bins, retbins=True, duplicates="drop")[1]
            else:
                raise ValueError("Invalid method or missing custom bins.")
    
    def transform(self, X):
        X_discretized = X.copy()

        columns = self.cols
        
        for col in columns:
            if col in self.bins_:
                X_discretized[col] = np.digitize(X[col], bins=self.bins_[col], right=False)
            else:
                raise ValueError(f"Binning not defined for column {col}. Did you forget to fit the discretizer?")
        
        return X_discretized

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

