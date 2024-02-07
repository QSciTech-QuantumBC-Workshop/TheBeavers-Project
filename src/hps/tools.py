from typing import Optional

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from umap import UMAP
import json
import os
import numpy as np


class UMAP_PCA_1d_REDUCER(BaseEstimator):
    def __init__(self):
        self.pca = PCA(n_components=1)
        self.umap = UMAP(n_components=2)

    def fit(self, X, y=None):
        self.umap.fit(X, y)
        self.pca.fit(self.umap.transform(X), y)
        return self

    def transform(self, X, y=None):
        return self.pca.transform(self.umap.transform(X))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        return self.umap.inverse_transform(self.pca.inverse_transform(X))


def to_json(obj, filename: Optional[str] = None):
    if isinstance(obj, dict):
        obj = {k: to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj = [to_json(v) for v in obj]
    elif isinstance(obj, tuple):
        obj = tuple(to_json(v) for v in obj)
    if hasattr(obj, "__json__"):
        data = obj.__json__()
    elif hasattr(obj, "to_json"):
        data = obj.to_json()
    elif isinstance(obj, np.ndarray):
        data = obj.tolist()
    else:
        data = obj
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            try:
                json.dump(data, f, indent=4)
            except Exception as e:
                print(f"Failed to dump {data} to {filename}: {e}")
    return data
