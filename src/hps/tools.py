from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from umap import UMAP


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


