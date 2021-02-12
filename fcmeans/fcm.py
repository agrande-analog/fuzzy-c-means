import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cdist

class FCM:
    def __init__(self, n_clusters=10, max_iter=150, m=2, error=1e-5, random_state=3):
        self.u, self.centers = None, None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state

    def fit(self, X):
        N = X.shape[0]
        C = self.n_clusters
        centers = []

        # u = np.random.dirichlet(np.ones(C), size=N)
        r = np.random.RandomState()
        u = r.rand(N,C)
        u = u / np.tile(u.sum(axis=1)[np.newaxis].T,C)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(X, u)
            u = self.next_u(X, centers)
            iteration += 1

            # Stopping rule
            if norm(u - u2) < self.error:
                break

        self.u = u
        self.centers = centers
        return self

    def fit_kpp(self, X):
        """
            Create cluster centroids using the k-means++ algorithm.
            Parameters
            ----------
            ds : numpy array
                The dataset to be used for centroid initialization.
            k : int
                The desired number of clusters for which centroids are required.
            Returns
            -------
            centroids : numpy array
                Collection of k centroids as a numpy array.
            Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
            """
        r0 = np.random.RandomState()
        centroids = [X[0]]

        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = r0.rand()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break

            centroids.append(X[i])

        C = np.array(centroids)

        u = self.next_u(X, C)
        centers = self.next_centers(X, u)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(X, u)
            u = self.next_u(X, centers)
            iteration += 1

            # Stopping rule
            if norm(u - u2) < self.error:
                break

        self.u = u
        self.centers = centers
        return self

    def next_centers(self, X, u):
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def next_u(self, X, centers):
        return self._predict(X, centers)

    def _predict(self, X, centers):
        power = float(2 / (self.m - 1))
        temp = cdist(X, centers) ** power
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_

        return np.nan_to_num(1 / denominator_.sum(2))

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        u = self._predict(X, self.centers)
        return np.argmax(u, axis=-1)
