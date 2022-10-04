
from .tukeyhutils import IGMM, heavytail2f


class GaussianLamberter:
    def __init__(self, mu=0.0, sigma=1.0, delta=0.25):
        self.mu = mu
        self.sigma = sigma
        self.delta = delta
        self.nbsteps = None

    def fit(self, X, maxnbepochs=100000):
        mu, std, delta, nbsteps = IGMM(X, 3.0, maxnpepochs=maxnbepochs)
        self.mu = mu
        self.sigma = std
        self.delta = delta
        self.nbsteps = nbsteps
        return self

    def transform(self, X):
        return heavytail2f(X, self.delta, mux=self.mu, sigmax=self.sigma)

    def fit_transform(self, X, maxnbepochs=100000):
        self.fit(X, maxnbepochs=maxnbepochs)
        return self.transform(X)
