
from .tukeyhutils import IGMM, heavytail2f


class GaussianLamberter:
    def __init__(self, mu=0.0, sigma=1.0, delta=0.25):
        self.mu = mu
        self.sigma = sigma
        self.delta = delta
        self.nbsteps = None

    def fit(self, X, maxnbepochs=100000):
        mu, std, delta, nbsteps = IGMM(X, 3.0, maxnpepochs=maxnbepochs, returnnbsteps=True)
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

    def __str__(self):
        displayedstr = 'GaussianLamberter: mu={}, sigma={}, delta={}; (number of steps: {})'.format(
            self.mu,
            self.sigma,
            self.delta,
            'None' if self.nbsteps is None else self.nbsteps
        )
        return displayedstr

    def __repr__(self):
        return super(GaussianLamberter, self).__repr__() + '\n' + self.__str__()
