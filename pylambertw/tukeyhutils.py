
import logging
from functools import partial

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import lambertw
from scipy.stats import norm as normaldist
from scipy.stats import median_abs_deviation
import numba as nb


@nb.njit(nb.float64[:](nb.float64[:], nb.float64))
def tukeyh(x, h):
    return x*np.exp(0.5*h*x*x)


def lambertWdelta(z, delta):
    if delta != 0.0:
        return np.sign(z) * np.sqrt(np.real(lambertw(delta*z*z, 0))/delta)
    else:
        return z


@nb.njit(nb.float64[:](nb.float64[:], nb.float64, nb.float64, nb.float64))
def f2heavytail(u, delta, mux, sigmax):
    return tukeyh(u, delta)*sigmax + mux


def heavytail2f(z, delta, mux=0.0, sigmax=1.0):
    return lambertWdelta((z-mux)/sigmax, delta)*sigmax + mux


def lambertWgaussiandist(y, delta, mux=0.0, sigmax=1.0):
    z = (y-mux) / sigmax
    factor1 = normaldist.pdf(lambertWdelta(z, delta)*sigmax+mux)
    factor2 = lambertWdelta(z, delta) / z / (1 + np.real(lambertw(delta*z*z)))
    return np.sum(factor1*factor2)


def derivative_lambertWgaussianMLE_z(zarray, delta):
    wdfcn = np.vectorize(partial(lambertWdelta, delta=delta))
    wdvaluessq = wdfcn(zarray)*wdfcn(zarray)
    denominator = 1 + np.real(lambertw(delta*zarray*zarray, 0))
    return np.sum(
        wdvaluessq/denominator * (0.5*wdvaluessq - 0.5 - 1./denominator)
    )


def find_delta_gradient_descent(zarray, learningrate=1e-6, delta0=0.5, tol=1e-6, maxnbepochs=10000):
    intresults = []

    delta = delta0
    for i in range(maxnbepochs):
        change = -learningrate * derivative_lambertWgaussianMLE_z(zarray, delta)
        if abs(change) < tol:
            break
        delta -= change

        intresults.append((i, delta))

    return delta, intresults


@nb.njit(nb.float64(nb.float64[:]))
def compute_kurtosis(x):
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    y = np.square(x-mean)
    return np.sum(np.square(y)) / (n * std*std*std*std)


def compute_delta_Taylor(z):
    kurtosis = compute_kurtosis(z)
    disc = 66*kurtosis - 162
    if disc > 0:
        return (np.sqrt(disc) - 6) / 66.
    else:
        return 0.


def compute_delta_GMM(z, kurtosis, initial_delta=None, tol=1e-7):
    if initial_delta is None:
        initial_delta = compute_delta_Taylor(z)
    initial_delta = np.array([initial_delta])
    f = lambda delta: np.abs(
            compute_kurtosis(lambertWdelta(z, delta)) - kurtosis
        )
    f = np.vectorize(f)

    sol = minimize(
        f,
        initial_delta,
        bounds=Bounds(lb=tol, ub=np.inf)
    )
    return sol.x[0]


def IGMM(y, kurtosis, tol=1e-7, maxnpepochs=10000):
    # delta = compute_delta_Taylor(y)
    mu = np.median(y)
    # std = np.std(y) * (1 - 2 * delta) ** 3
    std = median_abs_deviation(y) + 1e-3 * np.std(y)
    delta = compute_delta_Taylor((y-mu)/std)

    prev_delta, prev_mu, prev_std = delta + 2 * tol, mu + 2 * tol, std + 2 * tol

    k = 0
    while np.abs(delta - prev_delta) + np.abs(mu - prev_mu) + np.abs(std - prev_std) > tol and k < maxnpepochs:
        logging.info('{}  {}  {}'.format(mu, std, delta))
        z = (y - mu) / std
        prev_mu, prev_std, prev_delta = mu, std, delta
        delta = compute_delta_GMM(z, kurtosis, initial_delta=prev_delta)
        u = lambertWdelta(z, delta)
        x = u * std + mu
        mu, std = np.mean(x), np.std(x)
        k += 1

    return mu, std, delta
