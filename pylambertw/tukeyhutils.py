
import logging
from functools import partial
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds
from scipy.special import lambertw
from scipy.stats import norm as normaldist
from scipy.stats import median_abs_deviation
import numba as nb


@nb.njit(nb.float64[:](nb.float64[:], nb.float64))
def tukeyh(x: npt.NDArray[np.float64], h: float) -> npt.NDArray[np.float64]:
    return x*np.exp(0.5*h*x*x)


@nb.njit(nb.float64[:](nb.float64[:], nb.float64))
def lambertWdelta(z: npt.NDArray[np.float64], delta: float) -> npt.NDArray[np.float64]:
    if delta != 0.0:
        return np.sign(z) * np.sqrt(np.real(lambertw(delta*z*z, 0))/delta)
    else:
        return z


@nb.njit(nb.float64[:](nb.float64[:], nb.float64, nb.float64, nb.float64))
def f2heavytail(u: npt.NDArray[np.float64], delta: float, mux: float, sigmax: float) -> npt.NDArray[np.float64]:
    return tukeyh(u, delta)*sigmax + mux


@nb.njit(nb.float64[:](nb.float64[:], nb.float64, nb.float64, nb.float64))
def heavytail2f(z: npt.NDArray[np.float64], delta: float, mux: float, sigmax: float) -> npt.NDArray[np.float64]:
    return lambertWdelta((z-mux)/sigmax, delta)*sigmax + mux


@nb.njit(nb.float64(nb.float64[:], nb.float64, nb.float64, nb.float64))
def lambertWgaussiandist(y: npt.NDArray[np.float64], delta: float, mux: float, sigmax: float) -> float:
    z = (y-mux) / sigmax
    factor1 = normaldist.pdf(lambertWdelta(z, delta)*sigmax+mux)
    factor2 = lambertWdelta(z, delta) / z / (1 + np.real(lambertw(delta*z*z)))
    return np.sum(factor1*factor2)


@nb.njit(nb.float64(nb.float64[:], nb.float64))
def derivative_lambertWgaussianMLE_z(zarray: npt.NDArray[np.float64], delta: float) -> float:
    wdfcn = np.vectorize(partial(lambertWdelta, delta=delta))
    wdvaluessq = wdfcn(zarray)*wdfcn(zarray)
    denominator = 1 + np.real(lambertw(delta*zarray*zarray, 0))
    return np.sum(
        wdvaluessq/denominator * (0.5*wdvaluessq - 0.5 - 1./denominator)
    )


def find_delta_gradient_descent(
        zarray: npt.NDArray[np.float64],
        learningrate: float = 1e-6,
        delta0: float = 0.5,
        tol: float = 1e-6,
        maxnbepochs: int = 10000
) -> tuple[float, list[tuple[int, float]]]:
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
def compute_kurtosis(x: npt.NDArray[np.float64]) -> float:
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    y = np.square(x-mean)
    return np.sum(np.square(y)) / (n * std*std*std*std)


@nb.njit(nb.float64(nb.float64[:]))
def compute_delta_Taylor(z: npt.NDArray[np.float64]) -> float:
    kurtosis = compute_kurtosis(z)
    disc = 66*kurtosis - 162
    if disc > 0:
        return (np.sqrt(disc) - 6) / 66.
    else:
        return 0.


def compute_delta_GMM(
        z: npt.NDArray[np.float64],
        kurtosis: float,
        initial_delta: Optional[float] = None,
        tol: float = 1e-7
) -> float:
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


def IGMM(
        y: npt.NDArray[np.float64],
        kurtosis: float,
        tol: float = 1e-7,
        maxnpepochs: int = 10000,
        returnnbsteps: bool = False
) -> Union[tuple[float, float, float, int], tuple[float, float, float]]:
    mu = np.median(y)
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

    if returnnbsteps:
        return mu, std, delta, k
    else:
        return mu, std, delta
