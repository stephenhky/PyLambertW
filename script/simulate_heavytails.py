
from time import time
from multiprocessing import Pool
from functools import partial
from argparse import ArgumentParser
import os
import logging

import numpy as np
import pandas as pd

from pylambertw.tukeyhutils import f2heavytail, IGMM, compute_kurtosis


def generate_longtail_data(nbdata, delta, mu, sigma):
    z = np.random.normal(loc=0, scale=1, size=nbdata)
    return f2heavytail(z, delta, mu, sigma)


def simulate_single(parameter, nbdata):
    delta = parameter['delta']
    mu = parameter['mu']
    sigma = parameter['sigma']
    zarray = generate_longtail_data(nbdata, delta, mu, sigma)
    kurtosis = compute_kurtosis(zarray)
    starttime = time()
    inferred_mu, inferred_sigma, inferred_delta = IGMM(zarray, 3.0)
    endtime = time()
    results = {
        'nbdata': nbdata,
        'delta': delta,
        'mu': mu,
        'sigma': sigma,
        'kurtosis': kurtosis,
        'inferred_mu': inferred_mu,
        'inferred_sigma': inferred_sigma,
        'inferred_delta': inferred_delta,
        'duration': endtime - starttime
    }
    logging.info(results)
    return results


def get_argparser():
    argparser = ArgumentParser(description='Simulate heavy tail')
    argparser.add_argument('outputexcelfile', help='Output Excel File')
    argparser.add_argument('--nbpools', default=10, type=int, help='Number of pool workers (default: 10)')
    return argparser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    excelfilepath = args.outputexcelfile
    if not os.path.isdir(os.path.dirname(excelfilepath)):
        raise FileNotFoundError('Directory {} does not exist!'.format(os.path.dirname(excelfilepath)))

    deltas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    stat_parameters = [
        {'mu': 0.0, 'sigma': 1.0},
        {'mu': 0.0, 'sigma': 2.5},
        {'mu': 0.5, 'sigma': 1.0},
        {'mu': 0.5, 'sigma': 2.5},
        {'mu': 1.0, 'sigma': 1.0},
        {'mu': 1.0, 'sigma': 2.5},
        {'mu': 5.0, 'sigma': 1.0},
        {'mu': 5.0, 'sigma': 2.5},
        {'mu': 5.0, 'sigma': 10.0},
        {'mu': 10.0, 'sigma': 1.0},
        {'mu': 10.0, 'sigma': 10.0},
        {'mu': 10.0, 'sigma': 15.0}
    ]
    parameters = []
    for param in stat_parameters:
        for delta in deltas:
            copied_param = param.copy()
            copied_param['delta'] = delta
            parameters.append(copied_param)

    p = Pool(10)
    results = p.map(partial(simulate_single, nbdata=10000), parameters)

    df = pd.DataFrame.from_dict(results)
    df.to_excel(excelfilepath)
