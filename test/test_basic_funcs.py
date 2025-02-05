
import unittest
from math import exp

import numpy as np

import pylambertw.tukeyhutils as tutils


class TestBasic(unittest.TestCase):
    def test_tukeyh(self):
        np.testing.assert_almost_equal(
            tutils.tukeyh(np.array([-100., 0., 1]), 1.5),
            np.array([-np.inf, 0., exp(0.5*1.5)])
        )
        np.testing.assert_almost_equal(
            tutils.tukeyh(np.array([-100., 0., 11.2, 25.4]), 0.),
            np.array([-100., 0., 11.2, 25.4])
        )
        np.testing.assert_almost_equal(
            tutils.tukeyh(np.array([-110., 0., 34.2]), -10.),
            np.array([0., 0., 0.])
        )


if __name__ == '__main__':
    unittest.main()
