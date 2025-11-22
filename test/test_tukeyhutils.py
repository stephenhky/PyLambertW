import unittest
import numpy as np
from scipy.special import lambertw
import pylambertw.tukeyhutils as tutils


class TestTukeyhutils(unittest.TestCase):
    
    def test_tukeyh(self):
        # Test with positive delta
        result = tutils.tukeyh(np.array([-100., 0., 1]), 1.5)
        expected = np.array([-np.inf, 0., np.exp(0.5*1.5)])
        np.testing.assert_almost_equal(result, expected)
        
        # Test with zero delta
        result = tutils.tukeyh(np.array([-100., 0., 11.2, 25.4]), 0.)
        expected = np.array([-100., 0., 11.2, 25.4])
        np.testing.assert_almost_equal(result, expected)
        
        # Test with negative delta
        result = tutils.tukeyh(np.array([-110., 0., 34.2]), -10.)
        expected = np.array([0., 0., 0.])
        np.testing.assert_almost_equal(result, expected)
    
    def test_lambertWdelta(self):
        # Test with non-zero delta
        z = np.array([1.0, 2.0, -1.5])
        delta = 0.5
        result = tutils.lambertWdelta(z, delta)
        # Manual calculation: sign(z) * sqrt(lambertw(delta*z*z)/delta)
        expected = np.sign(z) * np.sqrt(np.real(lambertw(delta*z*z))/delta)
        np.testing.assert_almost_equal(result, expected)
        
        # Test with zero delta
        z = np.array([1.0, 2.0, -1.5])
        delta = 0.0
        result = tutils.lambertWdelta(z, delta)
        expected = z
        np.testing.assert_almost_equal(result, expected)
    
    def test_f2heavytail(self):
        u = np.array([0.0, 1.0, -1.0])
        delta = 0.25
        mux = 0.0
        sigmax = 1.0
        result = tutils.f2heavytail(u, delta, mux, sigmax)
        expected = u * np.exp(0.5 * delta * u * u) * sigmax + mux
        np.testing.assert_almost_equal(result, expected)
    
    def test_heavytail2f(self):
        z = np.array([0.0, 1.0, -1.0])
        delta = 0.25
        mux = 0.0
        sigmax = 1.0
        result = tutils.heavytail2f(z, delta, mux, sigmax)
        # Apply inverse transformation
        intermediate = tutils.lambertWdelta((z-mux)/sigmax, delta)
        expected = intermediate * sigmax + mux
        np.testing.assert_almost_equal(result, expected)
    
    def test_compute_kurtosis(self):
        # Test with normal distribution (kurtosis should be close to 3)
        np.random.seed(42)
        x = np.random.normal(0, 1, 10000)
        result = tutils.compute_kurtosis(x)
        # For a normal distribution, kurtosis should be approximately 3
        self.assertAlmostEqual(result, 3.0, places=1)
        
        # Test with exact values
        x = np.array([1, 2, 3, 4, 5])
        result = tutils.compute_kurtosis(x)
        # Manual calculation for this small array
        mean = np.mean(x)
        std = np.std(x)
        expected = np.sum((x - mean)**4) / (len(x) * std**4)
        self.assertAlmostEqual(result, expected, places=7)
    
    def test_compute_delta_Taylor(self):
        # Test with normal distribution (should return 0)
        np.random.seed(42)
        z = np.random.normal(0, 1, 10000)
        result = tutils.compute_delta_Taylor(z)
        self.assertEqual(result, 0.0)
        
        # Test with a distribution that should give positive delta
        # Create a heavy-tailed distribution
        z = np.array([1, 2, 3, 4, 5, 10, 15, 20])  # Heavy tail
        result = tutils.compute_delta_Taylor(z)
        self.assertGreaterEqual(result, 0.0)
    
    def test_IGMM(self):
        # Test with normal distribution
        np.random.seed(42)
        y = np.random.normal(0, 1, 1000)
        kurtosis = 3.0
        mu, std, delta = tutils.IGMM(y, kurtosis)
        
        # Check that results are reasonable
        self.assertIsInstance(mu, float)
        self.assertIsInstance(std, float)
        self.assertIsInstance(delta, float)
        self.assertGreater(std, 0)
        
        # Test with returnnbsteps parameter
        mu, std, delta, nbsteps = tutils.IGMM(y, kurtosis, returnnbsteps=True)
        self.assertIsInstance(nbsteps, int)
        self.assertGreaterEqual(nbsteps, 0)


if __name__ == '__main__':
    unittest.main()