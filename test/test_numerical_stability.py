import unittest
import numpy as np
import warnings
import pylambertw.tukeyhutils as tutils
import pylambertw.lamberter as lamberter


class TestNumericalStability(unittest.TestCase):
    
    def test_lambertWdelta_extreme_values(self):
        # Test with very large positive values
        z = np.array([1e10, 1e15])
        delta = 0.1
        result = tutils.lambertWdelta(z, delta)
        # Should not produce NaN or Inf
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        
        # Test with very large negative values
        z = np.array([-1e10, -1e15])
        delta = 0.1
        result = tutils.lambertWdelta(z, delta)
        # Should not produce NaN or Inf
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        
        # Test with very large delta
        z = np.array([1.0, 2.0])
        delta = 1e10
        result = tutils.lambertWdelta(z, delta)
        # Should not produce NaN or Inf
        self.assertFalse(np.any(np.isnan(result)))
        
        # Test with very small delta
        z = np.array([1.0, 2.0])
        delta = 1e-10
        result = tutils.lambertWdelta(z, delta)
        # Should not produce NaN or Inf
        self.assertFalse(np.any(np.isnan(result)))
    
    def test_tukeyh_extreme_values(self):
        # Test with very large positive values and positive h
        x = np.array([1e5, 1e10])
        h = 0.1
        result = tutils.tukeyh(x, h)
        # Should not produce NaN
        self.assertFalse(np.any(np.isnan(result)))
        
        # Test with very large negative values and positive h
        x = np.array([-1e5, -1e10])
        h = 0.1
        result = tutils.tukeyh(x, h)
        # Should not produce NaN
        self.assertFalse(np.any(np.isnan(result)))
        
        # Test with very large h
        x = np.array([1.0, 2.0])
        h = 1e5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = tutils.tukeyh(x, h)
        # Should not produce NaN (may produce Inf which is acceptable)
        self.assertFalse(np.any(np.isnan(result)))
    
    def test_IGMM_edge_cases(self):
        # Test with constant array (zero variance)
        y = np.array([1.0, 1.0, 1.0, 1.0])
        kurtosis = 3.0
        mu, std, delta = tutils.IGMM(y, kurtosis)
        # Standard deviation should be positive (regularization added)
        self.assertGreater(std, 0)
        
        # Test with very small array
        y = np.array([1.0, 2.0])
        kurtosis = 3.0
        mu, std, delta = tutils.IGMM(y, kurtosis)
        # Should not produce NaN or Inf
        self.assertFalse(np.isnan(mu))
        self.assertFalse(np.isnan(std))
        self.assertFalse(np.isnan(delta))
        self.assertFalse(np.isinf(mu))
        self.assertFalse(np.isinf(std))
        self.assertFalse(np.isinf(delta))
        
        # Test with array containing outliers
        y = np.array([1.0, 2.0, 3.0, 1e10])
        kurtosis = 3.0
        mu, std, delta = tutils.IGMM(y, kurtosis)
        # Should not produce NaN or Inf
        self.assertFalse(np.isnan(mu))
        self.assertFalse(np.isnan(std))
        self.assertFalse(np.isnan(delta))
        self.assertFalse(np.isinf(mu))
        self.assertFalse(np.isinf(std))
        self.assertFalse(np.isinf(delta))
    
    def test_GaussianLamberter_edge_cases(self):
        # Test with constant input
        lamberter_model = lamberter.GaussianLamberter()
        X = np.array([1.0, 1.0, 1.0, 1.0])
        try:
            result = lamberter_model.fit_transform(X)
            # Should not produce NaN or Inf
            self.assertFalse(np.any(np.isnan(result)))
            self.assertFalse(np.any(np.isinf(result)))
        except Exception:
            # If it raises an exception for constant input, that's acceptable
            pass
        
        # Test with very large values
        X = np.array([1e10, 1e15, -1e10])
        result = lamberter_model.fit_transform(X)
        # Should not produce NaN
        self.assertFalse(np.any(np.isnan(result)))
        
        # Test with single outlier
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        X[50] = 1e10  # Insert outlier
        result = lamberter_model.fit_transform(X)
        # Should not produce NaN
        self.assertFalse(np.any(np.isnan(result)))


if __name__ == '__main__':
    unittest.main()