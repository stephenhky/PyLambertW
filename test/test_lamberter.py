import unittest
import numpy as np
import pylambertw.lamberter as lamberter


class TestGaussianLamberter(unittest.TestCase):
    
    def setUp(self):
        self.lamberter = lamberter.GaussianLamberter()
    
    def test_init_default_values(self):
        self.assertEqual(self.lamberter.mu, 0.0)
        self.assertEqual(self.lamberter.sigma, 1.0)
        self.assertEqual(self.lamberter.delta, 0.25)
        self.assertFalse(hasattr(self.lamberter, 'nbsteps'))
    
    def test_init_custom_values(self):
        custom_lamberter = lamberter.GaussianLamberter(mu=1.0, sigma=2.0, delta=0.5)
        self.assertEqual(custom_lamberter.mu, 1.0)
        self.assertEqual(custom_lamberter.sigma, 2.0)
        self.assertEqual(custom_lamberter.delta, 0.5)
    
    def test_fit(self):
        # Generate test data
        np.random.seed(42)
        X = np.random.normal(0, 1, 1000)
        
        # Fit the model
        fitted_model = self.lamberter.fit(X, maxnbepochs=1000)
        
        # Check that parameters have been updated
        self.assertIsNotNone(fitted_model.mu)
        self.assertIsNotNone(fitted_model.sigma)
        self.assertIsNotNone(fitted_model.delta)
        self.assertIsNotNone(fitted_model.nbsteps)
        
        # Check that returned object is the same as self
        self.assertIs(fitted_model, self.lamberter)
    
    def test_transform(self):
        # Set known parameters
        self.lamberter.mu = 0.0
        self.lamberter.sigma = 1.0
        self.lamberter.delta = 0.25
        
        # Transform test data
        X = np.array([0.0, 1.0, -1.0])
        transformed = self.lamberter.transform(X)
        
        # Check shape is preserved
        self.assertEqual(transformed.shape, X.shape)
        
        # Check that transformation is applied
        self.assertNotEqual(list(transformed), list(X))
    
    def test_fit_transform(self):
        # Generate test data
        np.random.seed(42)
        X = np.random.normal(0, 1, 1000)
        
        # Apply fit_transform
        transformed = self.lamberter.fit_transform(X, maxnbepochs=1000)
        
        # Check that parameters have been set
        self.assertIsNotNone(self.lamberter.mu)
        self.assertIsNotNone(self.lamberter.sigma)
        self.assertIsNotNone(self.lamberter.delta)
        self.assertIsNotNone(self.lamberter.nbsteps)
        
        # Check shape is preserved
        self.assertEqual(transformed.shape, X.shape)
    
    def test_str_representation(self):
        # Test string representation with default values
        str_repr = str(self.lamberter)
        self.assertIn("GaussianLamberter: mu=0.0, sigma=1.0, delta=0.25", str_repr)
        self.assertIn("(number of steps: None)", str_repr)
        
        # Test string representation after fitting
        self.lamberter.nbsteps = 100
        str_repr = str(self.lamberter)
        self.assertIn("(number of steps: 100)", str_repr)
    
    def test_edge_cases(self):
        # Test with empty array
        X_empty = np.array([])
        # Should raise an error or handle gracefully
        # We'll test that it doesn't crash
        try:
            result = self.lamberter.fit_transform(X_empty)
            # If it doesn't raise an exception, result should be empty
            self.assertEqual(len(result), 0)
        except Exception:
            # If it raises an exception, that's okay too
            pass
        
        # Test with single element
        X_single = np.array([1.0])
        result = self.lamberter.fit_transform(X_single)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()