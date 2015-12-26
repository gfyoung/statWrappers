import os
import sys
import warnings
import unittest

wrapperDir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "wrappers"))
sys.path.insert(0, wrapperDir)

from ols import ols

import numpy as np

EPSILON = 1e-12
MAXINT = 1e12

class TestOLS(unittest.TestCase):
    def test_singular_matrix(self):
        x = np.ones((1, 10))
        y = np.array([range(10)])

        self.assertRaises(np.linalg.LinAlgError, ols, x, y)

    def test_no_x_names_single_x(self):
        seed = 1234567890
        np.random.seed(seed)

        x = np.random.rand(10)
        y = np.random.rand(10)

        reg = ols(x, y)

        expectedNames = ['const', 'x']
        self.assertEqual(reg.x_varnm, expectedNames)

    def test_no_x_names_multi_x(self):
        seed = 1234567890
        np.random.seed(seed)
        
        x = np.random.rand(10, 2)
        y = np.random.rand(10)

        reg = ols(x, y)

        expectedNames = ['const', 'x1', 'x2']
        self.assertEqual(reg.x_varnm, expectedNames)

    def test_strong_regression(self):
        seed = 1234567890
        np.random.seed(seed)

        # Due to the strong linearity of the regressions,
        # RuntimeWarnings may be raised when computing some
        # of the regression statistics. We can ignore these.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(10):
                coeff = np.random.random_integers(1, 100)
                inter = np.random.random_integers(1, 100)

                x = np.random.rand(10)
                y = np.array([coeff * i + inter for i in x])

                reg = ols(x, y)

                expected = np.array([inter, coeff])
                diff = abs(expected - reg.b)

                self.assertTrue(np.all(diff < EPSILON))
                self.assertTrue(np.all(reg.p < EPSILON))
                self.assertTrue(np.all(abs(reg.t) > MAXINT))

    def test_weak_regression(self):
        seed = 1234567890
        np.random.seed(seed)

        alpha = 0.1
        tStatMax = 1

        for i in range(10):
            x = np.random.random_integers(100, 200, 10)
            y = np.array([i * (-1)**index for index, i in enumerate(x)])

            reg = ols(x, y)

            self.assertTrue(np.all(reg.p > alpha))
            self.assertTrue(np.all(abs(reg.t) < tStatMax))

    def test_str_object(self):
        seed = 1234567890
        np.random.seed(seed)

        x = np.random.rand(10)
        y = np.random.rand(10)

        reg = ols(x, y)
        expected = "OLS Regression on 10 Observations"
        self.assertTrue(str(reg) == expected, "Strings don't match")

if __name__ == '__main__':
    unittest.main()
