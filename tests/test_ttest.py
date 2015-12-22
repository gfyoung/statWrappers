import os
import sys

wrapperDir = os.path.abspath(os.path.join(
    os.path.dirname( __file__), "..", "wrappers"))
sys.path.insert(0, wrapperDir)

from ttest import ttest_1samp, ttest_2samp
from random import random

import scipy.stats as stats
import numpy as np

import unittest

EPSILON = sys.float_info.epsilon


class TestTtest1Samp(unittest.TestCase):
    def test_invalid_alt_hyp(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        popmean = 5
        alt_hyp = 'bad_alt_hyp'

        self.assertRaises(ValueError, ttest_1samp, a, popmean, alt_hyp=alt_hyp)

    def test_invalid_alpha_type(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        popmean = 5
        alpha = 'bad_alpha'

        self.assertRaises(ValueError, ttest_1samp, a, popmean, alpha=alpha)

    def test_invalid_alpha_negative(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        popmean = 5
        alpha = -1

        self.assertRaises(ValueError, ttest_1samp, a, popmean, alpha=alpha)

    def test_invalid_alpha_too_large(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        popmean = 5
        alpha = 2

        self.assertRaises(ValueError, ttest_1samp, a, popmean, alpha=alpha)

    def test_same_results_changing_alpha(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        popmean = 4.5

        expected_t_stat = 0.54772255750516607
        expected_p_val = 0.59882713669728904

        for i in xrange(10):
            test = ttest_1samp(a, popmean, alpha=random())

            self.assertLessEqual(abs(test.t_stat - expected_t_stat), EPSILON)
            self.assertLessEqual(abs(test.p_val - expected_p_val), EPSILON)

    def test_same_t_stat_changing_alt_hyp(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        popmean = 4.5
        alpha = 0.5

        expected_t_stat = 0.54772255750516607
        expected_p_val = 0.59882713669728904
        expected_p_val_half = expected_p_val / 2.0

        for alt_hyp in ('unequal', 'less', 'greater'):
            test = ttest_1samp(a, popmean, alt_hyp=alt_hyp, alpha=alpha)
            self.assertLessEqual(abs(test.t_stat - expected_t_stat), EPSILON)

            if alt_hyp == 'unequal':
                self.assertLessEqual(abs(test.p_val - expected_p_val), EPSILON)

            else:
                self.assertLessEqual(abs(test.p_val -
                                         expected_p_val_half), EPSILON)

    def test_no_reject_null_hyp(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        popmean = 4.5
        alpha = 0.1

        test = ttest_1samp(a, popmean, alpha=random())
        self.assertLessEqual(alpha, test.p_val)

    def test_reject_null_hyp_unequal(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        popmean = 100
        alpha = 0.1

        test = ttest_1samp(a, popmean, alt_hyp='unequal', alpha=random())
        self.assertGreater(alpha, test.p_val)

    def test_reject_null_hyp_greater(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        popmean = 100
        alpha = 0.1

        test = ttest_1samp(a, popmean, alt_hyp='greater', alpha=random())
        self.assertGreater(alpha, test.p_val)

    def test_reject_null_hyp_less(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        popmean = 0.1
        alpha = 0.1

        test = ttest_1samp(a, popmean, alt_hyp='less', alpha=random())
        self.assertGreater(alpha, test.p_val)


class TestTtest2Samp(unittest.TestCase):
    def test_invalid_alt_hyp(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        alt_hyp = 'bad_alt_hyp'

        self.assertRaises(ValueError, ttest_2samp, a, b, alt_hyp=alt_hyp)

    def test_invalid_test_type(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_type = 'bad_test_type'

        self.assertRaises(ValueError, ttest_2samp, a, b, test_type=test_type)

    def test_invalid_alpha_type(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        alpha = 'bad_alpha'

        self.assertRaises(ValueError, ttest_2samp, a, b, alpha=alpha)

    def test_invalid_alpha_negative(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        alpha = -1

        self.assertRaises(ValueError, ttest_2samp, a, b, alpha=alpha)

    def test_invalid_alpha_too_large(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        alpha = 2

        self.assertRaises(ValueError, ttest_2samp, a, b, alpha=alpha)

    def test_same_results_changing_alpha(self):
        seed = 1234567890
        np.random.seed(seed)
        a = stats.norm.rvs(loc=5, scale=10, size=500)
        b = stats.norm.rvs(loc=5, scale=10, size=500)

        expected = {
            'ind': {
                True: {
                    't_stat': 0.29359890626535035,
                    'p_val': 0.76912545487694084
                    },
                False: {
                    't_stat': 0.29359890626535035,
                    'p_val': 0.76912546718732111
                    }
                },
            'rel': {
                True: {
                    't_stat': 0.3015015987751255,
                    'p_val': 0.76315763376265355
                    },
                False: {
                    't_stat': 0.3015015987751255,
                    'p_val': 0.76315763376265355
                    }
                }
            }

        for test_type in ('ind', 'rel'):
            for equal_var in (True, False):
                for i in xrange(10):
                    test = ttest_2samp(a, b, test_type=test_type,
                                       equal_var=equal_var, alpha=random())

                    expected_t_stat = expected[test_type][equal_var]['t_stat']
                    expected_p_val = expected[test_type][equal_var]['p_val']

                    self.assertLessEqual(abs(test.t_stat -
                                             expected_t_stat), EPSILON)
                    self.assertLessEqual(abs(test.p_val -
                                             expected_p_val), EPSILON)

    def test_same_t_stat_changing_alt_hyp(self):
        seed = 1234567890
        np.random.seed(seed)
        a = stats.norm.rvs(loc=5, scale=10, size=500)
        b = stats.norm.rvs(loc=5, scale=10, size=500)

        equal_var = True
        alpha = 0.5

        expected = {
            'ind': {
                'unequal': {
                    't_stat': 0.29359890626535035,
                    'p_val': 0.76912545487694084
                    },
                'less': {
                    't_stat': 0.29359890626535035,
                    'p_val': 0.76912545487694084 / 2.0
                    },
                'greater': {
                    't_stat': 0.29359890626535035,
                    'p_val': 0.76912545487694084 / 2.0
                    },
                },
            'rel': {
                'unequal': {
                    't_stat': 0.3015015987751255,
                    'p_val': 0.76315763376265355
                    },
                'less': {
                    't_stat': 0.3015015987751255,
                    'p_val': 0.76315763376265355 / 2.0
                    },
                'greater': {
                    't_stat': 0.3015015987751255,
                    'p_val': 0.76315763376265355 / 2.0
                    }
                }
            }

        for test_type in ('ind', 'rel'):
            for alt_hyp in ('unequal', 'less', 'greater'):
                for i in xrange(10):
                    test = ttest_2samp(a, b, test_type=test_type,
                                       equal_var=equal_var, alt_hyp=alt_hyp,
                                       alpha=alpha)

                    expected_t_stat = expected[test_type][alt_hyp]['t_stat']
                    expected_p_val = expected[test_type][alt_hyp]['p_val']

                    self.assertLessEqual(abs(test.t_stat -
                                             expected_t_stat), EPSILON)
                    self.assertLessEqual(abs(test.p_val -
                                             expected_p_val), EPSILON)

    def test_no_reject_null_hyp(self):
        alpha = 0.1
        diff = 0.1

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.array([i + diff for i in a])

        test = ttest_2samp(a, b, alpha=random())
        self.assertLessEqual(alpha, test.p_val)

    def test_reject_null_hyp_unequal(self):
        alpha = 0.1
        diff = -100

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.array([i + diff for i in a])

        test = ttest_2samp(a, b, alpha=random())
        self.assertGreater(alpha, test.p_val)

    def test_reject_null_hyp_greater(self):
        alpha = 0.1
        diff = -100

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.array([i + diff for i in a])

        test = ttest_2samp(a, b, alpha=random())
        self.assertGreater(alpha, test.p_val)

    def test_reject_null_hyp_less(self):
        alpha = 0.1
        diff = 100

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.array([i + diff for i in a])

        test = ttest_2samp(a, b, alpha=random())
        self.assertGreater(alpha, test.p_val)

if __name__ == '__main__':
    unittest.main()
