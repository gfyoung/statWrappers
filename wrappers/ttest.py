from __future__ import print_function
from time import localtime, strftime
from json import dump

import scipy.stats as stats


class ttest_1samp(object):
    def __init__(self, a, popmean, alt_hyp='unequal', alpha=None):
        """

        Initializes a 1-sample t-test for the mean of ONE group of observations.

        Parameters
        ----------
        a : array_like
            An array-like object of observations

        popmean : float
            The expected value of the mean in the null hypothesis, which is
            that the sample mean is equal to the population mean.

        alt_hyp : string, optional
            The alternative hypothesis. Allowed options are 'unequal',
            'greater', or 'less'. 'unequal' signifies that the alternative
            hypothesis is that the mean of `a` is not equal to `popmean`.
            'greater' signifies that the alternative hypothesis is that the
            mean of `a` is greater than `popmean`. 'less' signifies that the
            alternative hypothesis is that the mean of `a` is less than
            `popmean`. The default is 'unequal'.

        alpha : float, optional
            The cutoff value for the p-value computed during the t-test below
            which we can feel comfortable rejecting the null hypothesis. Note
            that this does not mean that we can accept the alternative hypothesis.

        """

        self.a = a
        self.popmean = popmean

        self.alt_hyp = alt_hyp
        self.alpha = alpha

        self.check_params()
        self.test()

    def check_params(self):
        """

        Checks the validity of the `alt_hyp` and `alpha` parameters passed into
        the __init__ method. Throws a ValueError if either parameter is found to
        be invalid. Note that this method is only meant to be called internally
        to the class and not externally.

        """

        if self.alt_hyp not in ('less', 'unequal', 'greater'):
            raise ValueError("Invalid alternative hypothesis. " +
                             "Expected 'less', 'unequal', or 'greater' " +
                             "but got: '" + self.alt_hyp + "'")

        if self.alpha:
            if type(self.alpha) not in (int, float):
                raise ValueError("Invalid alpha data type. " +
                                 "Expected 'int' or 'float' " +
                                 "but got: '" + type(self.alpha).__name__ + "'")

            if self.alpha < 0 or self.alpha > 1:
                raise ValueError("Invalid alpha data value. " +
                                 "Expected somewhere in range [0, 1] " +
                                 "but got a value of:", str(self.alpha))

    def test(self):
        """

        Performs the actual t-test and saves the computed t-statistic and
        p-value as attributes of the class instance.

        """

        self.t_stat, self.p_val = stats.ttest_1samp(self.a, self.popmean)

        if self.alt_hyp != 'unequal':
            self.p_val /= 2.0

    def summary(self):
        """

        Summarizes the results of the t-test performed and prints the results
        out into STDOUT, which is generally the console or terminal on which
        the code is being executed.

        """

        t = localtime()
        size = len(self.a)

        assumptions = '   Independent Observations'

        if self.alt_hyp == 'unequal':
            alternative = "Mean 1 != " + str(self.popmean)
            sided = "two-sided"

        else:
            sided = "one-sided"

            if self.alt_hyp == "less":
                alternative = "Mean 1 < " + str(self.popmean)

            else:
                alternative = "Mean 1 > " + str(self.popmean)

        print("\n==============================================================================")
        print("Significance Test: T-Test (" + sided + ")")
        print("Date:", strftime("%a, %d %b %Y", t))
        print("Time:", strftime("%H:%M:%S", t))

        print("\nAssumptions: ")
        print(assumptions)

        print("\nData Size:", str(size))
        print("Population Mean:", str(self.popmean))

        print("\nNull Hypothesis:        Mean 1 ==", self.popmean)
        print("Alternative Hypothesis:", alternative)

        print("\nT-Statistic:", str(self.t_stat))
        print("P-Value:", str(self.p_val))
        print("Alpha:", str(self.alpha))

        NULL = "\nReject Null Hypothesis: "
        ALT = "Accept Alternative Hypothesis: "

        YES = "Yes"
        NO = "No"

        if (self.alpha and self.p_val >= self.alpha) or \
           self.p_val == 0 or not self.alpha:
            print(NULL + NO)
            print(ALT + NO)

        else:
            print(NULL + YES)

            if self.alt_hyp == 'unequal' or \
               (self.alt_hyp == 'less' and self.t_stat < 0) or \
               (self.alt_hyp == 'greater' and self.t_stat > 0):
                print(ALT + YES)

            else:
                print(ALT + NO)

        print("==============================================================================")

    def to_file(self, filename=None):
        """

        Summarizes the results of the t-test performed and saves the results
        out into `filename`.

        Parameters
        ----------

        filename : string, optional
            The location of the file where the results will be stored. If no
            filename is provided, a default filename will be generated, and the
            results will be stored there.

        """

        t = localtime()
        size = len(self.a)

        assumptions = ['Independent Observations']

        if self.alt_hyp == 'unequal':
            alternative = "Mean 1 != " + str(self.popmean)
            sided = "two-sided"

        else:
            sided = "one-sided"

            if self.alt_hyp == "less":
                alternative = "Mean 1 < " + str(self.popmean)

            else:
                alternative = "Mean 1 > " + str(self.popmean)

        data = {}

        data['sig_test'] = "T-Test (" + sided + ")"
        data['date'] = strftime("%a, %d %b %Y", t)
        data['time'] = strftime("%H:%M:%S", t)
        data['assumptions'] = assumptions

        data['data_size'] = size
        data['pop_mean'] = self.popmean

        data['null_hyp'] = "Mean 1 ==", self.popmean
        data['alt_hyp'] = alternative
        data['t_stat'] = self.t_stat
        data['p_val'] = self.p_val

        if (self.alpha and self.p_val >= self.alpha) or \
           self.p_val == 0 or not self.alpha:
            data['reject_null'] = False
            data['accept_alt'] = False

        else:
            data['reject_null'] = True

            if self.alt_hyp == 'unequal' or \
               (self.alt_hyp == 'less' and self.t_stat < 0) or \
               (self.alt_hyp == 'greater' and self.t_stat > 0):
                data['accept_alt'] = True

            else:
                data['accept_alt'] = False

        filename = filename or strftime("%a_%d_%b_%Y_%H_%M_%S.json", t)

        with open(filename, 'w') as target:
            dump(data, target)

    def __str__(self):
        return "1-Sample T-Test on Data of Size " + str(len(self.a)) + \
               ", Hypothesized Population Mean of " + str(self.popmean)

    __repr__ = __str__


class ttest_2samp(object):
    def __init__(self, a, b, test_type='ind', equal_var=True,
                 alt_hyp='unequal', alpha=None):
        """

        Initializes a 2-sample t-test for the means of TWO groups of observations.

        Parameters
        ----------
        a, b: array_like
            Array-like objects of observations. They must have the same length.

        test_type : string, optional
            The type of 2-sample t-test that is to be performed on the samples.
            Allowed options are 'ind' and 'rel'. The former means that the samples
            were drawn independently of each other, while the latter means that the
            samples are related in some way (e.g. the populations from which they
            are drawn are similar). The default is 'ind'.

        equal_var : bool, optional
            Indicates whether the two populations these two samples are drawn
            from have equal variances (True) or not (False). The default is 'True'.

        alt_hyp : string, optional
            The alternative hypothesis. Allowed options are 'unequal',
            'greater', or 'less'. 'unequal' signifies that the alternative
            hypothesis is that the mean of `a` is not equal to the mean of
            `b`. 'greater' signifies that the alternative hypothesis is the
            mean of `a` is greater than the mean of `b`. 'less' signifies
            that the alternative hypothesis is that the mean of `a` is less
            than the mean of `b`. The default is 'unequal'.

        alpha : float, optional
            The cutoff value for the p-value computed during the t-test below
            which we can feel comfortable rejecting the null hypothesis, which
            is that the mean of `a` is equal to the mean of `b`. Note that this
            does not mean that we can accept the alternative hypothesis.

        """

        self.a = a
        self.b = b

        self.test_type = test_type

        self.equal_var = equal_var
        self.alt_hyp = alt_hyp
        self.alpha = alpha

        self.check_params()
        self.test()

    def check_params(self):
        """

        Checks the validity of the `test_type`, `alt_hyp`, and `alpha` parameters
        passed into the __init__ method. Throws a ValueError if any of those
        parameters are found to be invalid. Note that this method is only meant
        to be called internally to the class and not externally.

        """

        if self.test_type not in ('ind', 'rel'):
            raise ValueError("Invalid t-test type. " +
                             "Expected 'ind' or 'rel' " +
                             "but got: '" + self.test_type + "'")

        if self.alt_hyp not in ('less', 'unequal', 'greater'):
            raise ValueError("Invalid alternative hypothesis. " +
                             "Expected 'less', 'unequal', or 'greater' " +
                             "but got: '" + self.alt_hyp + "'")

        if self.alpha:
            if type(self.alpha) not in (int, float):
                raise ValueError("Invalid alpha data type. " +
                                 "Expected 'int' or 'float' " +
                                 "but got: '" + type(self.alpha).__name__ + "'")

            if self.alpha < 0 or self.alpha > 1:
                raise ValueError("Invalid alpha data value. " +
                                 "Expected somewhere in range [0, 1] " +
                                 "but got a value of:", str(self.alpha))

    def test(self):
        """

        Performs the actual t-test and saves the computed t-statistic and
        p-value as attributes of the class instance.

        """

        if self.test_type == 'ind':
            self.t_stat, self.p_val = stats.ttest_ind(self.a, self.b,
                                                      equal_var=self.equal_var)

        else:
            self.t_stat, self.p_val = stats.ttest_rel(self.a, self.b)

        if self.alt_hyp != 'unequal':
            self.p_val /= 2.0

    def summary(self):
        """

        Summarizes the results of the t-test performed and prints the results
        out into STDOUT, which is generally the console or terminal on which
        the code is being executed.

        """

        t = localtime()
        size = len(self.a)

        if self.test_type == 'ind':
            assumptions = '   Independent Samples'
            assumptions = assumptions + '\n   Equal Variances' if \
                          self.equal_var else assumptions + '\n\tEqual Variances'

        else:
            assumptions = 'Related Samples'

        if self.alt_hyp == 'unequal':
            alternative = "Mean 1 != Mean 2"
            sided = "two-sided"

        else:
            sided = "one-sided"

            if self.alt_hyp == "less":
                alternative = "Mean 1 < Mean 2"

            else:
                alternative = "Mean 1 > Mean 2"

        print("\n==============================================================================")
        print("Significance Test: T-Test (" + sided + ")")
        print("Date:", strftime("%a, %d %b %Y", t))
        print("Time:", strftime("%H:%M:%S", t))

        print("\nAssumptions: ")
        print(assumptions)

        print("\nData Size:", str(size))

        print("\nNull Hypothesis:        Mean 1 == Mean 2")
        print("Alternative Hypothesis:", alternative)

        print("\nT-Statistic:", str(self.t_stat))
        print("P-Value:", str(self.p_val))
        print("Alpha:", str(self.alpha))

        NULL = "\nReject Null Hypothesis: "
        ALT = "Accept Alternative Hypothesis: "

        YES = "Yes"
        NO = "No"

        if (self.alpha and self.p_val >= self.alpha) or \
           self.p_val == 0 or not self.alpha:
            print(NULL + NO)
            print(ALT + NO)

        else:
            print(NULL + YES)

            if self.alt_hyp == 'unequal' or \
               (self.alt_hyp == 'less' and self.t_stat < 0) or \
               (self.alt_hyp == 'greater' and self.t_stat > 0):
                print(ALT + YES)

            else:
                print(ALT + NO)

        print("==============================================================================")

    def to_file(self, filename=None):
        """

        Summarizes the results of the t-test performed and saves the results
        out into `filename`.

        Parameters
        ----------

        filename : string, optional
            The location of the file where the results will be stored. If no
            filename is provided, a default filename will be generated, and the
            results will be stored there.

        """

        t = localtime()
        size = len(self.a)

        if self.test_type == 'ind':
            assumptions = ['Independent Samples']
            assumptions = assumptions + ['Equal Variances'] if \
                          self.equal_var else assumptions + ['Equal Variances']

        else:
            assumptions = ['Related Samples']

        if self.alt_hyp == 'unequal':
            alternative = "Mean 1 != Mean 2"
            sided = "two-sided"

        else:
            sided = "one-sided"

            if self.alt_hyp == "less":
                alternative = "Mean 1 < Mean 2"

            else:
                alternative = "Mean 1 > Mean 2"

        data = {}

        data['sig_test'] = "T-Test (" + sided + ")"
        data['date'] = strftime("%a, %d %b %Y", t)
        data['time'] = strftime("%H:%M:%S", t)
        data['assumptions'] = assumptions

        data['data_size'] = size

        data['null_hyp'] = "Mean 1 == Mean 2"
        data['alt_hyp'] = alternative
        data['t_stat'] = self.t_stat
        data['p_val'] = self.p_val

        if (self.alpha and self.p_val >= self.alpha) or \
           self.p_val == 0 or not self.alpha:
            data['reject_null'] = False
            data['accept_alt'] = False

        else:
            data['reject_null'] = True

            if self.alt_hyp == 'unequal' or \
               (self.alt_hyp == 'less' and self.t_stat < 0) or \
               (self.alt_hyp == 'greater' and self.t_stat > 0):
                data['accept_alt'] = True

            else:
                data['accept_alt'] = False

        filename = filename or strftime("%a_%d_%b_%Y_%H_%M_%S.json", t)

        with open(filename, 'w') as target:
            dump(data, target)

    def __str__(self):
        return "2-Sample T-Test on Data of Size " + str(len(self.a))

    __repr__ = __str__

if __name__ == '__main__':
    from numpy import array

    a = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = array([i + 30 for i in a])

    t = ttest_2samp(a, b, alt_hyp='greater', alpha=0.01)
    t.summary()

    t = ttest_1samp(a, 10, alt_hyp='less', alpha=0.01)
    t.summary()
