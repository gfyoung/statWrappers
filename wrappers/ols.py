# Code largely adapted from the following link:
# https://scipy.github.io/old-wiki/pages/Cookbook/OLS2d68.html?action=AttachFile&do=view&target=ols.0.2.py
#
# More information on the original code, please visit this link:
# https://scipy.github.io/old-wiki/pages/Cookbook/OLS.html

from __future__ import division, print_function

from scipy import c_, ones, dot, stats, diff
from scipy.linalg import inv, solve, det

from numpy import log, nan, pi, sqrt, square, diagonal
from numpy.linalg import LinAlgError
from numpy.random import randn, seed

from time import localtime, strftime
from json import dump


class ols(object):
    def __init__(self, x, y, x_varnm=None, y_varnm='y'):
        """

        Initializes an ordinary least squares (OLS) analysis on a
        set of data points

        Parameters
        ----------
        x : numpy.ndarray
            A matrix of observations whose columns are considered
            to be the 'independent variables' in the regression.

        y : numpy.ndarray
            An array of observations that is considered to be the
            'dependent variable'. Note that the length of this array
            must be the same length as the matrix for the independent
            variable observations.

        x_varm: list, optional
            A list of names corresponding to the independent variables.
            If no list is provided, a list of default variable names are
            generated. However, if one is passed in, the length of the
            list must be the same width as the matrix for the independent
            variable observations.

        y_varnm : string, optional
            The name of the dependent variable. The default is 'y'.

        """

        self.x = c_[ones(x.shape[0]), x]

        if not x_varnm:
            if len(x.shape) == 1 or x.shape[1] == 1:
                self.x_varnm = ['const', 'x']

            else:
                self.x_varnm = ['const'] + \
                               ['x' + str(i) for i in
                                range(1, x.shape[1] + 1)]

        else:
            self.x_varnm = ['const'] + x_varnm

        self.y = y
        self.y_varnm = y_varnm

        self.estimate()

    def estimate(self):
        """

        Estimates the coefficients for each of the independent variables
        as well as the intercept and computes some basic statistics for
        the model being generated. These estimate and statistics are all
        stored as attributes of the OLS instance. They are as follows:

        b : estimates for the coefficients and intercept

        nobs : number of observations or data points

        ncoefs : number of coefficients plus the intercept

        df_e : degrees of freedom for the error

        df_r : degrees of freedom for the regression

        e : residuals from the regression

        sse : sum of the residuals squared

        se : standard errors for the coefficients and the intercept estimated

        t : t-statistics for the coefficients and the intercept estimated

        p : p-values for the coefficients and the intercept estimated

        R2 : R-squared statistic for the regression model

        R2adj : Adjusted R-squared statistic for the regression model

        F : F-statistic for the regression model

        Fpv : p-value for the F-statistic computed

        Further information about these statistics and values can be
        found in any standard statistics textbook or online with the
        appropriate search query.

        """

        # Estimate the coefficients, which is performed as follows:
        #
        # 'A' = matrix of the independent variable observations,
        # 'b' = column array of the dependent variable observations
        # 'x' = column array of the coefficients to be estimated.
        #
        # The estimate is performed by solving A_T * A * x = A_T * b
        # for 'x', where A_T is the transpose of A.
        #
        # Notes:
        #   1) This method assumes that A_T * A has an inverse.
        #   2) If A * x = b has a solution, the initial equation
        #      will find that solution.

        try:
            self.inv_xx = inv(dot(self.x.T, self.x))
            xy = dot(self.x.T, self.y)
            self.b = dot(self.inv_xx, xy)

        # inv(dot(self.x.T, self.x)) could not be computed because
        # dot(self.x.T, self.x) is a singular matrix. This except
        # block raises a LinAlgError with a more informative error
        # message for the user.
        except LinAlgError:
            msg = ("\n\nYour matrix of independent observations is singular!"
                   "\nUnfortunately, that means we cannot compute an OLS"
                   "\nmodel for your provided data. Terminating immediately.")

            raise LinAlgError(msg)

        self.nobs = self.y.shape[0]
        self.ncoef = self.x.shape[1]
        self.df_e = self.nobs - self.ncoef
        self.df_r = self.ncoef - 1

        self.e = self.y - dot(self.x, self.b)
        self.sse = dot(self.e, self.e) / self.df_e
        self.se = sqrt(diagonal(self.sse * self.inv_xx))
        self.t = self.b / self.se
        self.p = (1 - stats.t.cdf(abs(self.t), self.df_e)) * 2

        self.R2 = 1 - self.e.var() / self.y.var()
        self.R2adj = 1 - (1 - self.R2) * ((self.nobs - 1) /
                                          (self.nobs - self.ncoef))

        self.F = (self.R2 / self.df_r) / ((1 - self.R2) / self.df_e)
        self.Fpv = 1 - stats.f.cdf(self.F, self.df_r, self.df_e)

    def dw(self):
        """

        Calculates the Durbin-Waston statistic for the regression model
        residuals and returns it.

        """

        de = diff(self.e, 1)
        dw = dot(de, de) / dot(self.e, self.e)

        return dw

    def omni(self):
        """

        Performs the Omnibus test for normality on the regression model
        residuals. Returns the Omnibus statistic and the p-value associated
        with that statistic. Note that this test requires that there are
        at least data observations. In this case, 'nan' is returned for
        both values.

        For more information about the Omnibus normality test, please read
        the documentation regarding the stats.normaltest method from the
        SciPy library.

        """

        try:
            return stats.normaltest(self.e)

        # fewer than eight observations
        except ValueError:
            return nan, nan

    def JB(self):
        """

        Calculates the residual skewness and kurtosis for the regression model
        residuals and then performs the JB test for normality. Returns the skew
        and kurtosis statistics as well as the JB statistic and its associated
        p-value.

        """

        skew = stats.skew(self.e)
        kurtosis = 3 + stats.kurtosis(self.e)

        JB = (self.nobs / 6) * (square(skew) + (1 / 4) * square(kurtosis - 3))
        JBpv = 1 - stats.chi2.cdf(JB, 2)

        return JB, JBpv, skew, kurtosis

    def ll(self):
        """

        Calculate model log-likelihood as well as the AIC and BIC for the
        data provided and returns all of the computed values. Note that the
        last two statistics provide information on how well suited the OLS
        model is with respect to other potential models of the data (assume
        finite set of such models).

        """

        ll = -(self.nobs / 2) * (1 + log(2 * pi)) - (
            self.nobs / 2) * log(dot(self.e, self.e) / self.nobs)
        aic = -2 * ll / self.nobs + (2 * self.ncoef / self.nobs)
        bic = -2 * ll / self.nobs + (self.ncoef * log(self.nobs)) / self.nobs

        return ll, aic, bic

    def summary(self):
        """

        Summarizes the results of the regression performed along with many
        relevant statistics (e.g. the Durbin-Watson statistic) and prints
        the results out into STDOUT, which is generally the console or
        terminal on which the code is being executed.

        """

        t = localtime()

        ll, aic, bic = self.ll()
        JB, JBpv, skew, kurtosis = self.JB()
        omni, omnipv = self.omni()

        print('\n==============================================================================')
        print("Dependent Variable: " + self.y_varnm)
        print("Method: Least Squares")
        print("Date: ", strftime("%a, %d %b %Y", t))
        print("Time: ", strftime("%H:%M:%S", t))
        print('# obs:               %5.0f' % self.nobs)
        print('# variables:     %5.0f' % self.ncoef)
        print('==============================================================================')
        print('variable     coefficient     std. Error      t-statistic     prob.')
        print('==============================================================================')
        for i in range(len(self.x_varnm)):
            print('''% -5s          % -5.6f     % -5.6f     % -5.6f     % -5.6f''' % tuple([self.x_varnm[i], self.b[i], self.se[i], self.t[i], self.p[i]]))
        print('==============================================================================')
        print('Models stats                         Residual stats')
        print('==============================================================================')
        print('R-squared            % -5.6f         Durbin-Watson stat  % -5.6f' % tuple([self.R2, self.dw()]))
        print('Adjusted R-squared   % -5.6f         Omnibus stat        % -5.6f' % tuple([self.R2adj, omni]))
        print('F-statistic          % -5.6f         Prob(Omnibus stat)  % -5.6f' % tuple([self.F, omnipv]))
        print('Prob (F-statistic)   % -5.6f			JB stat             % -5.6f' % tuple([self.Fpv, JB]))
        print('Log likelihood       % -5.6f			Prob(JB)            % -5.6f' % tuple([ll, JBpv]))
        print('AIC criterion        % -5.6f         Skew                % -5.6f' % tuple([aic, skew]))
        print('BIC criterion        % -5.6f         Kurtosis            % -5.6f' % tuple([bic, kurtosis]))
        print('==============================================================================')

    def to_file(self, filename=None):
        """

        Summarizes the results of the regression performed along with many
        relevant statistics (e.g. the Durbin-Watson statistic) and saves the
        results out into `filename`.

        Parameters
        ----------
        filename : string, optional
            The location of the file where the results will be stored. If no
            filename is provided, a default filename will be generated, and the
            results will be stored there.

        """

        t = localtime()

        ll, aic, bic = self.ll()
        JB, JBpv, skew, kurtosis = self.JB()
        omni, omnipv = self.omni()

        data = {}

        data['dependent_var'] = self.y
        data['method'] = 'least squares'

        data['date'] = strftime("%a, %d %b %Y", t)
        data['time'] = strftime("%H:%M:%S", t)

        data['obs_count'] = self.nobs
        data['var_count'] = self.ncoefs

        estimates = {}

        for i in range(len(self.x_varnm)):
            estimate = {}

            estimate['estimate'] = self.b[i]
            estimate['std_error'] = self.se[i]
            estimate['t_stat'] = self.t[i]
            estimate['p_val'] = self.p[i]

            estimates[self.x_varnm[i]] = estimate

        data['estimates'] = estimates

        data['r_squared'] = self.R2
        data['r_squared_adj'] = self.R2adj

        data['durbin_watson'] = self.dw()

        data['omnibus_stat'] = omni
        data['omnibus_p_val'] = omnipv

        data['f_stat'] = self.F
        data['f_stat_p_val'] = self.Fpv

        data['jb_stat'] = JB
        data['jb_stat_p_val'] = JBpv

        data['skew'] = skew
        data['kurtosis'] = kurtosis

        data['aic_stat'] = aic
        data['bic_stat'] = bic
        data['log_likelihood'] = ll

        filename = filename or strftime("%a_%d_%b_%Y_%H_%M_%S.json", t)

        with open(filename, 'w') as target:
            dump(data, target)

    def __str__(self):
        return "OLS Regression on " + str(self.x.shape[0]) + " Observations"

    __repr__ = __str__
    __bytes__ = __str__
    __unicode__ = __str__

if __name__ == "__main__":
    from numpy import array, column_stack

    x = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = array([i + 0.1 for i in x])

    reg = ols(x, y, x_varnm=['good_pred'], y_varnm='nice_var')
    reg.summary()

    x1 = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x2 = array([i + 0.1 for i in x1])
    x = column_stack((x1, x2))

    y = array([(-1)**(i % 2) for i in x1])

    reg = ols(x, y, x_varnm=['bad_pred', 'bad_pred2'], y_varnm='mean_var')
    reg.summary()
