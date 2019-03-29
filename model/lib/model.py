import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats

from .preprocess import PreProcess
from .utils import chi2_test

class AttributionModel():
    """
    A class for attribution models. The OLS implementation is heavily based on Aurelien Ribes (CNRM-GAME) scilab code
    (see more in 'preprocess.py'). Also, Aurelien Ribes model proposed in 2017 is implemented following the reference:
        Ribes, Aurelien, et al. (2017) A new statistical approach to climate change detection and attribution.
        Climate Dynamics.

    :attribute X: numpy.ndarray
        Array of size nt x nf, where nf is the number of forcings, with the model output as a timeseries
    :attribute y: numpy.ndarray
        Array of size nt with observations as a timeseries

    :method ols(self, Cf, Proj, Z2, cons_test='AT99'):
        Ordinary Least Square (OLS) estimation of beta from the linear model y = beta * X + epsilon
    """

    def __init__(self, X, y):
        """
        :param X: numpy.ndarray
            Array of size nt x nf, where nf is the number of forcings, with the model output as a timeseries
        :param y: numpy.ndarray
            Array of size nt with observations as a timeseries
        """
        self.y = y
        self.X = X
        self.nt = y.shape[0]
        self.nr = self.nt - 1 # 1 stands for the number of spatial patterns (dealing only with timeseries)
        self.I = X.shape[1]
        
    def ols(self, Cf, Proj, Z2, cons_test='AT99'):
        """
        Ordinary Least Square (OLS) estimation of beta from the linear model y = beta * X + epsilon as discussed in the
        following reference:
            Allen, Myles R., and Simon FB Tett (1999) Checking for model consistency in optimal fingerprinting.
            Climate Dynamics.

        :param Cf: numpy.ndarray
            Covariance matrix. Be sure that Cf is invertible to use this model (look at PreProcess class)
        :param Proj: numpy.ndarray
            Array of zeros and ones, indicating which forcings in each simulation
        :param Z2: numpy.ndarray
            Array of size (nz1 x p) of control simulation used to compute consistensy test
        :param cons_test: str
            Which consistency test to be used
            - 'AT99' the formula provided by Allen & Tett (1999) (default)
        :return:
        Beta_hat: dict
            Dictionary with estimation of beta_hat and the upper and lower condiference intervals
        """

        # computes the covariance inverse
        Cf1 = np.linalg.inv(Cf)

        _Ft = np.linalg.multi_dot([self.X.T, Cf1, self.X])
        _Ft1 = np.linalg.inv(_Ft)
        Ft = np.linalg.multi_dot([_Ft1, self.X.T, Cf1]).T

        _y = self.y.reshape((self.nt, 1))
        beta_hat = np.linalg.multi_dot([_y.T, Ft, Proj.T])

        # 1-D confidence interval
        nz2 = Z2.shape[1]
        Z2t = Z2.T
        Var_valid = np.dot(Z2t.T, Z2t) / nz2
        Var_beta_hat = np.linalg.multi_dot([Proj, Ft.T, Var_valid, Ft, Proj.T])

        beta_hat_inf = beta_hat - 2. * stats.t.cdf(0.95, df=nz2) * np.sqrt(np.diag(Var_beta_hat).T)
        beta_hat_sup = beta_hat + 2. * stats.t.cdf(0.95, df=nz2) * np.sqrt(np.diag(Var_beta_hat).T)

        # consistency check
        epsilon = _y - np.linalg.multi_dot([self.X, np.linalg.inv(Proj), beta_hat.T])

        if cons_test == 'AT99': # formula provided by Allen & Tett (1999)
            d_cons = np.linalg.multi_dot([epsilon.T, np.linalg.pinv(Var_valid), epsilon]) / (self.nr - self.I)
            rien = stats.f.cdf(d_cons, dfn=self.nr-self.I, dfd=nz2)
            pv_cons = 1 - rien

        print("Consistency test: %s p-value: %.5f" % (cons_test, pv_cons))

        Beta_hat = {'beta_hat': beta_hat[0], 'beta_hat_inf': beta_hat_inf[0], 'beta_hat_sup': beta_hat_sup[0]}
        
        return Beta_hat 

    def ribes(self, Cxi, Cy):
        """
        Aurelien Ribes model proposed in 2017 is implemented following the reference:
        Ribes, Aurelien, et al. (2017) A new statistical approach to climate change detection and attribution.
        Climate Dynamics. It considers the following set of equations:

            Y_star = sum(X_star_i) for i from 1 to nf where nf is the number of forcings
            Y = Y_star + epsilon_y
            Xi = X_star_i + epsilon_xi

        Where epislon_y ~ N(0, Cy) and epislon_xi ~ N(0, Cxi)

        :param Cxi: numpy.ndarray
            Covariance matrix for each of the forcings Xi. Should be a 3D array (nt, nt, nf)
        :param Cy: numpy.ndarray
            Covariance matrix for the observations.
        :return:
        """
        X = self.X.sum(axis=1)
        Cx = Cxi.sum(axis=0)

        # Estimate the true state of variables (y) and (Xi) y_star and X_star_i using the MLE
        # y_star_hat and Xi_star_hat, respectively
        Xi_star_hat = np.zeros(self.X.shape)
        y_star_hat = self.y + np.linalg.multi_dot([Cy, np.linalg.inv(Cy + Cx), (X - self.y)])
        for i in range(Xi_star_hat.shape[1]):
            Xi_star_hat[:, i] = self.X[:, i] + np.linalg.multi_dot([Cxi[i], np.linalg.inv(Cy + Cx), (self.y - X)])

        # calculates variance for Y_star_hat
        Cy_star_hat = np.linalg.inv(np.linalg.inv(Cy) + np.linalg.inv(Cx))

        # calculates variance for Xi_star_hat
        Cxi_star_hat = np.zeros(Cxi.shape)
        for i in range(Cxi_star_hat.shape[0]):
            Cxi_temp = Cxi * 1.
            # sum for every j different than i
            Cxi_temp[i] = 0.
            Cxi_sum = Cxi_temp.sum(axis=0)

            Cxi_star_hat[i] = np.linalg.inv(np.linalg.inv(Cxi[i]) + np.linalg.inv(Cy + Cxi_sum))

        # hypothesis test: compare with chi-square distribution
        print('#' * 60)
        print('Hypothesis testing p-value for Chi-2 distribution and Maximum Likelihood ...')

        # (internal variability only)
        d_cons = np.linalg.multi_dot([self.y.T, np.linalg.inv(Cy), self.y])
        print('%30s: %.7f (%.7f)' % ('Internal variability only', chi2_test(d_cons, self.nt), np.exp(d_cons / -2.)))

        # (all forcings)
        d_cons = np.linalg.multi_dot([(self.y - X).T, np.linalg.inv(Cy + Cx), (self.y - X)])
        print('%30s: %.7f (%.7f)' % ('All forcings', chi2_test(d_cons, self.nt), np.exp(d_cons / -2.)))

        # (individual forcings)
        for i in range(self.X.shape[1]):
            d_cons = np.linalg.multi_dot([(self.y - self.X[:, i]).T, np.linalg.inv(Cy + Cxi[i]), (self.y - self.X[:, i])])
            print('%30s: %.7f (%.7f)' % ('Forcing no %d only' % (i+1), chi2_test(d_cons, self.nt), np.exp(d_cons / -2.)))

        return y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat

if __name__ == "__main__":

    # observations
    y = np.loadtxt('../example/HadCRUT4_1901-2010.asc')
    
    # forcing patterns
    X1 = np.loadtxt('../example/CNRM-CM5_ANT_1901-2010.asc')
    X2 = np.loadtxt('../example/CNRM-CM5_NAT_1901-2010.asc')
    
    X = np.stack([X1, X2], axis=1)

    # internal variability ensemble
    Z = np.loadtxt('../example/CTLruns.asc')
    nt = len(y)
    nz = int(len(Z) / nt)
    Z = Z.reshape((nz, nt)).T # every nt timesteps vector should be a column

    p = PreProcess(y, X, Z)
    Z1, Z2 = p.extract_Z2(frac=0.5)
    yc, Xc, Z1c, Z2c = p.proj_fullrank(Z1, Z2)

    # # OLS test
    # alpha1_array = np.arange(1e-11, 1e-3, 1e-6)
    #
    # beta_hat = []
    # beta_hat_sup = []
    # beta_hat_inf = []
    #
    # for alpha1 in alpha1_array:
    #     Cr = p.creg(Z1c.T, method='specified', alpha1=alpha1)
    #
    #     m = AttributionModel(Xc, yc)
    #     beta_ols = m.ols(Cr, np.array([[1, 0], [0, 1]]), Z2c)
    #
    #     beta_hat.append(beta_ols['beta_hat'])
    #     beta_hat_inf.append(beta_ols['beta_hat_inf'])
    #     beta_hat_sup.append(beta_ols['beta_hat_sup'])
    #
    # beta_hat = np.array(beta_hat)
    # beta_hat_sup = np.array(beta_hat_sup)
    # beta_hat_inf = np.array(beta_hat_inf)
    #
    # Cr = p.creg(Z1c.T, method='ledoit')
    #
    # m = AttributionModel(Xc, yc)
    # beta_ols = m.ols(Cr, np.array([[1, 0], [0, 1]]), Z2c)
    #
    # fig, [ax1, ax2] = plt.subplots(1, 2)
    #
    # ax1.plot(alpha1_array, beta_hat[:, 0], c='C0', linewidth=2)
    # ax1.fill_between(alpha1_array, beta_hat_inf[:, 0], beta_hat_sup[:, 0], color='C0', alpha=0.3)
    # ax1.axhline(beta_ols['beta_hat'][0], color='k', linewidth=2, linestyle='--')
    #
    # ax2.plot(alpha1_array, beta_hat[:, 1], c='C1', linewidth=2)
    # ax2.fill_between(alpha1_array, beta_hat_inf[:, 1], beta_hat_sup[:, 1], color='C1', alpha=0.3)
    # ax2.axhline(beta_ols['beta_hat'][1], color='k', linewidth=2, linestyle='--')
    #
    # for ax in [ax1, ax2]:
    #     ax.set_xscale("log")
    #
    # plt.show()

    Cr = p.creg(Z1c.T, method='ledoit')

    m = AttributionModel(Xc, yc)
    beta_ols = m.ols(Cr, np.array([[1, 0], [0, 1]]), Z2c)

    Cy = p.creg(Z1c.T, method='ledoit')
    Cx = p.creg(Z2c.T, method='ledoit')

    Cx = Cx[np.newaxis, :, :]
    Cxi = np.concatenate([Cx, Cx], axis=0)

    y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat = m.ribes(Cxi, Cy)
    print(beta_ols)

    fig, ax = plt.subplots()
    ax.plot(X1+X2, 'o-', label='ALL')
    ax.plot(X2, 'o-', label='NAT')
    ax.plot(X1, 'o-', label='ANT')
    ax.plot(y, 'o-', c='k', label='OBS')
    ax.legend()
    plt.show()    

    # X1_trend = stats.linregress(np.arange(X.shape[0]), X1)[0]
    # X2_trend = stats.linregress(np.arange(X.shape[0]), X2)[0]

    # X = np.array([X1_trend, X2_trend])
    # y = stats.linregress(np.arange(y.shape[0]), y)[0]

    # X = X[np.newaxis, :]
    # y = np.array([[y]])

    # Z_trend = np.zeros(Z.shape[1])
    # for i in range(Z_trend.shape[0]):
    #     Z_trend[i] = stats.linregress(np.arange(Z.shape[0]), Z[:, i])[0]

    # Z = Z_trend[np.newaxis, :]

    # p = PreProcess(y, X, Z)
    # Z1, Z2 = p.extract_Z2(frac=0.5)

    # Cy = p.creg(Z1.T, method='ledoit')
    # print(Cy)

    # m = AttributionModel(X, y)
    # n, p = Z.T.shape
    # Cy = np.dot(Z, Z.T) / n
    # Cx = np.dot(Z, Z.T) / n

    # Cx = Cx[np.newaxis, :, :]

    # Cxi = np.concatenate([Cx, Cx], axis=0)
    # y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat = m.ribes(Cxi, Cy)
