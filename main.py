import numpy as np
import pandas as pd

from model.lib.model import AttributionModel
from model.lib.preprocess import PreProcess
from model.lib.utils import Cv_estimate, Cm_estimate, project_vectors


def main(y, X, Z, uncorr, corr, init=1955, end=1995):
    """
    Main method for using Ribes et al. (2017) algorithm including observational and model error
    :param y: numpy.ndarray
        Vector with nt observations
    :param X: numpy.ndarray
        nt x nf array with model data where nf is the number of forcings
    :param Z: numpy.ndarray
        nt x nz array of pseudo-observations used to calculate internal variability covariance matrix
    :param uncorr: numpy.ndarray
        nt x nz array of ensemble of observations representing uncorrelated error
    :param corr: numpy.ndarray
        nt x nz array of ensemble of observations representing correlated error
    :param init: int
        year to start analysis
    :param end: end
        year to end the analysis
    :return:
    """
    # preprocess - all
    p = PreProcess(y, X, Z)
    Z1, Z2 = p.extract_Z2(frac=0.5)
    yc, Xc, Z1c, Z2c = p.proj_fullrank(Z1, Z2)

    # Compute covariance matrices for internal variability
    Cv1 = p.creg(Z1c.T, method='ledoit')
    Cv2 = p.creg(Z2c.T, method='ledoit')

    # scale covariance matrix by number of ensemble members
    nt = len(y)
    Cx_nat = Cm_estimate('historicalNat', Cv2, X[:, 1], how_nr='pandas', init=init, end=end) + \
             Cv_estimate('historicalNat', Cv2, init=init, end=end, how_nr='pandas')
    Cx_ghg = Cm_estimate('historicalGHG', Cv2, X[:, 2], how_nr='pandas', init=init, end=end) + \
             Cv_estimate('historicalGHG', Cv2, init=init, end=end, how_nr='pandas')
    Cx_oa = Cm_estimate('historicalOA', Cv2, X[:, 0], how_nr='loadtxt', init=init, end=end) + \
            Cv_estimate('historicalOA', Cv2, init=init, end=end, how_nr='loadtxt')

    # project the uncorrelated and correlated errors
    Zc_uncorr = project_vectors(nt, uncorr)
    Zc_corr = project_vectors(nt, corr)

    # and regularize its covariance matrix
    C_uncorr = p.creg(Zc_uncorr.T, method='ledoit')
    C_corr = p.creg(Zc_corr.T, method='ledoit')

    Cy = Cv1 + C_uncorr + C_corr
    Cxi = np.stack([Cx_oa, Cx_nat, Cx_ghg], axis=0)

    # starts attribution model
    m = AttributionModel(Xc, yc)
    y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat = m.ribes(Cxi, Cy)

    return y_star_hat, Xi_star_hat, Cy_star_hat, Cxi_star_hat

#################################################################


if __name__ == "__main__":

    df_y = pd.read_csv('data/obs/CRUTEM4_1955_1995.csv', index_col=0, parse_dates=True)
    y = df_y['anomaly'].values

    df_uncorr = pd.read_csv('data/obs/CRUTEM4_uncorrelated_1955_1995.csv', index_col=0, parse_dates=True)
    df_corr = pd.read_csv('data/obs/CRUTEM4_correlated_1955_1995.csv', index_col=0, parse_dates=True)

    Z_uncorr = df_uncorr.values
    Z_corr = df_corr.values

    # multi model ensemble
    df_X1 = pd.read_csv('data/model/historical/ensmean/CESM-LE_historical_1955_1995.csv', index_col=0, parse_dates=True)
    df_X2 = pd.read_csv('data/model/historicalNat/ensmean/CESM1-CAM5_historicalNat_1955_1995.csv', index_col=0, parse_dates=True)
    df_X3 = pd.read_csv('data/model/historicalGHG/ensmean/CESM1-CAM5_historicalGHG_1955_1995.csv', index_col=0, parse_dates=True)

    X1 = df_X1['anomaly'].values
    X2 = df_X2['anomaly'].values
    X3 = df_X3['anomaly'].values

    X = np.stack([X1 - X2 - X3, X2, X3], axis=1)

    df_Z = pd.read_csv('data/model/historical/ensemble/CESM-LE_historical_1955_1995.csv', index_col=0, parse_dates=True)
    Z = df_Z.values


    main(y, X, Z, Z_uncorr, Z_corr, init=1955, end=1995)



