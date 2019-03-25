import os

import numpy as np
import pandas as pd

from glob import glob
from scipy import stats

def speco(C):
    """
    This function computes eigenvalues and eigenvectors, in descending order
    :param C: numpy.ndarray
        A p x p symetric real matrix
    :return:
    P: numpy.ndarray
        The eigenvectors (P[:, i] is the ist eigenvector)
    D: numpy.ndarray
        The eigenvalues as a diagonal matrix
    """

    # Compute eigenvalues and eigenvectors (the eigenvectors are non unique so
    # the values may change from one software to another e.g. python, matlab,
    # scilab)
    D0, P0 = np.linalg.eig(C)

    # Take real part (to avoid numeric noise, eg small complex numbers)
    if np.max(np.imag(D0)) / np.max(np.real(D0)) > 1e-12:
        raise ValueError("Matrix is not symmetric")   

    # Check that C is symetric (<=> real eigen-values/-vectors)
    P1 = np.real(P0)
    D1 = np.real(D0)

    # sort eigenvalues in descending order and
    # get their indices to order the eigenvector
    Do = np.sort(D1)[::-1]
    o = np.argsort(D1)[::-1]

    P = P1[:, o]
    D = np.diag(Do)

    return P, D

def chi2_test(d_cons, df):
    """
    Check whether it is from a chi-squared distribution or not

    :param d_cons: float
        -2 log-likelihood
    :param df: int
        Degrees of freedom
    :return:
    pv_cons: float
        p-value for the test
    """
    rien = stats.chi2.cdf(d_cons, df=df)
    pv_cons = 1. - rien

    return pv_cons

def project_vectors(nt, X):
    """

    :param nt:
    :param X:
    :return:
    """
    M = np.eye(nt, nt) - np.ones((nt, nt)) / nt

    # Eigen-vectors/-values of M; note that rk(M)=nt-1, so M has one eigenvalue equal to 0.
    u, d = speco(M)

    # (nt-1) first eigenvectors (ie the ones corresponding to non-zero eigenvalues)
    U = u[:, :nt - 1].T

    return np.dot(U, X)

def SSM(exp, X_mm, init=1955, end=1995):
    """

    :param exp:
    :param X_mm:
    :param init:
    :param end:
    :return:
    """

    ifiles = glob('data/model/%s/ensmean/*_%s_%s.csv' % (exp, init, end))

    df = pd.DataFrame()
    for ifile in ifiles:
        df_temp = pd.read_csv(ifile, index_col=0, parse_dates=True)
        df = pd.concat([df, df_temp['anomaly'].to_frame(os.path.basename(ifile)[:-4])], axis=1)

    # remove columns (ensemble members with nan)
    df.dropna(inplace=True, axis=1)
    # gets ensemble values and multi model (mm) ensemble
    X = df.values

    # project the data
    Xc = project_vectors(X.shape[0], X)
    Xc_mm = project_vectors(X.shape[0], X_mm.reshape((X.shape[0], 1)))

    return np.diag(((Xc - Xc_mm) ** 2.).sum(axis=1))

def get_nruns(exp, how='pandas', init=1955, end=1995):
    """

    :param exp:
    :param how:
    :param init:
    :param end:
    :return:
    """

    if how == 'pandas':
        ifiles = glob('data/model/%s/ensemble/*_%s_%s.csv' % (exp, init, end))
        nruns = []

        for ifile in sorted(ifiles):
            df_temp = pd.read_csv(ifile, index_col=0, parse_dates=True)
            nruns.append(len(df_temp.columns))

        nruns = np.array(nruns)
    elif how == 'loadtxt':
        nruns = np.loadtxt('data/model/%s/ensemble/nruns_%s.txt' % (exp, exp))

    return nruns

def Cm_estimate(exp, Cv, X_mm, how_nr='pandas', init=1955, end=1995):
    """

    :param exp:
    :param Cv:
    :param SSM:
    :param nruns:
    :param nm:
    :return:
    """

    _SSM = SSM(exp, X_mm, init=init, end=end)

    # nruns - number of runs / nm - number of models
    nruns = get_nruns(exp, how=how_nr, init=init, end=end)
    nm = len(nruns)

    Cv_all = np.zeros(Cv.shape)
    for nr in nruns:
        Cv_all += Cv / nr

    Cm_hat = (1. / (nm - 1.)) * (_SSM - ((nm - 1.) / nm) * Cv_all)

    # set negative eigenvalues to zero and recompose the signal
    S, X = np.linalg.eig(Cm_hat)
    S[S < 0] = 0
    Cm_pos_hat = np.linalg.multi_dot([X, np.diag(S), np.linalg.inv(X)]) # spectral decomposition

    Cm_pos_hat = (1. + (1. / nm)) * Cm_pos_hat

    return Cm_pos_hat

def Cv_estimate(exp, Cv, how_nr='pandas', init=1955, end=1995):

    # nruns - number of runs / nm - number of models
    nruns = get_nruns(exp, how=how_nr, init=init, end=end)
    nm = len(nruns)

    Cv_all = np.zeros(Cv.shape)
    for nr in nruns:
        Cv_all += Cv / nr

    Cv_estimate = (1. / (nm ** 2.) * Cv_all)

    return Cv_estimate


if __name__ == "__main__":
    T = 11
    M = np.eye(T, T) - np.ones((T, T)) / T

    speco(M)
