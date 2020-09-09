###################################################################################
"""
  Modified scipy chi-square function to return the whole array of chi values for every 
variable-class pair instead of the average value for all the classes coresponding to a variable.
The original implementation can be accessed at:
https://github.com/scipy/scipy/tree/master/scipy/stats
"""
###################################################################################
import numpy as np
from scipy import special, stats
from scipy.sparse import issparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
                     safe_mask)
from sklearn.utils.extmath import safe_sparse_dot, row_norms

def _chisquare(f_obs, f_exp):

    f_obs = np.asarray(f_obs, dtype=np.float64)

    k = len(f_obs)
    # Reuse f_obs for chi-squared statistics
    chisq = f_obs
    chisq -= f_exp
    chisq **= 2
    with np.errstate(invalid="ignore"):
        chisq /= f_exp
    #chisq = chisq.sum(axis=0)
    return chisq

def chi2(X, y):
    X = check_array(X, accept_sparse='csr')
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(Y.T, X)  # n_classes * n_features

    feature_count = X.sum(axis=0).reshape(1, -1)
    class_prob = Y.mean(axis=0).reshape(1, -1)
    expected = np.dot(class_prob.T, feature_count)

    return _chisquare(observed, expected)
