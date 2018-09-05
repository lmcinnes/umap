# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
import numpy as np
import numba

_mock_identity = np.eye(2, dtype=np.float64)
_mock_ones = np.ones(2, dtype=np.float64)


@numba.njit(fastmath=True)
def euclidean(x, y):
    """Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


@numba.njit()
def standardised_euclidean(x, y, sigma=_mock_ones):
    """Euclidean distance standardised against a vector of standard
    deviations per coordinate.

    ..math::
        D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += ((x[i] - y[i]) ** 2) / sigma[i]

    return np.sqrt(result)


@numba.njit()
def manhattan(x, y):
    """Manhatten, taxicab, or l1 distance.

    ..math::
        D(x, y) = \sum_i |x_i - y_i|
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += np.abs(x[i] - y[i])

    return result


@numba.njit()
def chebyshev(x, y):
    """Chebyshev or l-infinity distance.

    ..math::
        D(x, y) = \max_i |x_i - y_i|
    """
    result = 0.0
    for i in range(x.shape[0]):
        result = max(result, np.abs(x[i] - y[i]))

    return result


@numba.njit()
def minkowski(x, y, p=2):
    """Minkowski distance.

    ..math::
        D(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{\frac{1}{p}}

    This is a general distance. For p=1 it is equivalent to
    manhattan distance, for p=2 it is Euclidean distance, and
    for p=infinity it is Chebyshev distance. In general it is better
    to use the more specialised functions for those distances.
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (np.abs(x[i] - y[i])) ** p

    return result ** (1.0 / p)


@numba.njit()
def weighted_minkowski(x, y, w=_mock_ones, p=2):
    """A weighted version of Minkowski distance.

    ..math::
        D(x, y) = \left(\sum_i w_i |x_i - y_i|^p\right)^{\frac{1}{p}}

    If weights w_i are inverse standard deviations of data in each dimension
    then this represented a standardised Minkowski distance (and is
    equivalent to standardised Euclidean distance for p=1).
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (w[i] * np.abs(x[i] - y[i])) ** p

    return result ** (1.0 / p)


@numba.njit()
def mahalanobis(x, y, vinv=_mock_identity):
    result = 0.0

    diff = np.empty(x.shape[0], dtype=np.float64)

    for i in range(x.shape[0]):
        diff[i] = x[i] - y[i]

    for i in range(x.shape[0]):
        tmp = 0.0
        for j in range(x.shape[0]):
            tmp += vinv[i, j] * diff[j]
        result += tmp * diff[i]

    return np.sqrt(result)


@numba.njit()
def hamming(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        if x[i] != y[i]:
            result += 1.0

    return float(result) / x.shape[0]


@numba.njit()
def canberra(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        denominator = np.abs(x[i]) + np.abs(y[i])
        if denominator > 0:
            result += np.abs(x[i] - y[i]) / denominator

    return result


@numba.njit()
def bray_curtis(x, y):
    numerator = 0.0
    denominator = 0.0
    for i in range(x.shape[0]):
        numerator += np.abs(x[i] - y[i])
        denominator += np.abs(x[i] + y[i])

    if denominator > 0.0:
        return float(numerator) / denominator
    else:
        return 0.0


@numba.njit()
def jaccard(x, y):
    num_non_zero = 0.0
    num_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_non_zero += x_true or y_true
        num_equal += x_true and y_true

    if num_non_zero == 0.0:
        return 0.0
    else:
        return float(num_non_zero - num_equal) / num_non_zero


@numba.njit()
def matching(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return float(num_not_equal) / x.shape[0]


@numba.njit()
def dice(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (2.0 * num_true_true + num_not_equal)


@numba.njit()
def kulsinski(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0:
        return 0.0
    else:
        return float(num_not_equal - num_true_true + x.shape[0]) / (
            num_not_equal + x.shape[0]
        )


@numba.njit()
def rogers_tanimoto(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


@numba.njit()
def russellrao(x, y):
    num_true_true = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true

    if num_true_true == np.sum(x != 0) and num_true_true == np.sum(y != 0):
        return 0.0
    else:
        return float(x.shape[0] - num_true_true) / (x.shape[0])


@numba.njit()
def sokal_michener(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


@numba.njit()
def sokal_sneath(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (0.5 * num_true_true + num_not_equal)


@numba.njit()
def haversine(x, y):
    if x.shape[0] != 2:
        raise ValueError("haversine is only defined for 2 dimensional data")
    sin_lat = np.sin(0.5 * (x[0] - y[0]))
    sin_long = np.sin(0.5 * (x[1] - y[1]))
    result = np.sqrt(sin_lat ** 2 + np.cos(x[0]) * np.cos(y[0]) * sin_long ** 2)
    return 2.0 * np.arcsin(result)


@numba.njit()
def yule(x, y):
    num_true_true = 0.0
    num_true_false = 0.0
    num_false_true = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_true_false += x_true and (not y_true)
        num_false_true += (not x_true) and y_true

    num_false_false = x.shape[0] - num_true_true - num_true_false - num_false_true

    if num_true_false == 0.0 or num_false_true == 0.0:
        return 0.0
    else:
        return (2.0 * num_true_false * num_false_true) / (
            num_true_true * num_false_false + num_true_false * num_false_true
        )


@numba.njit()
def cosine(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return 1.0
    else:
        return 1.0 - (result / np.sqrt(norm_x * norm_y))


@numba.njit()
def correlation(x, y):
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / np.sqrt(norm_x * norm_y))

@numba.njit()
def approx_log_Gamma(x):
    if x == 1:
        return 0
    #x2= 1/(x*x);
    return x*np.log(x) - x + 0.5*np.log(2.0*np.pi/x) + 1.0/(x*12.0) #+ x2*(-1.0/360.0) + x2* (1.0/1260.0 + x2*(-1.0/(1680.0)  + x2*(1.0/1188.0 + x2*(-691.0/360360.0 + x2*(1.0/156.0 + x2*(-3617.0/122400.0 + x2*(43687.0/244188.0 + x2*(-174611.0/125400.0) + x2*(77683.0/5796.0 + x2*(-236364091.0/1506960.0 + x2*(657931.0/300.0))))))))))))
                
@numba.njit()
def log_beta(x,y):
    a = min(x,y)
    b = max(x,y)
    if b < 5:
        value = -np.log(b)
        for i in range(1,a):
            value += np.log(i)-np.log(b+i)
        return value
    else:
        return approx_log_Gamma(x) + approx_log_Gamma(y) - approx_log_Gamma(x + y)

@numba.njit()
def log_single_beta(x):
    return np.log(2.0)*(-2.0*x+0.5) + 0.5*np.log(2.0*np.pi/x) + 0.125/x
# + x2*(-1.0/192.0 + x2* (1.0/640.0 + x2*(-17.0/(14336.0) + x2*(31.0/18432.0 + x2*(-691.0/180224.0 + x2*(5461.0/425984.0 + x2*(-929569.0/15728640.0 + x2*(3189151.0/8912896.0 + x2*(-221930581.0/79691776.0) + x2*(4722116521.0/176160768.0 + x2*(-968383680827.0/3087007744.0 + x2*(14717667114151.0/3355443200.0 ))))))))))))

@numba.njit()
def ll_dirichlet(data1, data2):
    """ The symmetric relative log likelihood of rolling data2 vs data 1 in n trials on a die that rolled data1 in sum(data1) trials. 
    
    ..math::
        D(data1, data2) = DirichletMultinomail(data2 | data1)  
    """

    n1 = np.sum(data1)
    n2 = np.sum(data2)

    log_b = 0.0
    self_denom1 = 0.0
    self_denom2 = 0.0

    for i in range(data1.shape[0]):
        if data1[i]*data2[i] > 0.9 :
            log_b += log_beta(data1[i], data2[i])
            self_denom1 += log_single_beta(data1[i]) 
            self_denom2 += log_single_beta(data2[i]) 

        else:
            if data1[i] > 0.9:
                self_denom1 += log_single_beta(data1[i])    

                
            if data2[i] > 0.9:
                self_denom2 += log_single_beta(data2[i]) 

#    return np.sqrt(1.0/n2*(log_denom1 - np.log(n2) - log_beta(n1,n2) - (self_denom2 - np.log(n2) - log_beta(n2,n2))) + 1.0/n1*(log_denom2 -np.log(n1) - log_beta(n2,n1) - (self_denom1 - np.log(n1) - log_beta(n1,n1))))

    return np.sqrt(1.0/n2*(log_b - log_beta(n1,n2) - (self_denom2 - log_single_beta(n2))) + 1.0/n1*(log_b - log_beta(n2,n1) - (self_denom1 - log_single_beta(n1))))


named_distances = {
    # general minkowski distances
    "euclidean": euclidean,
    "l2": euclidean,
    "manhattan": manhattan,
    "taxicab": manhattan,
    "l1": manhattan,
    "chebyshev": chebyshev,
    "linfinity": chebyshev,
    "linfty": chebyshev,
    "linf": chebyshev,
    "minkowski": minkowski,
    # Standardised/weighted distances
    "seuclidean": standardised_euclidean,
    "standardised_euclidean": standardised_euclidean,
    "wminkowski": weighted_minkowski,
    "weighted_minkowski": weighted_minkowski,
    "mahalanobis": mahalanobis,
    # Other distances
    "canberra": canberra,
    "cosine": cosine,
    "correlation": correlation,
    "haversine": haversine,
    "braycurtis": bray_curtis,
    "ll_dirichlet": ll_dirichlet,
    # Binary distances
    "hamming": hamming,
    "jaccard": jaccard,
    "dice": dice,
    "matching": matching,
    "kulsinski": kulsinski,
    "rogerstanimoto": rogers_tanimoto,
    "russellrao": russellrao,
    "sokalsneath": sokal_sneath,
    "sokalmichener": sokal_michener,
    "yule": yule,
}
