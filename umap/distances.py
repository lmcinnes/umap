# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
import numba
import numpy as np
import scipy.stats
from sklearn.metrics import pairwise_distances

_mock_identity = np.eye(2, dtype=np.float64)
_mock_cost = 1.0 - _mock_identity
_mock_ones = np.ones(2, dtype=np.float64)


@numba.njit()
def sign(a):
    if a < 0:
        return -1
    else:
        return 1


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


@numba.njit(fastmath=True)
def euclidean_grad(x, y):
    """Standard euclidean distance and its gradient.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
        \frac{dD(x, y)}{dx} = (x_i - y_i)/D(x,y)
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    d = np.sqrt(result)
    grad = (x - y) / (1e-6 + d)
    return d, grad


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


@numba.njit(fastmath=True)
def standardised_euclidean_grad(x, y, sigma=_mock_ones):
    """Euclidean distance standardised against a vector of standard
    deviations per coordinate with gradient.

    ..math::
        D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2 / sigma[i]
    d = np.sqrt(result)
    grad = (x - y) / (1e-6 + d * sigma)
    return d, grad


@numba.njit()
def manhattan(x, y):
    """Manhattan, taxicab, or l1 distance.

    ..math::
        D(x, y) = \sum_i |x_i - y_i|
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += np.abs(x[i] - y[i])

    return result


@numba.njit()
def manhattan_grad(x, y):
    """Manhattan, taxicab, or l1 distance with gradient.

    ..math::
        D(x, y) = \sum_i |x_i - y_i|
    """
    result = 0.0
    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        result += np.abs(x[i] - y[i])
        grad[i] = np.sign(x[i] - y[i])
    return result, grad


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
def chebyshev_grad(x, y):
    """Chebyshev or l-infinity distance with gradient.

    ..math::
        D(x, y) = \max_i |x_i - y_i|
    """
    result = 0.0
    max_i = 0
    for i in range(x.shape[0]):
        v = np.abs(x[i] - y[i])
        if v > result:
            result = v
            max_i = i
    grad = np.zeros(x.shape)
    grad[max_i] = np.sign(x[max_i] - y[max_i])

    return result, grad


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
def minkowski_grad(x, y, p=2):
    """Minkowski distance with gradient.

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

    grad = np.empty(x.shape[0], dtype=np.float32)
    for i in range(x.shape[0]):
        grad[i] = (
            pow(np.abs(x[i] - y[i]), (p - 1.0))
            * sign(x[i] - y[i])
            * pow(result, (1.0 / (p - 1)))
        )

    return result ** (1.0 / p), grad


@numba.njit()
def poincare(u, v):
    """Poincare distance.

    ..math::
        \delta (u, v) = 2 \frac{ \lVert  u - v \rVert ^2 }{ ( 1 - \lVert  u \rVert ^2 ) ( 1 - \lVert  v \rVert ^2 ) }
        D(x, y) = \operatorname{arcosh} (1+\delta (u,v))
    """
    sq_u_norm = np.sum(u * u)
    sq_v_norm = np.sum(v * v)
    sq_dist = np.sum(np.power(u - v, 2))
    return np.arccosh(1 + 2 * (sq_dist / ((1 - sq_u_norm) * (1 - sq_v_norm))))


@numba.njit()
def hyperboloid_grad(x, y):
    s = np.sqrt(1 + np.sum(x ** 2))
    t = np.sqrt(1 + np.sum(y ** 2))

    B = s * t
    for i in range(x.shape[0]):
        B -= x[i] * y[i]

    if B <= 1:
        B = 1.0 + 1e-8

    grad_coeff = 1.0 / (np.sqrt(B - 1) * np.sqrt(B + 1))

    # return np.arccosh(B), np.zeros(x.shape[0])

    grad = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        grad[i] = grad_coeff * (((x[i] * t) / s) - y[i])

    return np.arccosh(B), grad


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
def weighted_minkowski_grad(x, y, w=_mock_ones, p=2):
    """A weighted version of Minkowski distance with gradient.

    ..math::
        D(x, y) = \left(\sum_i w_i |x_i - y_i|^p\right)^{\frac{1}{p}}

    If weights w_i are inverse standard deviations of data in each dimension
    then this represented a standardised Minkowski distance (and is
    equivalent to standardised Euclidean distance for p=1).
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (w[i] * np.abs(x[i] - y[i])) ** p

    grad = np.empty(x.shape[0], dtype=np.float32)
    for i in range(x.shape[0]):
        grad[i] = (
            w[i] ** p
            * pow(np.abs(x[i] - y[i]), (p - 1.0))
            * sign(x[i] - y[i])
            * pow(result, (1.0 / (p - 1)))
        )

    return result ** (1.0 / p), grad


@numba.njit()
def mahalanobis(x, y, vinv=_mock_identity):
    result = 0.0

    diff = np.empty(x.shape[0], dtype=np.float32)

    for i in range(x.shape[0]):
        diff[i] = x[i] - y[i]

    for i in range(x.shape[0]):
        tmp = 0.0
        for j in range(x.shape[0]):
            tmp += vinv[i, j] * diff[j]
        result += tmp * diff[i]

    return np.sqrt(result)


@numba.njit()
def mahalanobis_grad(x, y, vinv=_mock_identity):
    result = 0.0

    diff = np.empty(x.shape[0], dtype=np.float32)

    for i in range(x.shape[0]):
        diff[i] = x[i] - y[i]

    grad_tmp = np.zeros(x.shape)
    for i in range(x.shape[0]):
        tmp = 0.0
        for j in range(x.shape[0]):
            tmp += vinv[i, j] * diff[j]
            grad_tmp[i] += vinv[i, j] * diff[j]
        result += tmp * diff[i]
    dist = np.sqrt(result)
    grad = grad_tmp / (1e-6 + dist)
    return dist, grad


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
def canberra_grad(x, y):
    result = 0.0
    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        denominator = np.abs(x[i]) + np.abs(y[i])
        if denominator > 0:
            result += np.abs(x[i] - y[i]) / denominator
            grad[i] = (
                np.sign(x[i] - y[i]) / denominator
                - np.abs(x[i] - y[i]) * np.sign(x[i]) / denominator ** 2
            )

    return result, grad


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
def bray_curtis_grad(x, y):
    numerator = 0.0
    denominator = 0.0
    for i in range(x.shape[0]):
        numerator += np.abs(x[i] - y[i])
        denominator += np.abs(x[i] + y[i])

    if denominator > 0.0:
        dist = float(numerator) / denominator
        grad = (np.sign(x - y) - dist) / denominator
    else:
        dist = 0.0
        grad = np.zeros(x.shape)

    return dist, grad


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
def haversine_grad(x, y):
    # spectral initialization puts many points near the poles
    # currently, adding pi/2 to the latitude avoids problems
    # TODO: reimplement with quaternions to avoid singularity

    if x.shape[0] != 2:
        raise ValueError("haversine is only defined for 2 dimensional data")
    sin_lat = np.sin(0.5 * (x[0] - y[0]))
    cos_lat = np.cos(0.5 * (x[0] - y[0]))
    sin_long = np.sin(0.5 * (x[1] - y[1]))
    cos_long = np.cos(0.5 * (x[1] - y[1]))

    a_0 = np.cos(x[0] + np.pi / 2) * np.cos(y[0] + np.pi / 2) * sin_long ** 2
    a_1 = a_0 + sin_lat ** 2

    d = 2.0 * np.arcsin(np.sqrt(min(max(abs(a_1), 0), 1)))
    denom = np.sqrt(abs(a_1 - 1)) * np.sqrt(abs(a_1))
    grad = np.array(
        [
            (
                sin_lat * cos_lat
                - np.sin(x[0] + np.pi / 2) * np.cos(y[0] + np.pi / 2) * sin_long ** 2
            ),
            (np.cos(x[0] + np.pi / 2) * np.cos(y[0] + np.pi / 2) * sin_long * cos_long),
        ]
    ) / (denom + 1e-6)
    return d, grad


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


@numba.njit(fastmath=True)
def cosine_grad(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 and norm_y == 0.0:
        dist = 0.0
        grad = np.zeros(x.shape)
    elif norm_x == 0.0 or norm_y == 0.0:
        dist = 1.0
        grad = np.zeros(x.shape)
    else:
        grad = -(x * result - y * norm_x) / np.sqrt(norm_x ** 3 * norm_y)
        dist = 1.0 - (result / np.sqrt(norm_x * norm_y))

    return dist, grad


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
def hellinger(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0

    for i in range(x.shape[0]):
        result += np.sqrt(x[i] * y[i])
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0 and l1_norm_y == 0:
        return 0.0
    elif l1_norm_x == 0 or l1_norm_y == 0:
        return 1.0
    else:
        return np.sqrt(1 - result / np.sqrt(l1_norm_x * l1_norm_y))


@numba.njit()
def hellinger_grad(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0

    grad_term = np.empty(x.shape[0])

    for i in range(x.shape[0]):
        grad_term[i] = np.sqrt(x[i] * y[i])
        result += grad_term[i]
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0 and l1_norm_y == 0:
        dist = 0.0
        grad = np.zeros(x.shape)
    elif l1_norm_x == 0 or l1_norm_y == 0:
        dist = 1.0
        grad = np.zeros(x.shape)
    else:
        dist_denom = np.sqrt(l1_norm_x * l1_norm_y)
        dist = np.sqrt(1 - result / dist_denom)
        grad_denom = 2 * dist
        grad_numer_const = (l1_norm_y * result) / (2 * dist_denom ** 3)

        grad = (grad_numer_const - (y / grad_term * dist_denom)) / grad_denom

    return dist, grad


@numba.njit()
def approx_log_Gamma(x):
    if x == 1:
        return 0
    # x2= 1/(x*x);
    return x * np.log(x) - x + 0.5 * np.log(2.0 * np.pi / x) + 1.0 / (x * 12.0)
    # + x2*(-1.0/360.0) + x2* (1.0/1260.0 + x2*(-1.0/(1680.0)  +\
    #  x2*(1.0/1188.0 + x2*(-691.0/360360.0 + x2*(1.0/156.0 +\
    #  x2*(-3617.0/122400.0 + x2*(43687.0/244188.0 + x2*(-174611.0/125400.0) +\
    #  x2*(77683.0/5796.0 + x2*(-236364091.0/1506960.0 + x2*(657931.0/300.0))))))))))))


@numba.njit()
def log_beta(x, y):
    a = min(x, y)
    b = max(x, y)
    if b < 5:
        value = -np.log(b)
        for i in range(1, int(a)):
            value += np.log(i) - np.log(b + i)
        return value
    else:
        return approx_log_Gamma(x) + approx_log_Gamma(y) - approx_log_Gamma(x + y)


@numba.njit()
def log_single_beta(x):
    return np.log(2.0) * (-2.0 * x + 0.5) + 0.5 * np.log(2.0 * np.pi / x) + 0.125 / x


# + x2*(-1.0/192.0 + x2* (1.0/640.0 + x2*(-17.0/(14336.0) +\
#  x2*(31.0/18432.0 + x2*(-691.0/180224.0 +\
#  x2*(5461.0/425984.0 + x2*(-929569.0/15728640.0 +\
#  x2*(3189151.0/8912896.0 + x2*(-221930581.0/79691776.0) +\
#  x2*(4722116521.0/176160768.0 + x2*(-968383680827.0/3087007744.0 +\
#  x2*(14717667114151.0/3355443200.0 ))))))))))))


@numba.njit()
def ll_dirichlet(data1, data2):
    """The symmetric relative log likelihood of rolling data2 vs data1
    in n trials on a die that rolled data1 in sum(data1) trials.

    ..math::
        D(data1, data2) = DirichletMultinomail(data2 | data1)
    """

    n1 = np.sum(data1)
    n2 = np.sum(data2)

    log_b = 0.0
    self_denom1 = 0.0
    self_denom2 = 0.0

    for i in range(data1.shape[0]):
        if data1[i] * data2[i] > 0.9:
            log_b += log_beta(data1[i], data2[i])
            self_denom1 += log_single_beta(data1[i])
            self_denom2 += log_single_beta(data2[i])

        else:
            if data1[i] > 0.9:
                self_denom1 += log_single_beta(data1[i])

            if data2[i] > 0.9:
                self_denom2 += log_single_beta(data2[i])

    return np.sqrt(
        1.0 / n2 * (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2)))
        + 1.0 / n1 * (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1)))
    )


@numba.njit(fastmath=True)
def symmetric_kl(x, y, z=1e-11):  # pragma: no cover
    """
    symmetrized KL divergence between two probability distributions

    ..math::
        D(x, y) = \frac{D_{KL}\left(x \Vert y\right) + D_{KL}\left(y \Vert x\right)}{2}
    """
    n = x.shape[0]
    x_sum = 0.0
    y_sum = 0.0
    kl1 = 0.0
    kl2 = 0.0

    for i in range(n):
        x[i] += z
        x_sum += x[i]
        y[i] += z
        y_sum += y[i]

    for i in range(n):
        x[i] /= x_sum
        y[i] /= y_sum

    for i in range(n):
        kl1 += x[i] * np.log(x[i] / y[i])
        kl2 += y[i] * np.log(y[i] / x[i])

    return (kl1 + kl2) / 2


@numba.njit(fastmath=True)
def symmetric_kl_grad(x, y, z=1e-11):  # pragma: no cover
    """
    symmetrized KL divergence and its gradient

    """
    n = x.shape[0]
    x_sum = 0.0
    y_sum = 0.0
    kl1 = 0.0
    kl2 = 0.0

    for i in range(n):
        x[i] += z
        x_sum += x[i]
        y[i] += z
        y_sum += y[i]

    for i in range(n):
        x[i] /= x_sum
        y[i] /= y_sum

    for i in range(n):
        kl1 += x[i] * np.log(x[i] / y[i])
        kl2 += y[i] * np.log(y[i] / x[i])

    dist = (kl1 + kl2) / 2
    grad = (np.log(y / x) - (x / y) + 1) / 2

    return dist, grad


@numba.njit()
def correlation_grad(x, y):
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
        dist = 0.0
        grad = np.zeros(x.shape)
    elif dot_product == 0.0:
        dist = 1.0
        grad = np.zeros(x.shape)
    else:
        dist = 1.0 - (dot_product / np.sqrt(norm_x * norm_y))
        grad = ((x - mu_x) / norm_x - (y - mu_y) / dot_product) * dist

    return dist, grad


@numba.njit(fastmath=True)
def sinkhorn_distance(
    x, y, M=_mock_identity, cost=_mock_cost, maxiter=64
):  # pragma: no cover
    p = (x / x.sum()).astype(np.float32)
    q = (y / y.sum()).astype(np.float32)

    u = np.ones(p.shape, dtype=np.float32)
    v = np.ones(q.shape, dtype=np.float32)

    for n in range(maxiter):
        t = M @ v
        u[t > 0] = p[t > 0] / t[t > 0]
        t = M.T @ u
        v[t > 0] = q[t > 0] / t[t > 0]

    pi = np.diag(v) @ M @ np.diag(u)
    result = 0.0
    for i in range(pi.shape[0]):
        for j in range(pi.shape[1]):
            if pi[i, j] > 0:
                result += pi[i, j] * cost[i, j]

    return result


@numba.njit(fastmath=True)
def spherical_gaussian_energy_grad(x, y):  # pragma: no cover
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    sigma = np.abs(x[2]) + np.abs(y[2])
    sign_sigma = np.sign(x[2])

    dist = (mu_1 ** 2 + mu_2 ** 2) / (2 * sigma) + np.log(sigma) + np.log(2 * np.pi)
    grad = np.empty(3, np.float32)

    grad[0] = mu_1 / sigma
    grad[1] = mu_2 / sigma
    grad[2] = sign_sigma * (1.0 / sigma - (mu_1 ** 2 + mu_2 ** 2) / (2 * sigma ** 2))

    return dist, grad


@numba.njit(fastmath=True)
def diagonal_gaussian_energy_grad(x, y):  # pragma: no cover
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    sigma_11 = np.abs(x[2]) + np.abs(y[2])
    sigma_12 = 0.0
    sigma_22 = np.abs(x[3]) + np.abs(y[3])

    det = sigma_11 * sigma_22
    sign_s1 = np.sign(x[2])
    sign_s2 = np.sign(x[3])

    if det == 0.0:
        # TODO: figure out the right thing to do here
        return mu_1 ** 2 + mu_2 ** 2, np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    cross_term = 2 * sigma_12
    m_dist = (
        np.abs(sigma_22) * (mu_1 ** 2)
        - cross_term * mu_1 * mu_2
        + np.abs(sigma_11) * (mu_2 ** 2)
    )

    dist = (m_dist / det + np.log(np.abs(det))) / 2.0 + np.log(2 * np.pi)
    grad = np.empty(6, dtype=np.float32)

    grad[0] = (2 * sigma_22 * mu_1 - cross_term * mu_2) / (2 * det)
    grad[1] = (2 * sigma_11 * mu_2 - cross_term * mu_1) / (2 * det)
    grad[2] = sign_s1 * (sigma_22 * (det - m_dist) + det * mu_2 ** 2) / (2 * det ** 2)
    grad[3] = sign_s2 * (sigma_11 * (det - m_dist) + det * mu_1 ** 2) / (2 * det ** 2)

    return dist, grad


@numba.njit(fastmath=True)
def gaussian_energy_grad(x, y):  # pragma: no cover
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    # Ensure width are positive
    x[2] = np.abs(x[2])
    y[2] = np.abs(y[2])

    # Ensure heights are positive
    x[3] = np.abs(x[3])
    y[3] = np.abs(y[3])

    # Ensure angle is in range -pi,pi
    x[4] = np.arcsin(np.sin(x[4]))
    y[4] = np.arcsin(np.sin(y[4]))

    # Covariance entries for y
    a = y[2] * np.cos(y[4]) ** 2 + y[3] * np.sin(y[4]) ** 2
    b = (y[2] - y[3]) * np.sin(y[4]) * np.cos(y[4])
    c = y[3] * np.cos(y[4]) ** 2 + y[2] * np.sin(y[4]) ** 2

    # Sum of covariance matrices
    sigma_11 = x[2] * np.cos(x[4]) ** 2 + x[3] * np.sin(x[4]) ** 2 + a
    sigma_12 = (x[2] - x[3]) * np.sin(x[4]) * np.cos(x[4]) + b
    sigma_22 = x[2] * np.sin(x[4]) ** 2 + x[3] * np.cos(x[4]) ** 2 + c

    # Determinant of the sum of covariances
    det_sigma = np.abs(sigma_11 * sigma_22 - sigma_12 ** 2)
    x_inv_sigma_y_numerator = (
        sigma_22 * mu_1 ** 2 - 2 * sigma_12 * mu_1 * mu_2 + sigma_11 * mu_2 ** 2
    )

    if det_sigma < 1e-32:
        return (
            mu_1 ** 2 + mu_2 ** 2,
            np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        )

    dist = x_inv_sigma_y_numerator / det_sigma + np.log(det_sigma) + np.log(2 * np.pi)

    grad = np.zeros(5, np.float32)
    grad[0] = (2 * sigma_22 * mu_1 - 2 * sigma_12 * mu_2) / det_sigma
    grad[1] = (2 * sigma_11 * mu_2 - 2 * sigma_12 * mu_1) / det_sigma

    grad[2] = mu_2 * (mu_2 * np.cos(x[4]) ** 2 - mu_1 * np.cos(x[4]) * np.sin(x[4]))
    grad[2] += mu_1 * (mu_1 * np.sin(x[4]) ** 2 - mu_2 * np.cos(x[4]) * np.sin(x[4]))
    grad[2] *= det_sigma
    grad[2] -= x_inv_sigma_y_numerator * np.cos(x[4]) ** 2 * sigma_22
    grad[2] -= x_inv_sigma_y_numerator * np.sin(x[4]) ** 2 * sigma_11
    grad[2] += x_inv_sigma_y_numerator * 2 * sigma_12 * np.sin(x[4]) * np.cos(x[4])
    grad[2] /= det_sigma ** 2 + 1e-8

    grad[3] = mu_1 * (mu_1 * np.cos(x[4]) ** 2 - mu_2 * np.cos(x[4]) * np.sin(x[4]))
    grad[3] += mu_2 * (mu_2 * np.sin(x[4]) ** 2 - mu_1 * np.cos(x[4]) * np.sin(x[4]))
    grad[3] *= det_sigma
    grad[3] -= x_inv_sigma_y_numerator * np.sin(x[4]) ** 2 * sigma_22
    grad[3] -= x_inv_sigma_y_numerator * np.cos(x[4]) ** 2 * sigma_11
    grad[3] -= x_inv_sigma_y_numerator * 2 * sigma_12 * np.sin(x[4]) * np.cos(x[4])
    grad[3] /= det_sigma ** 2 + 1e-8

    grad[4] = (x[3] - x[2]) * (
        2 * mu_1 * mu_2 * np.cos(2 * x[4]) - (mu_1 ** 2 - mu_2 ** 2) * np.sin(2 * x[4])
    )
    grad[4] *= det_sigma
    grad[4] -= x_inv_sigma_y_numerator * (x[3] - x[2]) * np.sin(2 * x[4]) * sigma_22
    grad[4] -= x_inv_sigma_y_numerator * (x[2] - x[3]) * np.sin(2 * x[4]) * sigma_11
    grad[4] -= x_inv_sigma_y_numerator * 2 * sigma_12 * (x[2] - x[3]) * np.cos(2 * x[4])
    grad[4] /= det_sigma ** 2 + 1e-8

    return dist, grad


@numba.njit(fastmath=True)
def spherical_gaussian_grad(x, y):  # pragma: no cover
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    sigma = x[2] + y[2]
    sigma_sign = np.sign(sigma)

    if sigma == 0:
        return 10.0, np.array([0.0, 0.0, -1.0], dtype=np.float32)

    dist = (
        (mu_1 ** 2 + mu_2 ** 2) / np.abs(sigma)
        + 2 * np.log(np.abs(sigma))
        + np.log(2 * np.pi)
    )
    grad = np.empty(3, dtype=np.float32)

    grad[0] = (2 * mu_1) / np.abs(sigma)
    grad[1] = (2 * mu_2) / np.abs(sigma)
    grad[2] = sigma_sign * (
        -(mu_1 ** 2 + mu_2 ** 2) / (sigma ** 2) + (2 / np.abs(sigma))
    )

    return dist, grad


# Special discrete distances -- where x and y are objects, not vectors


def get_discrete_params(data, metric):
    if metric == "ordinal":
        return {"support_size": float(data.max() - data.min()) / 2.0}
    elif metric == "count":
        min_count = scipy.stats.tmin(data)
        max_count = scipy.stats.tmax(data)
        lambda_ = scipy.stats.tmean(data)
        normalisation = count_distance(min_count, max_count, poisson_lambda=lambda_)
        return {
            "poisson_lambda": lambda_,
            "normalisation": normalisation / 2.0,  # heuristic
        }
    elif metric == "string":
        lengths = np.array([len(x) for x in data])
        max_length = scipy.stats.tmax(lengths)
        max_dist = max_length / 1.5  # heuristic
        normalisation = max_dist / 2.0  # heuristic
        return {"normalisation": normalisation, "max_dist": max_dist / 2.0}  # heuristic

    else:
        return {}


@numba.jit()
def categorical_distance(x, y):
    if x == y:
        return 0.0
    else:
        return 1.0


@numba.jit()
def hierarchical_categorical_distance(x, y, cat_hierarchy=[{}]):
    n_levels = float(len(cat_hierarchy))
    for level, cats in enumerate(cat_hierarchy):
        if cats[x] == cats[y]:
            return float(level) / n_levels
    else:
        return 1.0


@numba.njit()
def ordinal_distance(x, y, support_size=1.0):
    return abs(x - y) / support_size


@numba.jit()
def count_distance(x, y, poisson_lambda=1.0, normalisation=1.0):
    lo = int(min(x, y))
    hi = int(max(x, y))

    log_lambda = np.log(poisson_lambda)

    if lo < 2:
        log_k_factorial = 0.0
    elif lo < 10:
        log_k_factorial = 0.0
        for k in range(2, lo):
            log_k_factorial += np.log(k)
    else:
        log_k_factorial = approx_log_Gamma(lo + 1)

    result = 0.0

    for k in range(lo, hi):
        result += k * log_lambda - poisson_lambda - log_k_factorial
        log_k_factorial += np.log(k)

    return result / normalisation


@numba.njit()
def levenshtein(x, y, normalisation=1.0, max_distance=20):
    x_len, y_len = len(x), len(y)

    # Opt out of some comparisons
    if abs(x_len - y_len) > max_distance:
        return abs(x_len - y_len) / normalisation

    v0 = np.arange(y_len + 1).astype(np.float64)
    v1 = np.zeros(y_len + 1)

    for i in range(x_len):

        v1[i] = i + 1

        for j in range(y_len):
            deletion_cost = v0[j + 1] + 1
            insertion_cost = v1[j] + 1
            substitution_cost = int(x[i] == y[j])

            v1[j + 1] = min(deletion_cost, insertion_cost, substitution_cost)

        v0 = v1

        # Abort early if we've already exceeded max_dist
        if np.min(v0) > max_distance:
            return max_distance / normalisation

    return v0[y_len] / normalisation


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
    "poincare": poincare,
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
    "hellinger": hellinger,
    "haversine": haversine,
    "braycurtis": bray_curtis,
    "ll_dirichlet": ll_dirichlet,
    "symmetric_kl": symmetric_kl,
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
    # Special discrete distances
    "categorical": categorical_distance,
    "ordinal": ordinal_distance,
    "hierarchical_categorical": hierarchical_categorical_distance,
    "count": count_distance,
    "string": levenshtein,
}

named_distances_with_gradients = {
    # general minkowski distances
    "euclidean": euclidean_grad,
    "l2": euclidean_grad,
    "manhattan": manhattan_grad,
    "taxicab": manhattan_grad,
    "l1": manhattan_grad,
    "chebyshev": chebyshev_grad,
    "linfinity": chebyshev_grad,
    "linfty": chebyshev_grad,
    "linf": chebyshev_grad,
    "minkowski": minkowski_grad,
    # Standardised/weighted distances
    "seuclidean": standardised_euclidean_grad,
    "standardised_euclidean": standardised_euclidean_grad,
    "wminkowski": weighted_minkowski_grad,
    "weighted_minkowski": weighted_minkowski_grad,
    "mahalanobis": mahalanobis_grad,
    # Other distances
    "canberra": canberra_grad,
    "cosine": cosine_grad,
    "correlation": correlation_grad,
    "hellinger": hellinger_grad,
    "haversine": haversine_grad,
    "braycurtis": bray_curtis_grad,
    "symmetric_kl": symmetric_kl_grad,
    # Special embeddings
    "spherical_gaussian_energy": spherical_gaussian_energy_grad,
    "diagonal_gaussian_energy": diagonal_gaussian_energy_grad,
    "gaussian_energy": gaussian_energy_grad,
    "hyperboloid": hyperboloid_grad,
}

DISCRETE_METRICS = (
    "categorical",
    "hierarchical_categorical",
    "ordinal",
    "count",
    "string",
)

SPECIAL_METRICS = (
    "hellinger",
    "ll_dirichlet",
    "symmetric_kl",
    "poincare",
    hellinger,
    ll_dirichlet,
    symmetric_kl,
    poincare,
)


@numba.njit(parallel=True)
def parallel_special_metric(X, Y=None, metric=hellinger):
    if Y is None:
        result = np.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                result[i, j] = metric(X[i], X[j])
                result[j, i] = result[i, j]
    else:
        result = np.zeros((X.shape[0], Y.shape[0]))

        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                result[i, j] = metric(X[i], Y[j])

    return result


def pairwise_special_metric(X, Y=None, metric="hellinger", kwds=None):
    if callable(metric):
        if kwds is not None:
            kwd_vals = tuple(kwds.values())
        else:
            kwd_vals = ()

        @numba.njit(fastmath=True)
        def _partial_metric(_X, _Y=None):
            return metric(_X, _Y, *kwd_vals)

        return pairwise_distances(X, Y, metric=_partial_metric)
    else:
        special_metric_func = named_distances[metric]
    return parallel_special_metric(X, Y, metric=special_metric_func)
