import numpy as np
import pytest

import umap.distances as dist


def numerical_gradient(f, x, eps=1e-6, forward_only=False):
    """
    Finite-difference gradient of scalar function f at x.

    Parameters
    ----------
    f : callable
        Scalar function f(x).
    x : ndarray
        Point at which to evaluate the gradient.
    eps : float
        Finite-difference step size.
    forward_only : bool, default=False
        If True, use forward differences only:
            (f(x + eps) - f(x)) / eps
        Otherwise, use central differences.
    """
    grad = np.zeros_like(x, dtype=np.float32)

    fx = f(x) if forward_only else None

    for i in range(x.size):
        x_fwd = x.copy()
        x_fwd[i] += eps

        if forward_only:
            grad[i] = (f(x_fwd) - fx) / eps
        else:
            x_bwd = x.copy()
            x_bwd[i] -= eps
            grad[i] = (f(x_fwd) - f(x_bwd)) / (2.0 * eps)

    return grad


def numerical_grad_x(dist, x, y, eps=1e-6, dist_kwargs=None, forward_only=False):
    """
    Numerical gradient of dist(x, y) with respect to x only.
    """
    return numerical_gradient(lambda z: dist(z, y, **dist_kwargs), x, eps, forward_only)


def sample_normal_pairs(n, d, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    x = rng.normal(size=(n, d))
    y = rng.normal(size=(n, d))
    return x, y


def sample_dirichlet_pairs(n, d, alpha=1.0, rng=None):
    # For hellinger
    if rng is None:
        rng = np.random.default_rng()
    x = rng.dirichlet(alpha=np.full(d, alpha), size=n)
    y = rng.dirichlet(alpha=np.full(d, alpha), size=n)
    return x, y


def sample_abundance_pairs(n, d, shape=2.0, scale=1.0, rng=None):
    # For bray curtis
    if rng is None:
        rng = np.random.default_rng()
    x = rng.gamma(shape=shape, scale=scale, size=(n, d))
    y = rng.gamma(shape=shape, scale=scale, size=(n, d))
    return x, y


def assert_gradient_matches_finite_diff(
    dist,
    grad,
    dist_kwargs=None,
    sampler=sample_normal_pairs,
    dim=8,
    n_samples=1_00,
    forward_only=False,
    skip_close_coords=False,
    max_tol=1e-5,
    mean_tol=1e-6,
):

    rng = np.random.default_rng(0)
    x, y = sampler(n_samples, dim, rng=rng)

    if dist_kwargs is None:
        dist_kwargs = dict()

    numeric_grad = np.vstack(
        [
            numerical_grad_x(
                dist, x[i], y[i], dist_kwargs=dist_kwargs, forward_only=forward_only
            )
            for i in range(len(x))
        ]
    )

    analytic_dist, analytic_grad = zip(
        *[grad(x[i], y[i], **dist_kwargs) for i in range(len(x))]
    )

    analytic_dist = np.hstack(analytic_dist)
    analytic_grad = np.vstack(analytic_grad)

    ### Check Close to Finite Difference

    if skip_close_coords:
        # Skip coords near zero for distances like Hellinger
        close_coords = (np.abs(x) < 1e-3) | (np.abs(y) < 1e-3)
        numeric_grad[close_coords] = 0
        analytic_grad[close_coords] = 0

    coord_errors = np.abs(numeric_grad - analytic_grad)

    assert coord_errors.max() < max_tol, f"Max tol exceeded: {coord_errors.max()}"
    assert coord_errors.mean() < mean_tol, f"Mean tol exceeded: {coord_errors.mean()}"

    ### Check grad dists match with non-grad dists
    true_dist = np.array([dist(x[i], y[i], **dist_kwargs) for i in range(len(x))])
    assert np.max(np.abs(true_dist - analytic_dist)) < 1e-6, "Distance mismatch"


@pytest.mark.parametrize("dim", [4, 16, 64])
def test_euclidean_gradient(
    dim,
):
    assert_gradient_matches_finite_diff(
        dist.euclidean,
        dist.euclidean_grad,
        sampler=sample_normal_pairs,
        dim=dim,
    )


@pytest.mark.parametrize("dim", [4, 16, 64])
@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_minkowski_gradient(dim, p):
    assert_gradient_matches_finite_diff(
        dist.minkowski,
        dist.minkowski_grad,
        sampler=sample_normal_pairs,
        dim=dim,
        dist_kwargs={"p": p},
        forward_only=p == 1,
    )


@pytest.mark.parametrize("dim", [4, 16, 64])
@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_weighted_minkowski_gradient(dim, p):
    rng = np.random.default_rng(0)
    assert_gradient_matches_finite_diff(
        dist.weighted_minkowski,
        dist.weighted_minkowski_grad,
        sampler=sample_normal_pairs,
        dim=dim,
        dist_kwargs={"p": p, "w": rng.uniform(size=dim)},
        forward_only=p == 1,
    )


@pytest.mark.parametrize("dim", [4, 16, 64])
def test_cosine_gradient(
    dim,
):
    assert_gradient_matches_finite_diff(
        dist.cosine,
        dist.cosine_grad,
        sampler=sample_normal_pairs,
        dim=dim,
    )


@pytest.mark.parametrize("dim", [4, 16, 64])
def test_manhattan_gradient(
    dim,
):
    assert_gradient_matches_finite_diff(
        dist.manhattan,
        dist.manhattan_grad,
        sampler=sample_normal_pairs,
        dim=dim,
        forward_only=True,
        # skip_close_coords=True,
    )


@pytest.mark.parametrize("dim", [4, 16, 64])
def test_chebyshev_gradient(
    dim,
):
    assert_gradient_matches_finite_diff(
        dist.chebyshev,
        dist.chebyshev_grad,
        sampler=sample_normal_pairs,
        dim=dim,
    )


@pytest.mark.parametrize("dim", [4, 16, 64])
def test_correlation_gradient(
    dim,
):
    assert_gradient_matches_finite_diff(
        dist.correlation,
        dist.correlation_grad,
        sampler=sample_normal_pairs,
        dim=dim,
    )


@pytest.mark.parametrize("dim", [4, 16, 64])
def test_braycurtis_gradient(
    dim,
):
    assert_gradient_matches_finite_diff(
        dist.bray_curtis,
        dist.bray_curtis_grad,
        sampler=sample_abundance_pairs,
        dim=dim,
    )


@pytest.mark.parametrize("dim", [4, 16, 64])
def test_hellinger_gradient(dim):
    assert_gradient_matches_finite_diff(
        dist.hellinger,
        dist.hellinger_grad,
        sampler=sample_dirichlet_pairs,
        dim=dim,
        skip_close_coords=True,
    )


@pytest.mark.parametrize("dim", [4, 16, 64])
def test_standardised_euclidean_gradient(dim):
    rng = np.random.default_rng(0)
    sigma = rng.uniform(low=0.5, high=2.0, size=dim).astype(np.float64)
    assert_gradient_matches_finite_diff(
        dist.standardised_euclidean,
        dist.standardised_euclidean_grad,
        sampler=sample_normal_pairs,
        dim=dim,
        dist_kwargs={"sigma": sigma},
    )


@pytest.mark.parametrize("dim", [4, 16, 64])
def test_mahalanobis_gradient(dim):
    rng = np.random.default_rng(0)
    diag = rng.uniform(low=0.5, high=2.0, size=dim).astype(np.float64)
    vinv = np.diag(diag)
    # require float64 mahalanobis accuracy for finite difference method
    assert_gradient_matches_finite_diff(
        dist.mahalanobis_f64,
        dist.mahalanobis_grad,
        sampler=sample_normal_pairs,
        dim=dim,
        dist_kwargs={"vinv": vinv},
    )


# TODO
# canberra
# symmetric_kl
# haversine
# hyperboloid
# gaussian_energy
# spherical_gaussian_energy
# diagonal_gaussian_energy
