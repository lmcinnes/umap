import pytest
import numba
import numpy as np
from numpy.testing import assert_array_equal
from umap import distances as dist

# --------
# Fixtures
# --------


@pytest.fixture(scope="function")
def stashed_chunked_implementation():
    @numba.njit(parallel=True, nogil=True)
    def stashed_chunked_parallel_special_metric(
        X, Y=None, metric=dist.named_distances["hellinger"], chunk_size=16
    ):
        if Y is None:
            row_size = col_size = X.shape[0]
        else:
            row_size = X.shape[0]
            col_size = Y.shape[0]
        result = np.zeros((row_size, col_size), dtype=np.float32)
        if Y is None:
            size = X.shape[0]
            n_row_chunks = (size // chunk_size) + 1
            for chunk_idx in numba.prange(n_row_chunks):
                n = chunk_idx * chunk_size
                chunk_end_n = min(n + chunk_size, size)
                for m in range(n, size, chunk_size):
                    chunk_end_m = min(m + chunk_size, size)
                    if n == m:
                        for i in range(n, chunk_end_n):
                            for j in range(m, chunk_end_m):
                                if j > i:
                                    d = metric(X[i], X[j])
                                    result[i, j] = d
                                    result[j, i] = d
                    else:
                        for i in range(n, chunk_end_n):
                            for j in range(m, chunk_end_m):
                                d = metric(X[i], X[j])
                                result[i, j] = d
                                result[j, i] = d
        else:
            n_row_chunks = (row_size // chunk_size) + 1
            for chunk_idx in numba.prange(n_row_chunks):
                n = chunk_idx * chunk_size
                chunk_end_n = min(n + chunk_size, row_size)
                for m in range(0, col_size, chunk_size):
                    chunk_end_m = min(m + chunk_size, col_size)
                    for i in range(n, chunk_end_n):
                        for j in range(m, chunk_end_m):
                            d = metric(X[i], Y[j])
                            result[i, j] = d

        return result

    return stashed_chunked_parallel_special_metric


@pytest.fixture(scope="function")
def chunked_parallel_full_iterations():
    @numba.njit(parallel=True, nogil=True)
    def chunked_parallel_special_metric(
        X, Y=None, metric=dist.named_distances["hellinger"], chunk_size=16
    ):
        if Y is None:
            XX, symmetrical = X, True
            row_size = col_size = X.shape[0]
        else:
            XX, symmetrical = Y, False
            row_size, col_size = X.shape[0], Y.shape[0]

        result = np.zeros((row_size, col_size), dtype=np.float32)
        n_row_chunks = (row_size // chunk_size) + 1
        for chunk_idx in numba.prange(n_row_chunks):
            n = chunk_idx * chunk_size
            chunk_end_n = min(n + chunk_size, row_size)
            m_start = n if symmetrical else 0
            for m in range(m_start, col_size, chunk_size):
                chunk_end_m = min(m + chunk_size, col_size)
                for i in range(n, chunk_end_n):
                    for j in range(m, chunk_end_m):
                        result[i, j] = metric(X[i], XX[j])
        return result

    return chunked_parallel_special_metric


@pytest.fixture(scope="function")
def workaround_590_impl():
    @numba.njit(parallel=True, nogil=True)
    def chunked_parallel_special_metric(
        X, Y=None, metric=dist.named_distances["hellinger"], chunk_size=16
    ):
        if Y is None:
            size = X.shape[0]
            result = np.zeros((size, size), dtype=np.float32)
            n_row_chunks = (size // chunk_size) + 1
            for chunk_idx in numba.prange(n_row_chunks):
                n = chunk_idx * chunk_size
                chunk_end_n = min(n + chunk_size, size)
                for m in range(n, size, chunk_size):
                    chunk_end_m = min(m + chunk_size, size)
                    if n == m:
                        for i in range(n, chunk_end_n):
                            for j in range(m, chunk_end_m):
                                if j > i:
                                    d = metric(X[i], X[j])
                                    result[i, j] = d
                                    result[j, i] = d
                    else:
                        for i in range(n, chunk_end_n):
                            for j in range(m, chunk_end_m):
                                d = metric(X[i], X[j])
                                result[i, j] = d
                                result[j, i] = d
            return result

        row_size = X.shape[0]
        col_size = Y.shape[0]
        result = np.zeros((row_size, col_size), dtype=np.float32)
        n_row_chunks = (row_size // chunk_size) + 1
        for chunk_idx in numba.prange(n_row_chunks):
            n = chunk_idx * chunk_size
            chunk_end_n = min(n + chunk_size, row_size)
            for m in range(0, col_size, chunk_size):
                chunk_end_m = min(m + chunk_size, col_size)
                for i in range(n, chunk_end_n):
                    for j in range(m, chunk_end_m):
                        d = metric(X[i], Y[j])
                        result[i, j] = d

        return result

    return chunked_parallel_special_metric


@pytest.fixture(scope="function")
def benchmark_data(request):
    shape = request.param
    spatial_data = np.random.randn(*shape).astype(np.float32)
    return np.abs(spatial_data)


# ---------------------------------------------------------------

# Uncomment this to skip the tests
# @pytest.mark.skip(reason="Focus on benchmark for now. This passes!")
def test_chunked_parallel_alternative_implementations(
    spatial_data, workaround_590_impl, chunked_parallel_full_iterations
):
    # Base tests that must pass!
    dist_matrix_x = workaround_590_impl(np.abs(spatial_data[:-2]))
    dist_matrix_xy = workaround_590_impl(
        np.abs(spatial_data[:-2]), np.abs(spatial_data[:-2])
    )

    dist_matrix_x_full = chunked_parallel_full_iterations(np.abs(spatial_data[:-2]))
    dist_matrix_xy_full = chunked_parallel_full_iterations(
        np.abs(spatial_data[:-2]), np.abs(spatial_data[:-2])
    )

    assert_array_equal(
        dist_matrix_x_full,
        dist_matrix_x,
        err_msg="Distances don't match for metric hellinger",
    )

    assert_array_equal(
        dist_matrix_xy_full,
        dist_matrix_xy,
        err_msg="Distances don't match for metric hellinger",
    )


# Uncomment this to skip the tests
# @pytest.mark.skip(reason="Focus on benchmark for now. This passes!")
def test_chunked_parallel_special_metric_implementation_hellinger(
    spatial_data, stashed_chunked_implementation, chunked_parallel_full_iterations
):

    # Base tests that must pass!
    dist_matrix_x = chunked_parallel_full_iterations(np.abs(spatial_data[:-2]))
    dist_matrix_xy = chunked_parallel_full_iterations(
        np.abs(spatial_data[:-2]), np.abs(spatial_data[:-2])
    )
    test_matrix = np.array(
        [
            [
                dist.hellinger_grad(np.abs(spatial_data[i]), np.abs(spatial_data[j]))[0]
                for j in range(spatial_data.shape[0] - 2)
            ]
            for i in range(spatial_data.shape[0] - 2)
        ]
    ).astype(np.float32)

    assert_array_equal(
        test_matrix,
        dist_matrix_x,
        err_msg="Distances don't match for metric hellinger",
    )

    assert_array_equal(
        test_matrix,
        dist_matrix_xy,
        err_msg="Distances don't match for metric hellinger",
    )

    # Test to compare chunked_parallel different implementations
    dist_x_stashed = stashed_chunked_implementation(np.abs(spatial_data[:-2]))
    dist_xy_stashed = stashed_chunked_implementation(
        np.abs(spatial_data[:-2]), np.abs(spatial_data[:-2])
    )

    assert_array_equal(
        dist_xy_stashed,
        dist_matrix_xy,
        err_msg="Distances don't match between stashed and current chunked_parallel implementations with X and Y!",
    )

    assert_array_equal(
        dist_x_stashed,
        dist_matrix_x,
        err_msg="Distances don't match between stashed and current chunked_parallel implementations with X only!",
    )

    # test hellinger on different X and Y Pair
    spatial_data_two = np.random.randn(10, 20)
    dist_stashed_diff_pair = stashed_chunked_implementation(
        np.abs(spatial_data[:-2]), spatial_data_two
    )
    dist_chunked_diff_pair = dist.chunked_parallel_special_metric(
        np.abs(spatial_data[:-2]), spatial_data_two
    )

    assert_array_equal(
        dist_stashed_diff_pair,
        dist_chunked_diff_pair,
        err_msg="Distances don't match between stashed and current chunked_parallel implementations",
    )


# ----------------------------
# 1st Group Benchmark: X only
# (Worst Case)
# ----------------------------


@pytest.mark.benchmark(
    group="benchmark_single_param",
)
@pytest.mark.parametrize(
    "benchmark_data",
    [(10 * s, 10 * s) for s in list(range(0, 101, 10))[1:]],
    indirect=["benchmark_data"],
)
def test_benchmark_full_iterations_x_only(
    benchmark,
    benchmark_data,
    chunked_parallel_full_iterations,
):

    # single argument
    benchmark.pedantic(
        chunked_parallel_full_iterations,
        kwargs={"X": benchmark_data, "Y": None},
        warmup_rounds=5,
        iterations=10,
        rounds=10,
    )


@pytest.mark.benchmark(
    group="benchmark_single_param",
)
@pytest.mark.parametrize(
    "benchmark_data",
    [(10 * s, 10 * s) for s in list(range(0, 101, 10))[1:]],
    indirect=["benchmark_data"],
)
def test_benchmark_workaround_590_x_only(
    benchmark,
    benchmark_data,
    workaround_590_impl,
):

    # single argument
    benchmark.pedantic(
        workaround_590_impl,
        kwargs={"X": benchmark_data, "Y": None},
        warmup_rounds=5,
        iterations=10,
        rounds=10,
    )


# ----------------------------
# 2nd Group Benchmark: X and Y
# ----------------------------


@pytest.mark.benchmark(
    group="benchmark_X_Y_params",
)
@pytest.mark.parametrize(
    "benchmark_data",
    [(10 * s, 10 * s) for s in list(range(0, 101, 10))[1:]],
    indirect=["benchmark_data"],
)
def test_benchmark_full_iterations_x_y(
    benchmark,
    benchmark_data,
    chunked_parallel_full_iterations,
):

    # single argument
    benchmark.pedantic(
        chunked_parallel_full_iterations,
        kwargs={"X": benchmark_data, "Y": benchmark_data},
        warmup_rounds=5,
        iterations=10,
        rounds=10,
    )


@pytest.mark.benchmark(
    group="benchmark_X_Y_params",
)
@pytest.mark.parametrize(
    "benchmark_data",
    [(10 * s, 10 * s) for s in list(range(0, 101, 10))[1:]],
    indirect=["benchmark_data"],
)
def test_benchmark_workaround_590_x_y(
    benchmark,
    benchmark_data,
    workaround_590_impl,
):

    # single argument
    benchmark.pedantic(
        workaround_590_impl,
        kwargs={"X": benchmark_data, "Y": benchmark_data},
        warmup_rounds=5,
        iterations=10,
        rounds=10,
    )
