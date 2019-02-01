import numpy as np
import numba

###############################################################################
# Hungarian matching code translated from
# https://github.com/scipy/scipy/blob/master/scipy/optimize/_hungarian.py
###############################################################################
@numba.njit()
def _step1(cost, marked, row_uncovered, col_uncovered):
    """Steps 1 and 2 in the Wikipedia page."""

    # Step 1: For each row of the matrix, find the smallest element and
    # subtract it from every element in its row.
    # cost -= np.min(cost, axis=1)[:, np.newaxis]
    for i in range(cost.shape[0]):
        cost[i] -= cost[i].min()
    # Step 2: Find a zero (Z) in the resulting matrix. If there is no
    # starred zero in its row or column, star Z. Repeat for each element
    # in the matrix.
    for i, j in zip(*np.nonzero(cost == 0)):
        if col_uncovered[j] and row_uncovered[i]:
            marked[i, j] = 1
            col_uncovered[j] = False
            row_uncovered[i] = False

    row_uncovered[:] = True
    col_uncovered[:] = True
    return 3


@numba.njit()
def _step3(cost, marked, row_uncovered, col_uncovered):
    """
    Cover each column containing a starred zero. If n columns are covered,
    the starred zeros describe a complete set of unique assignments.
    In this case, Go to DONE, otherwise, Go to Step 4.
    """
    m = (marked == 1)
    for j in range(col_uncovered.shape[0]):
        for i in range(m.shape[0]):
            if m[i, j] > 0:
                col_uncovered[j] = False
                break

    if m.sum() < cost.shape[0]:
        return 4
    else:
        return 0


@numba.njit()
def _step4(cost, marked, row_uncovered, col_uncovered, z0):
    """
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    # We convert to int as numpy operations are faster on int
    C = (cost == 0)
    covered_C = C * 1.0
    for i in range(row_uncovered.shape[0]):
        if row_uncovered[i] == 0:
            covered_C[i, :] = 0

    for j in range(col_uncovered.shape[0]):
        if col_uncovered[j] == 0:
            covered_C[:, j] = 0

    n = C.shape[0]
    m = C.shape[1]

    while True:
        # Find an uncovered zero
        flat_max = np.argmax(covered_C)
        row = flat_max // m
        col = flat_max % m
        if covered_C[row, col] == 0:
            return 6
        else:
            marked[row, col] = 2
            # Find the first starred element in the row
            star_col = np.argmax(marked[row] == 1)
            if marked[row, star_col] != 1:
                # Could not find one
                z0[0] = row
                z0[1] = col
                return 5
            else:
                col = star_col
                row_uncovered[row] = False
                col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * row_uncovered
                covered_C[row] = 0


@numba.njit()
def _step5(cost, marked, row_uncovered, col_uncovered, z0, path):
    """
    Construct a series of alternating primed and starred zeros as follows.
    Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always be one).
    Continue until the series terminates at a primed zero that has no starred
    zero in its column. Unstar each starred zero of the series, star each
    primed zero of the series, erase all primes and uncover every line in the
    matrix. Return to Step 3
    """
    count = 0
    path[count, 0] = z0[0]
    path[count, 1] = z0[1]

    while True:
        # Find the first starred element in the col defined by
        # the path.
        row = np.argmax(marked[:, path[count, 1]] == 1)
        if marked[row, path[count, 1]] != 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = np.argmax(marked[path[count, 0]] == 2)
        if marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if marked[path[i, 0], path[i, 1]] == 1:
            marked[path[i, 0], path[i, 1]] = 0
        else:
            marked[path[i, 0], path[i, 1]] = 1

    row_uncovered[:] = True
    col_uncovered[:] = True
    # Erase all prime markings
    for i in range(marked.shape[0]):
        for j in range(marked.shape[1]):
            if marked[i, j] == 2:
                marked[i, j] = 0
    return 3


@numba.njit()
def _step6(cost, marked, row_uncovered, col_uncovered):
    """
    Add the value found in Step 4 to every element of each covered row,
    and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered lines.
    """
    # the smallest uncovered value in the matrix
    if np.any(row_uncovered) and np.any(col_uncovered):
        minval = np.inf
        for i in range(row_uncovered.shape[0]):
            if row_uncovered[i] > 0:
                for j in range(col_uncovered.shape[0]):
                    if col_uncovered[j] > 0 and cost[i, j] < minval:
                        minval = cost[i, j]

        for i in range(row_uncovered.shape[0]):
            if row_uncovered[i] == 0:
                cost[i, :] += minval
        for j in range(col_uncovered.shape[0]):
            if col_uncovered[j] > 0:
                cost[:, j] -= minval
    return 4


@numba.njit()
def hungarian(cost_matrix):
    cost = cost_matrix.copy()
    n = cost.shape[0]
    m = cost.shape[1]
    marked = np.zeros(cost.shape, dtype=np.int8)
    row_uncovered = np.ones(n, dtype=np.int8)
    col_uncovered = np.ones(m, dtype=np.int8)
    z0 = np.zeros(2, dtype=np.int32)
    path = np.zeros((n + m, 2), dtype=np.int32)

    state = _step1(cost, marked, row_uncovered, col_uncovered)

    while state > 0:
        if state == 3:
            state = _step3(cost, marked, row_uncovered, col_uncovered)
        elif state == 4:
            state = _step4(cost, marked, row_uncovered, col_uncovered, z0)
        elif state == 5:
            state = _step5(cost, marked, row_uncovered, col_uncovered, z0, path)
        elif state == 6:
            state = _step6(cost, marked, row_uncovered, col_uncovered)
        else:
            raise ValueError('Invalid state encountered in Hungarian matching algorithm')

    return np.nonzero(marked == 1)

##############################################################################
# Curvature code
##############################################################################

@numba.njit()
def wasserstein(knns1, knns2, data, metric):
    knns1_set = set(knns1)
    knns2_set = set(knns2)
    compare1 = knns1_set - knns2_set
    compare2 = knns2_set - knns1_set
    if len(compare1) == 0 or len(compare2) == 0:
        return 0.0
    n_neighbors = len(knns1)
    tmp_distances = np.zeros((len(compare1), len(compare2)))
    for i, idx1 in enumerate(compare1):
        for j, idx2 in enumerate(compare2):
            tmp_distances[i, j] = metric(data[idx1], data[idx2])

    row_matching, col_matching = hungarian(tmp_distances)
    result = 0.0
    for i, j in zip(row_matching, col_matching):
        result += tmp_distances[i, j]
    return result / n_neighbors


@numba.njit()
def sectional_curvature(index, knns, knndists, data, metric):
    neighbors = knns[index]
    dists = knndists[index]
    sc = 0.0
    for i, neigh in enumerate(neighbors):
        if dists[i] > 0:
            c = 1 - wasserstein(knns[index], knns[neigh], data, metric) / dists[i]
            sc += c
    return sc / neighbors.shape[0]


@numba.njit(parallel=True)
def compute_curvatures(data, knn_indices, knn_dists, metric):
    result = np.empty(data.shape[0], dtype=np.float32)
    for i in range(data.shape[0]):
        result[i] = sectional_curvature(i, knn_indices, knn_dists, data, metric)
    return result