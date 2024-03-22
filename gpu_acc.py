# https://docs.cupy.dev/en/stable/user_guide/kernel.html
import itertools as iter
import math
import time

import cupy as cp
import numpy as np
import scipy as sp
import scipy.linalg.blas as sblas
import sympy as sy
from sympy.plotting import plot


def triangle_row_starts():
    """Starting indices of a dense lower triangle matrix."""
    p = 0
    i = 0
    while True:
        i += p
        yield i
        p += 1


def acc_to_2d_numpy(n_rows, cell_size, acc):
    """Convert in-GPU accumulator representation to 2d numpy array.
       For diagnostics and GPU->host export.
       This may produce output that is easier to inspect than one of acc_to_4d_numpy."""
    out = np.zeros((n_rows * cell_size, n_rows * cell_size), dtype=DTYPE)
    row_len = 1
    idx = 0
    for i in range(n_rows):
        for j in range(row_len):
            for ci in range(cell_size):
                for cj in range(cell_size):
                    out[i * cell_size + ci, j * cell_size + cj] = acc[idx]
                    idx += 1
        row_len += 1
    return out


def acc_to_4d_numpy(n_rows, cell_size, acc):
    """Convert in-GPU accumulator representation to a logically laid out numpy array.
       For diagnostics and GPU->host export.
       first 2 indexes select variable pair, second 2 indexes select powers of sums."""
    out = np.zeros((n_rows, n_rows, cell_size, cell_size), dtype=DTYPE)
    row_len = 1
    idx = 0
    for i in range(n_rows):
        for j in range(row_len):
            for ci in range(cell_size):
                for cj in range(cell_size):
                    out[i, j, ci, cj] = acc[idx]
                    idx += 1
        row_len += 1
    return out


# Floats type to use. Update CUDA kernels when changing this.
DTYPE = cp.float64

print("CUDA status:", cp.cuda.Device(), cp.cuda.device.get_compute_capability(), cp.cuda.get_current_stream())

with open('gpu_acc.cu', 'r') as f:
    gpu_acc_source = f.read()
GPU_ACC_MODULE = cp.RawModule(code=gpu_acc_source)

CK_POWERS = GPU_ACC_MODULE.get_function('powers')
print("ck_powers:", CK_POWERS.attributes)

CK_PAIRS_UPDATE = GPU_ACC_MODULE.get_function('pairs_update')
print("ck_pairs_update:", CK_PAIRS_UPDATE.attributes)

CK_MOMENTS = GPU_ACC_MODULE.get_function('moments')


class MomentsGpu:
    def __init__(self, m: tuple[int, int], n_x: int, x_bias=None):
        # Required max moment's power degrees
        self.m = m
        # Number of scalar measurements in a sample X
        self.n_x = n_x
        # Estimated mean to subtract from samples.
        assert x_bias is None or x_bias.shape == (self.n_x,)
        if x_bias is not None:
            self.x_bias = x_bias
        else:
            self.x_bias = np.zeros((1, self.n_x), dtype=DTYPE)
        # Max power degree to calc for each variable. max(m) for off-diagonal, sum(m) on the diagonal
        # Using sum to get all the values necessary for later calculations, as max(m) <= sum(m)
        self.max_single_pow = sum(m)
        # Sums of a single measurement powers. It is re-used on subsequent updates.
        self.x_powers = cp.zeros((n_x, self.max_single_pow), dtype=DTYPE)
        # Accumulates power sums of each variable
        self.x_powers_acc = cp.zeros((n_x, self.max_single_pow), dtype=DTYPE)
        assert self.x_powers.size == self.n_x * self.max_single_pow
        assert self.x_powers.shape == self.x_powers_acc.shape

        self.pairs_acc_cell_size = max(m)
        # Side size of the pairs acc triangle matrix.
        # Excluding diagonal in acc size since we're accumulating single var powers separately in x_powers_acc.
        self.pairs_acc_size = n_x - 1
        # In theory, if we were only requiring one-way moments (e.g. E(XaXbXb) for only a <= b)
        #   then the matrix may be reduced to shape self.m
        self.cell_matrix_size = self.pairs_acc_cell_size ** 2
        row_starts = triangle_row_starts()
        row_index = np.array([next(row_starts) * self.cell_matrix_size
                              for _ in range(self.pairs_acc_size + 1)], dtype=np.uint32)
        print("> row_index=\n", row_index)

        pairs_acc_length = row_index[-1]
        assert pairs_acc_length == math.floor(
            (self.pairs_acc_size + 1) * self.pairs_acc_size / 2) * self.cell_matrix_size
        self.end_cell_i = row_index[-1]
        self.pairs_acc_row_is = cp.array(row_index, dtype=cp.uint32)
        self.pairs_acc = cp.zeros((pairs_acc_length,), dtype=DTYPE)

        # Number of processed samples
        self.n_samples = 0

    def update(self, xs):
        # Just curious, if this holds:
        assert CK_POWERS.max_threads_per_block == CK_PAIRS_UPDATE.max_threads_per_block
        assert xs.shape[1] == self.n_x

        # TODO DRY Scheduling sizes calculation.
        block_size = min(CK_POWERS.max_threads_per_block, self.n_x)
        n_blocks = self.n_x // block_size
        if self.n_x % block_size != 0:
            n_blocks += 1

        # TODO Update for whole batch in the same operation.
        for i in range(xs.shape[0]):
            x = cp.array(np.subtract(xs[i], self.x_bias))
            CK_POWERS((n_blocks,), (block_size,),
                      (self.max_single_pow, x, self.x_powers))  # grid (number of blocks), block and arguments
            print("> x_powers=\n", self.x_powers)

            # TODO DRY Scheduling sizes calculation.
            block_size = min(CK_PAIRS_UPDATE.max_threads_per_block, self.n_x)
            n_blocks = self.n_x // block_size
            if self.n_x % block_size != 0:
                n_blocks += 1

            print(f"Scheduling tasks as {n_blocks} x {block_size}")
            # Parameters are: grid (number of blocks), block size, followed by list of the kernel arguments.

            assert self.pairs_acc_row_is[-1] == self.end_cell_i
            CK_PAIRS_UPDATE((n_blocks,), (block_size,),
                            (self.n_x, self.max_single_pow, self.x_powers, self.x_powers_acc,
                             self.pairs_acc_cell_size,
                             self.end_cell_i, self.pairs_acc_row_is,
                             self.pairs_acc))
            self.n_samples += 1

    def moments(self, m):
        assert m[0] <= self.m[0] and m[1] <= self.m[1]

        # TODO (Implement) Generic case of (m1,m2)
        assert m == (1, 1), "Only covariance is supported. Generic case is not implemented yet."
        ms = cp.array(self.n_x)

        # TODO DRY Scheduling sizes calculation.
        block_size = min(CK_PAIRS_UPDATE.max_threads_per_block, self.n_x)
        n_blocks = self.n_x // block_size
        if self.n_x % block_size != 0:
            n_blocks += 1

        # const int m1, const int m2, const T n_samples, const int dim_x,
        # const int max_single_p, const T *x_powers_acc,
        # const int pair_cell_size, const int end_cell_i, const int *pair_acc_cell_row_is, const T *pair_acc,
        # T *moments
        CK_MOMENTS((n_blocks,), (block_size,),
                   (m[0], m[1], self.n_samples, self.n_x,
                    self.max_single_pow, self.x_powers_acc,
                    self.pairs_acc_cell_size,
                    self.end_cell_i, self.pairs_acc_row_is,
                    self.pairs_acc,
                    ms))
        return ms


def mean(batch_xs):
    return np.mean(batch_xs, 0)


def test_acc_update_1():
    nx = 3
    acc = MomentsGpu((1, 2), nx)
    x = (np.arange(nx, dtype=DTYPE) + 1).reshape((1, nx))
    acc.update(x)
    expected_pairs = np.array([[2., 2., 0., 0.],
                               [4., 4., 0., 0.],
                               [3., 3., 6., 12.],
                               [9., 9., 18., 36.]])
    actual_pairs = acc_to_2d_numpy(acc.pairs_acc_size, acc.pairs_acc_cell_size, cp.asnumpy(acc.pairs_acc))
    assert np.array_equal(expected_pairs, actual_pairs)

    expected_powers = np.array([[1., 1., 1.],
                                [2., 4., 8.],
                                [3., 9., 27.]])
    assert np.array_equal(expected_powers, cp.asnumpy(acc.x_powers_acc))

    # TODO Test correlation.
    # correlations = acc.moments((1, 1))
    # expected_correlations = np.array([0, 0, 0])
    # assert np.array_equal(expected_correlations, cp.asnumpy(correlations))


def test_acc_update_detrend():
    nx = 3
    n_samples = 5
    batch_x = (cp.arange(nx * n_samples, dtype=DTYPE) + 1).reshape((5, nx))
    print("batch:\n", batch_x)
    mean = np.mean(batch_x, 0)
    acc = MomentsGpu((1, 2), nx, x_bias=mean)
    acc.update(batch_x)
    acc.update(batch_x)
    acc.update(batch_x)
    expected_pairs = np.array([[2., 2., 0., 0.],
                               [4., 4., 0., 0.],
                               [3., 3., 6., 12.],
                               [9., 9., 18., 36.]])
    actual_pairs = acc_to_2d_numpy(acc.pairs_acc_size, acc.pairs_acc_cell_size, cp.asnumpy(acc.pairs_acc))
    assert np.array_equal(expected_pairs, actual_pairs)

    expected_powers = np.array([[1., 1., 1.],
                                [2., 4., 8.],
                                [3., 9., 27.]])
    assert np.array_equal(expected_powers, cp.asnumpy(acc.x_powers_acc))

    # TODO Test correlation.
    # correlations = acc.moments((1, 1))
    # expected_correlations = np.array([0, 0, 0])
    # assert np.array_equal(expected_correlations, cp.asnumpy(correlations))


def lab():
    # https://docs.cupy.dev/en/stable/user_guide/performance.html
    nx = 5000
    m = 4, 4
    acc = MomentsGpu(m, nx)
    x = (cp.arange(nx, dtype=DTYPE) + 1).reshape((1, nx))
    print("> x=\n", x)
    acc.update(x)

    cp.cuda.get_current_stream().synchronize()
    print(f"n={nx}, p={acc.max_single_pow}")
    pairs_acc = cp.asnumpy(acc.pairs_acc)
    print("> acc=\n", pairs_acc)
    pairs_acc_np = acc_to_2d_numpy(acc.pairs_acc_size, acc.pairs_acc_cell_size, pairs_acc)
    print("> acc=\n", pairs_acc_np)


print("\n=== TESTS ===")
test_acc_update_1()
test_acc_update_1()  # Looking for memory conflicts. Second call should not fail :)
test_acc_update_detrend()
print("TESTS OK\n")

# lab()
print("\nDONE")

