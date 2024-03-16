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

print(cp.cuda.Device(), cp.cuda.device.get_compute_capability(), cp.cuda.get_current_stream())

with open('gpu_acc.cu', 'r') as f:
    gpu_acc_source = f.read()
GPU_ACC_MODULE = cp.RawModule(code=gpu_acc_source)

CK_POWERS = GPU_ACC_MODULE.get_function('powers')
print("ck_powers:", CK_POWERS.attributes)

CK_PAIRS_UPDATE = GPU_ACC_MODULE.get_function('pairs_update')
print("ck_pairs_update:", CK_PAIRS_UPDATE.attributes)

CK_MOMENTS = GPU_ACC_MODULE.get_function('moments')


class MomentsGpu:
    def __init__(self, m: tuple[int, int], n_x: int):
        # Required max moment's power degrees
        self.m = m
        # Number of scalar measurements in a sample X
        self.n_x = n_x
        # Max power degree to calc for each variable. max(m) for off-diagonal, sum(m) on the diagonal
        # Using sum to get all the values necessary for later calculations, as max(m) <= sum(m)
        self.max_single_p = sum(m)
        # Sums of single measurement powers
        self.x_powers = cp.zeros((n_x, self.max_single_p), dtype=DTYPE)
        # Accumulates power sums of each variable
        self.x_powers_acc = cp.zeros((n_x, self.max_single_p), dtype=DTYPE)
        assert self.x_powers.size == self.n_x * self.max_single_p
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
        self.pairs_acc_row_indexes = cp.array(row_index, dtype=cp.uint32)
        self.pairs_acc = cp.zeros((pairs_acc_length,), dtype=DTYPE)

        # Number of processed samples
        self.n_updates = 0

    def update(self, x):
        # Just curious, if it holds
        assert CK_POWERS.max_threads_per_block == CK_PAIRS_UPDATE.max_threads_per_block
        assert x.size == self.n_x

        # TODO DRY Scheduling sizes calculation.
        block_size = min(CK_POWERS.max_threads_per_block, self.n_x)
        n_blocks = self.n_x // block_size
        if self.n_x % block_size != 0:
            n_blocks += 1
        print(n_blocks, block_size)

        CK_POWERS((n_blocks,), (block_size,),
                  (self.max_single_p, x, self.x_powers))  # grid (number of blocks), block and arguments
        print("> x_powers=\n", self.x_powers)

        # TODO DRY Scheduling sizes calculation.
        block_size = min(CK_PAIRS_UPDATE.max_threads_per_block, self.n_x)
        n_blocks = self.n_x // block_size
        if self.n_x % block_size != 0:
            n_blocks += 1

        print(f"Scheduling tasks as {n_blocks} x {block_size}")
        # Parameters are: grid (number of blocks), block size, followed by list of the kernel arguments.

        assert self.pairs_acc_row_indexes[-1] == self.end_cell_i
        CK_PAIRS_UPDATE((n_blocks,), (block_size,),
                        (self.n_x, self.max_single_p, self.x_powers, self.x_powers_acc,
                         self.pairs_acc_cell_size,
                         self.end_cell_i, self.pairs_acc_row_indexes,
                         self.pairs_acc))

    def moments(self, m):
        assert m[0] <= self.m[0] and m[1] <= self.m[1]

        # TODO Implement generic case.
        assert m == (1, 1)
        ms = cp.array(self.n_x)

        # TODO DRY Scheduling sizes calculation.
        block_size = min(CK_PAIRS_UPDATE.max_threads_per_block, self.n_x)
        n_blocks = self.n_x // block_size
        if self.n_x % block_size != 0:
            n_blocks += 1

        CK_MOMENTS((n_blocks,), (block_size,), (m[0], m[1], ms))

        return ms


def test_acc_update_1():
    nx = 3
    acc = MomentsGpu((1, 2), nx)
    x = cp.arange(nx, dtype=DTYPE) + 1
    print(f"> acc.pairs_acc.shape= {acc.pairs_acc.shape}")
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

    # correlations = acc.moments((1, 1))
    # expected_correlations = np.array([[0, 0, 0],
    #                                   [0, 0, 0],
    #                                   [0, 0, 0]])
    # assert np.array_equal(expected_correlations, cp.asnumpy(correlations))


def lab():
    # https://docs.cupy.dev/en/stable/user_guide/performance.html
    nx = 3
    m = 1, 2
    acc = MomentsGpu(m, nx)
    x = cp.arange(nx, dtype=DTYPE) + 1
    print("> x=\n", x)
    acc.update(x)

    cp.cuda.get_current_stream().synchronize()
    print(f"n={nx}, p={acc.max_single_p}")
    pairs_acc = cp.asnumpy(acc.pairs_acc)
    print("> acc=\n", pairs_acc)
    pairs_acc_np = acc_to_2d_numpy(acc.pairs_acc_size, acc.pairs_acc_cell_size, pairs_acc)
    print("> acc=\n", pairs_acc_np)


print("\n=== TESTS ===")
test_acc_update_1()
# cp.cuda.get_current_stream().synchronize()
test_acc_update_1()  # Looking for conflicts. Second call should not fail :)
print("TESTS OK\n")

lab()
print("\nDONE")
