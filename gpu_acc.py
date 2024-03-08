# https://docs.cupy.dev/en/stable/user_guide/kernel.html
import itertools as iter
import math

import cupy as cp
import numpy as np
import scipy as sp
import scipy.linalg.blas as sblas
import sympy as sy
from sympy.plotting import plot

print(cp.cuda.Device(), cp.cuda.device.get_compute_capability(), cp.cuda.get_current_stream())

# Floats type to use. Update CUDA kernels when changing this.
dtype = cp.float64
m = (1, 2)  # The moment's power degrees that we need
n = 3  # Number of scalar values in a measurement X
# Max power degree to calc for each variable. max(m) for off-diagonal, sum(m) on the diagonal
# Using sum to get all the values necessary for later calculations, as max(m) <= sum(m)
max_single_p = sum(m)

with open('gpu_acc.cu', 'r') as f:
    gpu_acc_source = f.read()
gpu_acc_module = cp.RawModule(code=gpu_acc_source)

# ---------- X POWERS ----------

ck_powers = gpu_acc_module.get_function('powers')

print(ck_powers.attributes)
x = cp.arange(n, dtype=dtype) + 1
print("> x=\n", x)
x_powers = cp.zeros((n, max_single_p), dtype=dtype)

block_size = min(ck_powers.max_threads_per_block, n)
n_blocks = n // block_size
if n % block_size != 0:
    n_blocks += 1
print(n_blocks, block_size)

ck_powers((n_blocks,), (block_size,), (max_single_p, x, x_powers))  # grid (number of blocks), block and arguments
print("> x_powers=\n", x_powers)


def row_starts():
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
    out = np.zeros((n_rows * cell_size, n_rows * cell_size), dtype=dtype)
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
    out = np.zeros((n_rows, n_rows, cell_size, cell_size), dtype=dtype)
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


# ---------- X PAIRS ----------

ck_pairs_update = gpu_acc_module.get_function('pairs_update')

# n_acc_cols = n
# n_acc_rows = math.ceil(n / 2)

# Accumulates power sums of each variable
x_powers_acc = cp.zeros((n, max_single_p), dtype=dtype)

starts = row_starts()
acc_cell_size = max(m)
# Excluding diagonal in acc size since we're accumulating single var powers separately in x_powers_acc.
acc_size = n - 1
cell_matrix_size = acc_cell_size * acc_cell_size
row_index = np.array([next(starts) * cell_matrix_size
                      for _ in range(acc_size + 1)], dtype=np.uint32)
print("> row_index=\n", row_index)

acc_row_indexes = cp.array(row_index, dtype=cp.uint32)
# Excluding diagonal in acc size since we're accumulating single var powers separately in x_powers_acc.
acc = cp.zeros((row_index[-1],), dtype=dtype)

block_size = min(ck_pairs_update.max_threads_per_block, n)
n_blocks = n // block_size
if n % block_size != 0:
    n_blocks += 1

print(f"Scheduling tasks as {n_blocks} x {block_size}")
# Parameters are: grid (number of blocks), block size, followed by list of the kernel arguments.
assert x_powers.size == n * max_single_p
assert x_powers.shape == x_powers_acc.shape
assert row_index[-1] == math.floor((acc_size + 1) * acc_size / 2) * cell_matrix_size
assert acc_cell_size * acc_cell_size == cell_matrix_size
ck_pairs_update((n_blocks,), (block_size,),
                (n, max_single_p, x_powers, x_powers_acc, acc_cell_size, acc_row_indexes, acc))

print(f"n={n}, p={max_single_p}")
print("> acc=\n", acc)
print("> acc=\n", acc_to_2d_numpy(acc_size, acc_cell_size, acc))

print("\nDONE")
