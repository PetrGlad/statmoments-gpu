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
max_p = sum(m)

ck_powers = cp.RawKernel(
    r"""
  typedef double T;
  
  extern "C" __global__
  void powers(const int p, const T* x, T* y) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const T z = x[tid];
    T c = z;
    int n = p;
    int i = tid * p;
    while (true) {
        // printf("<%d (b=%d, t=%d)> [%d] at %d = %f # %d\n", tid, blockIdx.x, threadIdx.x, i, n, c, p);
        y[i++] = c;
        if (n <= 1) break;
        n--;
        c *= z;
    }
  }
  """,
    "powers",
)

print(ck_powers.attributes)

# ---------- X POWERS ----------

x = cp.arange(n, dtype=dtype) + 1
print("> x=\n", x)
x_powers = cp.zeros((n, max_p), dtype=dtype)

block_size = min(ck_powers.max_threads_per_block, n)
n_blocks = n // block_size
if n % block_size != 0:
    n_blocks += 1
print(n_blocks, block_size)

ck_powers((n_blocks,), (block_size,), (max_p, x, x_powers))  # grid (number of blocks), block and arguments
print("> x_powers=\n", x_powers)


def row_starts():
    p = 0
    i = 0
    while True:
        i += p
        yield i
        p += 1


ck_pairs_update = cp.RawKernel(
    r"""
  typedef double T;
  
  extern "C" __global__
  void pairs_update(const int n,
                    const int max_p, const T* x_powers, const T* x_powers_acc, 
                    const int cell_size, const int* acc_row_indexes, T* acc) {
    // Using `_i` as shorthand for "index", `i` alone means row index.
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("<%d (b=%d, t=%d)> # %d %d \n", tid, blockIdx.x, threadIdx.x, n, p);
    const int cell_mat_size = cell_size * cell_size;
    const int row_i = tid;
    const int next_row_i = acc_row_indexes[row_i + 1];
    int acc_i = acc_row_indexes[row_i];
    int col_i = 0;
    // TODO (scheduling) Updating a single row of the accumulator matrix for now. 
    //      Should be a contiguous range of rows
    const int powers_idx_a = row_i * cell_size;
    int powers_idx_b = 0;
    // For each pair cell of the triangle matrix row.
    printf("|before acc_i %d, next_row_i %d\n", acc_i, next_row_i);
    for (; acc_i < next_row_i; col_i++) {
      printf("|row acc_i %d, col_i %d\n", acc_i, col_i);
      
      // TODO Also update powers acc here
      
      for (int pa = 0; pa < cell_size; pa++) {
        for (int pb = 0; pb < cell_size; pb++) {
          printf("|cell tid %d, acc_i %d, col_i %d, pa %d, pb %d, xp1 %f, xp2 %f\n",
                 tid, acc_i, col_i, pa, pb, x_powers[powers_idx_a + pa],  x_powers[powers_idx_b + pb]);
          acc[acc_i] += x_powers[powers_idx_a + pa] * x_powers[powers_idx_b + pb];
          acc_i++;
        }
      }
      
      FIXME 1.Reduce matrix acc size, 1. Update powers vector acc.
      
      powers_idx_b += max_p;
    }
  }
  """,
    "pairs_update",
)


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

# n_acc_cols = n
# n_acc_rows = math.ceil(n / 2)

# Accumulates power sums of each variable
x_powers_acc = cp.zeros((n, max_p), dtype=dtype)

starts = row_starts()
cell_size = max(m)
cell_matrix_size = cell_size * cell_size
row_index = np.array([next(starts) * cell_matrix_size
                      for _ in range(n + 1)], dtype=np.uint32)
print("> row_index=\n", row_index)
assert row_index[-1] == math.floor((n + 1) * n / 2) * cell_matrix_size

acc_row_indexes = cp.array(row_index, dtype=cp.uint32)
# TODO Exclude diagonal acc size only needs to be n-1 since we're
#  accumulating single var powers separately in x_powers_acc.
acc = cp.zeros((row_index[-1],), dtype=dtype)

block_size = min(ck_pairs_update.max_threads_per_block, n)
n_blocks = n // block_size
if n % block_size != 0:
    n_blocks += 1

print(n_blocks, block_size)
# grid (number of blocks), block size, followed by list user of arguments
ck_pairs_update((n_blocks,), (block_size,),
                (n, max_p, x_powers, x_powers_acc, cell_size, acc_row_indexes, acc))

print(f"n={n}, p={max_p}")
print("> acc=\n", acc)
print("> acc=\n", acc_to_2d_numpy(n, cell_size, acc))

print("\nDONE")
