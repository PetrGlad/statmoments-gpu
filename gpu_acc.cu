typedef double T;

extern "C" __global__
void powers(const int p, const T *x, T *y) {
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

extern "C" __global__
void pairs_update(const int n,
                  const int max_single_p, const T *x_powers, T *x_powers_acc,
                  const int cell_size, const int *acc_row_indexes, T *acc) {
    // Using `_i` as shorthand for "index", `i` alone means row index.
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("<%d (b=%d, t=%d)> # %d %d \n", tid, blockIdx.x, threadIdx.x, n, max_single_p);
    const int cell_mat_size = cell_size * cell_size;
    const int row_i = tid;
    const int next_row_i = acc_row_indexes[row_i + 1];
    int acc_i = acc_row_indexes[row_i];
    int col_i = 0;
    // TODO (scheduling) Updating a single row of the accumulator matrix for now.
    //      Should be a contiguous range of rows.
    const int powers_idx = row_i * cell_size;
    // Update a diagonal cell.
    for (int p = 0; p < max_single_p; p++) {
        // We could sum the whole in one go, but then it would be a step that have to be scheduled separately.
        // Alternatively this loop can be a part of powers calculation step.
        // These are options to consider later (need some benchmarks to decide which one is better).
        x_powers_acc[powers_idx * max_single_p + p] += x_powers[powers_idx * max_single_p + p];
    }
    // Calculate pair powers.
    int powers_idx_b = 0;
    printf("|before acc_i %d, next_row_i %d\n", acc_i, next_row_i);
    // +1 to exclude diagonal (calculated separately).
    const int powers_idx_a = powers_idx + 1;
    // For each pair cell of the triangle matrix row.
    // Updating as lower triangle of cells (excluding diagonal).
    for (; acc_i < next_row_i; col_i++) {
        printf("|row acc_i %d, col_i %d\n", acc_i, col_i);
        for (int pa = 0; pa < cell_size; pa++) {
            for (int pb = 0; pb < cell_size; pb++) {
                printf("|cell tid %d, acc_i %d, col_i %d, pa %d, pb %d, xp1 %f, xp2 %f\n",
                       tid, acc_i, col_i, pa, pb, x_powers[powers_idx_a + pa], x_powers[powers_idx_b + pb]);
                acc[acc_i] += x_powers[powers_idx_a + pa] * x_powers[powers_idx_b + pb];
                acc_i++;
            }
        }
        powers_idx_b += max_single_p;
    }
}