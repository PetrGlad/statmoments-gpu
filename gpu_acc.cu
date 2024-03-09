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
                  const int pair_cell_size, const int *pair_acc_row_indexes, T *pair_acc) {
    // Using `_i` as shorthand for "index", `i` alone means row index.
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("<%d (b=%d, t=%d)> n=%d max_single_p=%d pair_sell_size=%d\n", tid, blockIdx.x, threadIdx.x, n, max_single_p,
           pair_cell_size);
    const int row_i = tid;

    // Update a diagonal cell.
    for (int p = 0; p < max_single_p; p++) {
        // We could sum the whole in one go, but then it would be a step that have to be scheduled separately.
        // Alternatively this loop can be a part of powers calculation step.
        // These are options to consider later (need some benchmarks to decide which one is better).
        const int i = row_i * max_single_p + p;
//        printf("|powers max_p=%d i=%d, x_powers[i]=%f\n", max_single_p, i, x_powers[i]);
        x_powers_acc[i] += x_powers[i];
    }

    // Calculate pair powers.
    const int next_acc_row_i = pair_acc_row_indexes[row_i + 1];
    int acc_i = pair_acc_row_indexes[row_i];
    // TODO (scheduling) Updating a single row of the accumulator matrix for now.
    //      Should be a contiguous range of rows.
//    const int cell_mat_size = pair_cell_size * pair_cell_size;
    printf("|before acc_i %d, next_acc_row_i %d\n", acc_i, next_acc_row_i);
    // +1 to exclude diagonal (it is calculated separately).
    const int powers_a_start_i = (row_i + 1) * max_single_p;
    int powers_b_start_i = 0;
    // For each pair cell of the triangle matrix row.
    // Updating as lower triangle of cells (excluding diagonal).
    // (col_i - current cell column. It is here for debugging, can be removed)
    for (int col_i = 0; acc_i < next_acc_row_i; col_i++) {
//        printf("|row acc_i %d, col_i %d\n", acc_i, col_i);
        for (int pa = 0; pa < pair_cell_size; pa++) {
            for (int pb = 0; pb < pair_cell_size; pb++) {
                printf("|cell tid %d, acc_i %d, col_i %d, pa %d, pb %d, xp1 %f, xp2 %f\n",
                       tid, acc_i, col_i, pa, pb, x_powers[powers_a_start_i + pa], x_powers[powers_b_start_i + pb]);
                pair_acc[acc_i] += x_powers[powers_a_start_i + pa] * x_powers[powers_b_start_i + pb];
                acc_i++;
            }
        }
        powers_b_start_i += max_single_p;
    }
}


//extern "C" __global__ inline
//void single_sum(const int max_single_p, T *x_powers_acc, const int i, const int p) {
//    return x_powers_acc[max_single_p * i + p];
//}

extern "C" __global__
void moments(const int m1, const int m2, const int n,
             const int max_single_p, T *x_powers_acc,
             const int pair_cell_size, const int *acc_row_indexes, T *pair_acc, T *moments) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // Not implemented
    printf("<<<moments(...) is not implemented>>>");
    const int cell_mat_size = pair_cell_size * pair_cell_size;
    const int row_i = tid;
    const int powers_idx = row_i * pair_cell_size;
    T ei = 0;
//    for (int col_i = 0; acc_i < next_row_i; col_i++) {
//        T ej = 0;
//        moments[i * n + j]
//    }
}