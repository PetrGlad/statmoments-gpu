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
void pairs_update(const int dim_x,
                  const int max_single_p, const T *x_powers, T *x_powers_acc,
                  const int pair_cell_size, const int end_cell_i, const int *pair_acc_cell_row_i, T *pair_acc) {
    assert(0 < dim_x);
    assert(0 < max_single_p);
    assert(0 <= end_cell_i);
    assert(0 < pair_cell_size);
    // Using `_i` as shorthand for "index", `i` alone means row index.
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
//    printf("<%d (b=%d, t=%d)> dim_x=%d max_single_p=%d pair_sell_size=%d\dim_x", tid, blockIdx.x, threadIdx.x, dim_x, max_single_p,
//           pair_cell_size);
    const int row_i = tid;

    // Update a diagonal cell.
    for (int p = 0; p < max_single_p; p++) {
        // We could sum the whole in one go, but then it would be a step that have to be scheduled separately.
        // Alternatively this loop can be a part of powers calculation step.
        // These are options to consider later (need some benchmarks to decide which one is better).
        const int i = row_i * max_single_p + p;
//        printf("|powers max_p=%d i=%d, x_powers[i]=%f\dim_x", max_single_p, i, x_powers[i]);
        x_powers_acc[i] += x_powers[i];
    }

    if (row_i > 0) { // Skip first diagonal pair
        // Calculate pair powers.
        const int next_row_cell_i = pair_acc_cell_row_i[row_i];
        printf("end_cell_i=%d next_row_cell_i=%d\n", end_cell_i, next_row_cell_i);
        assert(0 <= next_row_cell_i && next_row_cell_i <= end_cell_i);
        int cell_i = pair_acc_cell_row_i[row_i - 1];
        assert(0 <= cell_i && cell_i <= next_row_cell_i);
        // TODO (scheduling) Updating a single row of the accumulator matrix for now.
        //      Should be a contiguous range of rows.
//    const int cell_mat_size = pair_cell_size * pair_cell_size;
//    printf("|before cell_i %d, next_row_cell_i %d\dim_x", cell_i, next_row_cell_i);
        // Only non diagonal pairs are kept in pait acc
        // That means one of the indices should start with 1
        // +1 to exclude diagonal (it is calculated separately).
        const int powers_a_start_i = row_i * max_single_p;
        int powers_b_start_i = 0;
        // For each pair cell of the triangle matrix row.
        // Updating as lower triangle of cells (excluding diagonal).
        // (col_i - current cell column. It is here for debugging, can be removed)
        printf("sizeof(pair_acc) = %d   sizeof(*pair_acc) = %d\n", sizeof(pair_acc), sizeof(*pair_acc));
        for (int col_i = 0; cell_i < next_row_cell_i; col_i++) {
//        printf("|row cell_i %d, col_i %d\dim_x", cell_i, col_i);
            for (int pa = 0; pa < pair_cell_size; pa++) {
                for (int pb = 0; pb < pair_cell_size; pb++) {
//                printf("|cell tid %d, cell_i %d, col_i %d, pa %d, pb %d, xp1 %f, xp2 %f\dim_x",
//                       tid, cell_i, col_i, pa, pb, x_powers[powers_a_start_i + pa], x_powers[powers_b_start_i + pb]);
                    printf("cell_i=%d next_row_cell_i=%d\n", cell_i, next_row_cell_i);
                    assert(cell_i < end_cell_i);
                    pair_acc[cell_i] += x_powers[powers_a_start_i + pa] * x_powers[powers_b_start_i + pb];
                    cell_i++;
                }
            }
            powers_b_start_i += max_single_p;
        }
    }
}


//extern "C" __global__ inline
//void single_sum(const int max_single_p, T *x_powers_acc, const int i, const int p) {
//    return x_powers_acc[max_single_p * i + p];
//}

extern "C" __global__
void moments(const int m1, const int m2, const T count, const int dim_x,
             const int max_single_p, T *x_powers_acc,
             const int pair_cell_size, const int *pair_acc_row_indexes, T *pair_acc, T *moments) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    printf("<<<moments(...) is not implemented>>>");
    int single_powers_a_i = tid * max_single_p;
    int single_powers_b_i = max_single_p;

    const int cell_mat_length = pair_cell_size * pair_cell_size;
    const int row_i = tid;
    const int pair_powers_i = row_i * pair_cell_size;
    T ei = 0;
    const int next_acc_row_i = pair_acc_row_indexes[row_i + 1];
    int acc_i = pair_acc_row_indexes[row_i];
    // TODO (scheduling) Updating a single row of the accumulator matrix for now.
    //      Should be a contiguous range of rows.
//    const int cell_mat_size = pair_cell_size * pair_cell_size;
//    printf("|before acc_i %d, next_acc_row_i %d\n", acc_i, next_acc_row_i);
    // +1 to exclude diagonal (it is calculated separately).
    const int powers_a_start_i = (row_i + 1) * max_single_p;
    int powers_b_start_i = 0;
    // For each pair cell of the triangle matrix row.
    // Updating as lower triangle of cells (excluding diagonal).
    // (col_i - current cell column. It is here for debugging, can be removed)
//    for (int col_i = 0; acc_i < next_acc_row_i; col_i++) {
//        T* powers_a = x_powers_acc[powers_a_start_i + pa];
//        x_powers_acc[powers_b_start_i + pb];
//
//        powers_b_start_i += max_single_p;
//    }
}
