package cudablock

// topkParallelSource is a fixed candidate, not a qualified faster kernel.
// Exactly 128 threads cooperate on each row using 1152 shared bytes. Each
// selected index is excluded on the next pass. Numeric comparisons preserve
// infinities, signed-zero source bits and lowest original index ties.
const topkParallelSource = `
extern "C" __global__ void topk_fp32(
    const float* input, const int* lengths, const int* offsets,
    int* indices, float* values, int* counts, int* status,
    unsigned long long row_stride, int width, int k, int rows, int want_values) {
    int row = (int)blockIdx.x, tid = (int)threadIdx.x;
    if (row >= rows) return;
    __shared__ float candidate_value[128];
    __shared__ int candidate_index[128];
    __shared__ int selected[32];
    unsigned long long out = (unsigned long long)row * k;
    int n = lengths[row], offset = offsets[row];
    if (tid == 0) {
        status[row] = 0; counts[row] = 0;
        for (int j = 0; j < k; ++j) {
            indices[out + j] = -1;
            if (want_values) values[out + j] = __int_as_float(0xff800000);
        }
        if (n < 0 || n > width) status[row] = 2;
        else if (offset < 0 || (n > 0 && (long long)offset + n - 1 > 2147483647LL)) status[row] = 3;
    }
    __syncthreads();
    if (status[row]) return;
    unsigned long long base = (unsigned long long)row * row_stride;
    int has_nan = 0;
    for (int i = tid; i < n; i += 128) {
        float v = input[base + i];
        has_nan |= v != v;
    }
    candidate_index[tid] = has_nan;
    __syncthreads();
    for (int step = 64; step > 0; step /= 2) {
        if (tid < step) candidate_index[tid] |= candidate_index[tid + step];
        __syncthreads();
    }
    if (tid == 0 && candidate_index[0]) status[row] = 1;
    __syncthreads();
    if (status[row]) return;
    int count = n < k ? n : k;
    for (int j = 0; j < count; ++j) {
        int best = -1;
        float best_value = __int_as_float(0xff800000);
        for (int i = tid; i < n; i += 128) {
            bool used = false;
            for (int p = 0; p < j; ++p) used |= selected[p] == i;
            float v = input[base + i];
            if (!used && (best < 0 || v > best_value || (v == best_value && i < best))) {
                best = i; best_value = v;
            }
        }
        candidate_index[tid] = best;
        candidate_value[tid] = best_value;
        __syncthreads();
        for (int step = 64; step > 0; step /= 2) {
            if (tid < step) {
                int a = candidate_index[tid], b = candidate_index[tid + step];
                float av = candidate_value[tid], bv = candidate_value[tid + step];
                if (b >= 0 && (a < 0 || bv > av || (bv == av && b < a))) {
                    candidate_index[tid] = b; candidate_value[tid] = bv;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            selected[j] = candidate_index[0];
            indices[out + j] = selected[j] + offset;
            if (want_values) values[out + j] = candidate_value[0];
        }
        __syncthreads();
    }
    if (tid == 0) counts[row] = count;
}
`
