package cudablock

// TopKBF16Source implements ordinary selection from BF16 storage. Conversion
// copies each uint16 into the high half of an FP32 word; it never rounds or
// reinterprets input as FP16. Optional values are FP32 decoded source values.
const TopKBF16Source = `
extern "C" __global__ void topk_bf16(
    const unsigned short* input, const int* lengths, const int* offsets,
    int* indices, float* values, int* counts, int* status,
    unsigned long long row_stride, int width, int k, int rows, int want_values) {
    int row = (int)blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    unsigned long long out = (unsigned long long)row * k;
    status[row] = 0; counts[row] = 0;
    for (int j = 0; j < k; ++j) {
        indices[out + j] = -1;
        if (want_values) values[out + j] = __int_as_float(0xff800000);
    }
    int n = lengths[row], offset = offsets[row];
    if (n < 0 || n > width) { status[row] = 2; return; }
    if (offset < 0 || (n > 0 && (long long)offset + n - 1 > 2147483647LL)) {
        status[row] = 3; return;
    }
    unsigned long long base = (unsigned long long)row * row_stride;
    for (int i = 0; i < n; ++i) {
        float v = __uint_as_float((unsigned int)input[base + i] << 16);
        if (v != v) { status[row] = 1; return; }
    }
    int selected[32];
    int count = n < k ? n : k;
    for (int j = 0; j < count; ++j) {
        int best = -1;
        float best_value = __int_as_float(0xff800000);
        for (int i = 0; i < n; ++i) {
            bool used = false;
            for (int p = 0; p < j; ++p) used |= selected[p] == i;
            float v = __uint_as_float((unsigned int)input[base + i] << 16);
            if (!used && (best < 0 || v > best_value || (v == best_value && i < best))) {
                best = i; best_value = v;
            }
        }
        selected[j] = best;
        indices[out + j] = best + offset;
        if (want_values) values[out + j] = best_value;
    }
    counts[row] = count;
}
`
