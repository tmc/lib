package cudablock

import "github.com/tmc/lib/nvidia/x/cudakernel"

// TopKSource is an independently written ordinary CUDA repeated-selection
// baseline. One thread processes each row; no TMA or cluster support is assumed.
// DeepSelect bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18 was consulted for its
// interface and short-row caveats; no implementation was copied.
const TopKSource = `
extern "C" __global__ void topk_fp32(
    const float* input, const int* lengths, const int* offsets,
    int* indices, float* values, int* counts, int* status,
    unsigned long long row_stride, int width, int k, int rows, int want_values) {
    int row = (int)blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    unsigned long long out = (unsigned long long)row * k;
    status[row] = 0;
    counts[row] = 0;
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
        float v = input[base + i];
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
            float v = input[base + i];
            if (!used && (best < 0 || v > best_value || (v == best_value && i < best))) {
                best = i;
                best_value = v;
            }
        }
        selected[j] = best;
        indices[out + j] = best + offset;
        if (want_values) values[out + j] = best_value;
    }
    counts[row] = count;
}
`

// TopKSchema describes topk_fp32's fixed CUDA parameter ABI.
func TopKSchema() (cudakernel.Schema, error) {
	return cudakernel.NewSchema(
		cudakernel.Param{Name: "input", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "lengths", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "offsets", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "indices", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "values", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "counts", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "status", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "row_stride", Type: "u64"},
		cudakernel.Param{Name: "width", Type: "s32"},
		cudakernel.Param{Name: "k", Type: "s32"},
		cudakernel.Param{Name: "rows", Type: "s32"},
		cudakernel.Param{Name: "want_values", Type: "s32"},
	)
}
