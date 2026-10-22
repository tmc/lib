#include <math.h>
#include <cuda_fp16.h>

// Actual admitted W1 row, selected by the previous proposal (sequential
// Markov chain). Q8_0 block scale is stored half followed by 32 signed codes.
extern "C" __global__ void ds4_dspark_markov_row(
    const unsigned char *weights, float *state, const int *proposal,
    unsigned int initial, unsigned int row, unsigned int rank,
    unsigned int type) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= rank) return;
    unsigned int token = row == 0 ? initial : (unsigned int)proposal[row - 1];
    unsigned long long offset = (unsigned long long)token * rank;
    if (type == 0) state[d] = ((const float *)weights)[offset + d];
    else if (type == 1) state[d] = __half2float(((const __half *)weights)[offset + d]);
    else {
        const unsigned char *block = weights + (offset / 32 + d / 32) * 34;
        float scale = __half2float(*(const __half *)block);
        state[d] = scale * (float)((const signed char *)(block + 2))[d % 32];
    }
}

extern "C" __global__ void ds4_dspark_markov_argmax(
    const float *logits, const float *bias, int *proposal,
    unsigned int *status, unsigned int row, unsigned int vocab) {
    __shared__ float values[256];
    __shared__ unsigned int ids[256];
    unsigned int tid = threadIdx.x;
    float best = -INFINITY;
    unsigned int token = 0;
    for (unsigned int i = tid; i < vocab; i += 256) {
        float value = logits[i] + bias[i];
        if (!isfinite(value)) atomicOr(status, 1u << row);
        if (value > best || (value == best && i < token)) { best = value; token = i; }
    }
    values[tid] = best; ids[tid] = token;
    __syncthreads();
    for (unsigned int stride = 128; stride != 0; stride /= 2) {
        if (tid < stride) {
            float value = values[tid + stride];
            unsigned int id = ids[tid + stride];
            if (value > values[tid] || (value == values[tid] && id < ids[tid])) {
                values[tid] = value; ids[tid] = id;
            }
        }
        __syncthreads();
    }
    if (tid == 0) proposal[row] = (int)ids[0];
}

extern "C" __global__ void ds4_dspark_confidence_input(
    const float *hidden, const float *state, float *features,
    unsigned int rank) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < 4096) features[d] = hidden[d];
    else if (d < 4096 + rank) features[d] = state[d - 4096];
}

// A zero threshold is the upstream confidence-disabled policy. All Markov
// tokens were computed sequentially; confidence selects a leading prefix.
// Argmax records one error bit per row. Errors beyond the first rejected
// confidence row must not invalidate the prefix that upstream returns.
extern "C" __global__ void ds4_dspark_confidence_prefix(
    float *confidence, unsigned int *length, unsigned int *status,
    unsigned int count, float threshold) {
    if (threadIdx.x != 0) return;
    unsigned int accepted = count;
    unsigned int row_errors = *status;
    unsigned int error = 0;
    for (unsigned int row = 0; row < count; row++) {
        float value = confidence[row];
        if (!isfinite(value)) { error = 2u; accepted = 0; break; }
        float probability = 1.0f / (1.0f + expf(-value));
        confidence[row] = probability;
        if (threshold > 0.0f && probability < threshold) { accepted = row; break; }
        if (row_errors & (1u << row)) { error = 1u; accepted = 0; break; }
    }
    *status = error;
    *length = accepted;
}
