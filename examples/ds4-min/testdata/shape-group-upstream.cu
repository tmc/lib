__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) return;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t n_total_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[n_total_expert] = sum;
    }
}

__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) return;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}
