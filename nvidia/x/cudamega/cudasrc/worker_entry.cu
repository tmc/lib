
// start_order is the instrument, not part of the model: start_order[0] is a
// ticket counter and start_order[1 + cta*task_count + task] is the ticket the
// CTA held when it took that task, or zero for a task that CTA never took.
// Tickets are dealt by one device-scope atomic add, so they are a total order
// over task starts across the whole grid and can be compared between CTAs.
extern "C" __global__ void dag_worker(const unsigned *input, unsigned *scratch,
    unsigned *completed, unsigned long long *start_order, volatile unsigned *abort_code,
    unsigned n, unsigned long long epoch) {
  if (gridDim.x != grid_ctas || gridDim.y != 1 || gridDim.z != 1 || blockDim.x != grid_threads || blockDim.y != 1 || blockDim.z != 1) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, %[1]du);
    return;
  }
  if (!n || !epoch) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, %[2]du);
    return;
  }
